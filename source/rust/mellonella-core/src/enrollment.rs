//! Speaker embedding pool: immutable enrollment anchors plus a single
//! λ-residual self-augmented embedding adapted from runtime audio.
//!
//! `EmbeddingPool` holds:
//!
//! * `anchors`  — written once by explicit enrollment, never deleted.
//!   Their element-wise mean (`anchor_centroid`) is the immutable
//!   reference `E_enroll`.
//! * `evidence` — a running EMA of high-confidence runtime embeddings
//!   (admitted candidates). `None` until the first admission.
//! * `adapted`  — the residual-anchored self-augmentation
//!   `λ·E_enroll + (1-λ)·evidence`, recomputed whenever `evidence`
//!   changes. This is what runtime scoring matches against alongside
//!   the centroid.
//!
//! ### Why λ-residual instead of a FIFO (#118)
//!
//! The previous design kept a `VecDeque` of up to N admitted
//! embeddings and, on drift, cleared the whole FIFO in one
//! all-or-nothing `maybe_reset`. That reset was a visible
//! discontinuity in the score scale. The λ-residual scheme — from
//! *Adaptive Speaker Embedding Self-Augmentation* (arXiv:2601.12769),
//! `E_aug = λ·E_enroll + (1-λ)·E_avg` — replaces both the FIFO and the
//! reset with one smoothly-updated embedding: the `λ·E_enroll` term is
//! a permanent residual pull toward enrollment, so the adapted
//! embedding can track the runtime acoustic condition without ever
//! drifting far from the enrolled identity, and there is no
//! discontinuity to reset.
//!
//! Admission is still gated by `can_auto_learn` (anchor-distance
//! check) on top of the caller's `should_admit_auto_learn` score / F0
//! / speech-duration gates.
//!
//! JSON persistence ([`EmbeddingPool::to_json`] / [`from_json`] /
//! [`save`](Self::save) / [`load`](Self::load)) is schema version 2.
//! Version 1 payloads (the old `auto_learn` FIFO) are still accepted
//! on load — the FIFO is migrated to `evidence` by taking its mean.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::gating::cos_similarity;

/// Persistence-schema version. Version 2 replaced the `auto_learn`
/// FIFO with the `evidence` EMA (#118). Version 1 payloads are still
/// accepted on load and migrated; any other value is rejected.
pub const PERSISTENCE_VERSION: u32 = 2;

/// Drift-safety + adaptation knobs consumed by [`EmbeddingPool`].
#[derive(Debug, Clone, Copy)]
pub struct EmbeddingPoolConfig {
    /// Maximum `1 - max_cos(anchors)` allowed for a candidate to be
    /// admitted into the adapted embedding.
    pub anchor_distance_threshold: f32,
    /// λ — residual pull toward the enrollment centroid in
    /// `adapted = λ·centroid + (1-λ)·evidence`. `0.0` lets the adapted
    /// embedding follow runtime evidence freely; `1.0` pins it to the
    /// enrollment centroid (no adaptation). Default `0.1`
    /// (arXiv:2601.12769) — a starting point pending calibration.
    pub adapt_residual_lambda: f32,
    /// η — EMA step for folding a newly-admitted candidate into the
    /// running `evidence`: `evidence = (1-η)·evidence + η·candidate`.
    /// Higher = faster tracking of the current acoustic condition,
    /// lower = steadier. Default `0.3`; a starting point pending
    /// calibration.
    pub adapt_rate: f32,
    /// When `true`, [`EmbeddingPool::bootstrap_seed`] is allowed to
    /// install the very first anchor from a runtime embedding instead
    /// of requiring an explicit enrollment recording. The streaming
    /// engine uses this to let the LADSPA / APO plugins start with an
    /// empty pool and learn the dominant speaker's profile as they
    /// hear it, without an upfront enrollment step.
    ///
    /// Default `false` — preserves the existing contract where empty
    /// pools never learn (`can_auto_learn` short-circuits to `false`).
    pub allow_bootstrap_from_runtime: bool,
}

impl Default for EmbeddingPoolConfig {
    fn default() -> Self {
        Self {
            anchor_distance_threshold: 0.4,
            adapt_residual_lambda: 0.1,
            adapt_rate: 0.3,
            allow_bootstrap_from_runtime: false,
        }
    }
}

/// F0 statistics captured from the explicit enrollment recording.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct EnrollmentMetadata {
    #[serde(default)]
    pub f0_mu: f32,
    #[serde(default)]
    pub f0_sigma: f32,
}

/// Speaker pool: immutable anchors + one λ-residual adapted embedding.
#[derive(Debug, Clone)]
pub struct EmbeddingPool {
    config: EmbeddingPoolConfig,
    anchors: Vec<Vec<f32>>,
    /// Running EMA of admitted runtime embeddings. `None` until the
    /// first admission.
    evidence: Option<Vec<f32>>,
    metadata: EnrollmentMetadata,
    /// Element-wise mean of `anchors` (`E_enroll`), recomputed whenever
    /// the anchor set changes. `None` while there are no anchors.
    anchor_centroid: Option<Vec<f32>>,
    /// `anchors` agglomerated into at most [`MAX_CONDITION_GROUPS`]
    /// acoustic conditions, one mean vector each. Recomputed whenever
    /// the anchor set changes; these — not the raw anchors — are what
    /// [`Self::match_score`] compares against. Empty while there are
    /// no anchors.
    condition_centroids: Vec<Vec<f32>>,
    /// Cached `λ·anchor_centroid + (1-λ)·evidence`, recomputed whenever
    /// `evidence` changes. `None` until the first admission. Cached so
    /// per-refresh [`Self::match_score`] does not re-blend the vector.
    adapted: Option<Vec<f32>>,
}

/// Upper bound on how many distinct acoustic conditions an enrollment
/// profile is modelled as holding (morning voice, evening voice, an
/// extra `+声紋を追加登録` registration, …).
pub const MAX_CONDITION_GROUPS: usize = 4;
/// Target anchors per condition group. Averaging this many 3 s windows
/// is what cancels the per-window noise that makes a single anchor an
/// easy target for an unrelated speaker.
pub const MIN_ANCHORS_PER_GROUP: usize = 4;
/// Two groups are the same acoustic condition only if their means are
/// at least this similar.
///
/// Measured intra-session anchor similarity is mean 0.85 / p10 0.80, so
/// 0.70 merges windows of one recording while leaving a genuinely
/// separate registration standing on its own while the profile remains
/// within the four-condition hard cap.
pub const CONDITION_MERGE_COS: f32 = 0.70;

/// Element-wise mean of `vectors`, or `None` for an empty slice.
///
/// For a single vector the mean is that vector verbatim (multiplying
/// each f32 by `1.0` is exact), so a single-anchor pool scores
/// byte-identically to the pre-#117 `cos_sim_max` path.
fn elementwise_mean(vectors: &[Vec<f32>]) -> Option<Vec<f32>> {
    let dim = vectors.first()?.len();
    let mut sum = vec![0.0f32; dim];
    for v in vectors {
        debug_assert_eq!(v.len(), dim, "embeddings must share a dimension");
        for (s, &x) in sum.iter_mut().zip(v.iter()) {
            *s += x;
        }
    }
    // Vector count is a handful (5-10 enrollment windows), exact in f32.
    #[allow(clippy::cast_precision_loss)]
    let inv = 1.0 / vectors.len() as f32;
    for s in &mut sum {
        *s *= inv;
    }
    Some(sum)
}

/// Group `anchors` into acoustic conditions and return one mean vector
/// per group.
///
/// Agglomerative: start with one group per anchor and repeatedly merge
/// the closest pair (by cosine between group means), subject to two
/// stopping rules that pull in opposite directions and are both load
/// bearing:
///
/// * **the budget** — `anchors / MIN_ANCHORS_PER_GROUP` capped at
///   [`MAX_CONDITION_GROUPS`] — forces the windows of one recording to
///   collapse into a few averaged means, which is what removes the
///   per-anchor lottery an impostor otherwise wins on one noisy window.
/// * **[`CONDITION_MERGE_COS`]** stops the *soft* budget from
///   steamrollering a genuinely separate registration into the average
///   of another one. The floor is waived only while more than
///   [`MAX_CONDITION_GROUPS`] groups remain: the hard cap is the security
///   property that prevents a noisy but still accepted enrollment from
///   recreating the raw-anchor lottery. Profiles with four or fewer
///   distinct conditions keep those conditions separate.
///
/// A single-anchor pool yields that anchor verbatim, so
/// [`EmbeddingPool::match_score`] stays exactly `cos(emb, anchor)`.
fn condition_groups(anchors: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if anchors.is_empty() {
        return Vec::new();
    }
    let budget = (anchors.len() / MIN_ANCHORS_PER_GROUP).clamp(1, MAX_CONDITION_GROUPS);
    let mut groups: Vec<Vec<Vec<f32>>> = anchors.iter().map(|a| vec![a.clone()]).collect();
    let mut means: Vec<Vec<f32>> = anchors.to_vec();
    while groups.len() > budget {
        let mut best = (0_usize, 1_usize, f32::NEG_INFINITY);
        for i in 0..means.len() {
            for j in i + 1..means.len() {
                let s = cos_similarity(&means[i], &means[j]);
                if s > best.2 {
                    best = (i, j, s);
                }
            }
        }
        let (i, j, similarity) = best;
        // Honour the acoustic-condition boundary once the hard security
        // cap has been reached. Before that point we must keep merging:
        // an enrollment with twelve mutually 0.60-similar anchors is
        // accepted by the GUI's 0.40 consistency floor, but leaving all
        // twelve separate would recreate the exact per-anchor lottery
        // this grouping exists to remove.
        if similarity < CONDITION_MERGE_COS && groups.len() <= MAX_CONDITION_GROUPS {
            break;
        }
        let merged = groups.remove(j);
        means.remove(j);
        groups[i].extend(merged);
        means[i] = elementwise_mean(&groups[i]).expect("non-empty group");
    }
    debug_assert!(groups.len() <= MAX_CONDITION_GROUPS);
    means
}

impl EmbeddingPool {
    #[must_use]
    pub fn new(config: EmbeddingPoolConfig) -> Self {
        Self {
            config,
            anchors: Vec::new(),
            evidence: None,
            metadata: EnrollmentMetadata::default(),
            anchor_centroid: None,
            condition_centroids: Vec::new(),
            adapted: None,
        }
    }

    #[must_use]
    pub fn config(&self) -> EmbeddingPoolConfig {
        self.config
    }

    #[must_use]
    pub fn anchors(&self) -> &[Vec<f32>] {
        &self.anchors
    }

    /// The running evidence EMA — the mean-tracked runtime embedding
    /// before the residual pull is applied. `None` until the first
    /// admission. Mostly a diagnostics / persistence accessor; runtime
    /// scoring uses [`Self::adapted`].
    #[must_use]
    pub fn evidence(&self) -> Option<&[f32]> {
        self.evidence.as_deref()
    }

    /// The λ-residual self-augmented embedding
    /// `λ·anchor_centroid + (1-λ)·evidence`, or `None` until the first
    /// admission.
    #[must_use]
    pub fn adapted(&self) -> Option<&[f32]> {
        self.adapted.as_deref()
    }

    #[must_use]
    pub fn metadata(&self) -> EnrollmentMetadata {
        self.metadata
    }

    /// Number of distinct reference embeddings: every anchor, plus the
    /// adapted embedding when one exists.
    #[must_use]
    pub fn len(&self) -> usize {
        self.anchors.len() + usize::from(self.adapted.is_some())
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.anchors.is_empty() && self.adapted.is_none()
    }

    /// Iterate anchors first, then the adapted embedding (if any).
    pub fn iter(&self) -> impl Iterator<Item = &[f32]> {
        self.anchors
            .iter()
            .map(Vec::as_slice)
            .chain(self.adapted.as_deref())
    }

    pub fn add_anchors<I, V>(&mut self, embeddings: I)
    where
        I: IntoIterator<Item = V>,
        V: Into<Vec<f32>>,
    {
        for emb in embeddings {
            self.anchors.push(emb.into());
        }
        self.anchor_centroid = elementwise_mean(&self.anchors);
        self.condition_centroids = condition_groups(&self.anchors);
        // The adapted embedding depends on the centroid; keep it in
        // sync (a no-op in the normal flow — anchors are added once at
        // enrollment, before any runtime admission).
        self.recompute_adapted();
    }

    /// Element-wise mean of the anchor embeddings (`E_enroll`), or
    /// `None` when the pool has no anchors yet.
    #[must_use]
    pub fn anchor_centroid(&self) -> Option<&[f32]> {
        self.anchor_centroid.as_deref()
    }

    /// The acoustic-condition means [`Self::match_score`] compares
    /// against — see [`condition_groups`]. Empty for an empty pool.
    #[must_use]
    pub fn condition_centroids(&self) -> &[Vec<f32>] {
        &self.condition_centroids
    }

    /// Pool match score for `emb`: the **maximum** cosine similarity
    /// over the acoustic-condition means plus the λ-residual adapted
    /// embedding.
    ///
    /// Each term earns its place:
    ///
    /// * **per-condition max** — a profile deliberately holds anchors
    ///   from several acoustic conditions (morning voice, evening voice,
    ///   extra registrations added via `+声紋を追加登録`). Those conditions
    ///   sit in *different* directions in embedding space, so averaging
    ///   them all into one vector scores every one of them worse than
    ///   itself. Taking the max over the retained conditions preserves
    ///   distinct registrations without averaging all of them into one;
    ///   profiles beyond the four-condition security cap compress the
    ///   closest conditions together.
    /// * **averaging inside a condition** — the max runs over
    ///   [`condition_groups`] means, *not* over raw anchors. A raw-anchor
    ///   max hands an unrelated speaker one lottery ticket per anchor,
    ///   and with a 20 s enrollment that is a dozen chances to get lucky
    ///   on a single noisy window. Measured on same-sex, same-language
    ///   speaker pairs at `theta_pass = 0.45`, dropping the raw-anchor
    ///   terms takes an impostor from 10.8 % of windows leaking to 0.0 %,
    ///   for +0.4 pp of enrolled-speaker loss. (The earlier raw-anchor
    ///   measurement that justified the max — +0.028 on the target's p05
    ///   for +0.004 of impostor max — was taken against an *opposite-sex*
    ///   impostor, where the margin was wide enough to hide the cost.)
    /// * **adapted** — lets a candidate matching the current runtime
    ///   condition score well even when enrollment was recorded under a
    ///   different one.
    ///
    /// Floors at `0.0`; returns `0.0` for an empty pool. For a
    /// single-anchor pool this is exactly `cos(emb, anchor)`.
    #[must_use]
    pub fn match_score(&self, emb: &[f32]) -> f32 {
        let best = self
            .condition_centroids
            .iter()
            .map(Vec::as_slice)
            .chain(self.adapted.as_deref())
            .map(|reference| cos_similarity(emb, reference))
            .fold(f32::NEG_INFINITY, f32::max);
        if best.is_finite() {
            best.max(0.0)
        } else {
            0.0
        }
    }

    pub fn set_f0_stats(&mut self, mu: f32, sigma: f32) {
        self.metadata = EnrollmentMetadata {
            f0_mu: mu,
            f0_sigma: sigma,
        };
    }

    /// `1 - max_cos(anchors)`. Returns `None` if there are no anchors.
    #[must_use]
    pub fn anchor_distance(&self, emb: &[f32]) -> Option<f32> {
        if self.anchors.is_empty() {
            return None;
        }
        let best = self
            .anchors
            .iter()
            .map(|a| cos_similarity(emb, a))
            .fold(f32::NEG_INFINITY, f32::max);
        Some(1.0 - best)
    }

    /// Drift-safety check before folding a candidate into the adapted
    /// embedding. Returns `false` when there are no anchors yet.
    ///
    /// `bootstrap_from_runtime` doesn't relax this check — pool needs
    /// an anchor before `adapt` will accept anything. Use
    /// [`Self::bootstrap_seed`] for the empty-pool case; once a seed
    /// anchor exists, normal `adapt` admissions accumulate on top.
    #[must_use]
    pub fn can_auto_learn(&self, emb: &[f32]) -> bool {
        match self.anchor_distance(emb) {
            Some(d) => d <= self.config.anchor_distance_threshold,
            None => false,
        }
    }

    /// Seed the very first anchor from a runtime embedding, plus the
    /// f0 prior to use against subsequent refreshes. Idempotent: only
    /// fires once, when the pool is still empty *and*
    /// [`EmbeddingPoolConfig::allow_bootstrap_from_runtime`] is on.
    ///
    /// Once it succeeds the normal [`Self::adapt`] path takes over —
    /// later high-confidence runtime candidates accumulate into the
    /// adapted embedding via the usual `anchor_distance_threshold` /
    /// `adapt_rate` machinery. Callers that want a multi-anchor
    /// bootstrap should call this once with the first admission and
    /// then let `adapt` accumulate.
    ///
    /// Returns `true` iff the seed was installed.
    pub fn bootstrap_seed<V: Into<Vec<f32>>>(&mut self, emb: V, f0_mu: f32, f0_sigma: f32) -> bool {
        if !self.config.allow_bootstrap_from_runtime || !self.is_empty() {
            return false;
        }
        let emb = emb.into();
        self.anchors.push(emb);
        self.anchor_centroid = elementwise_mean(&self.anchors);
        self.condition_centroids = condition_groups(&self.anchors);
        self.metadata = EnrollmentMetadata { f0_mu, f0_sigma };
        // `evidence` stays None — the bootstrapped anchor IS the
        // entire pool for now; subsequent `adapt` calls will populate
        // evidence and refresh `adapted` from `anchor_centroid +
        // evidence` as usual.
        self.recompute_adapted();
        true
    }

    /// Fold a high-confidence runtime candidate into the adapted
    /// embedding. Returns `true` when the candidate cleared
    /// [`Self::can_auto_learn`] and was incorporated.
    ///
    /// First admission seeds `evidence` with the candidate verbatim;
    /// later admissions EMA it in at rate `adapt_rate`. Either way the
    /// cached `adapted = λ·centroid + (1-λ)·evidence` is refreshed, so
    /// the candidate can never pull the reference more than `(1-λ)` of
    /// the way off the enrollment centroid.
    pub fn adapt<V: Into<Vec<f32>>>(&mut self, emb: V) -> bool {
        let emb = emb.into();
        if !self.can_auto_learn(&emb) {
            return false;
        }
        match &mut self.evidence {
            None => self.evidence = Some(emb),
            Some(evidence) => {
                debug_assert_eq!(
                    evidence.len(),
                    emb.len(),
                    "candidate dimension must match the evidence EMA"
                );
                let eta = self.config.adapt_rate;
                for (e, &c) in evidence.iter_mut().zip(emb.iter()) {
                    *e = (1.0 - eta) * *e + eta * c;
                }
            }
        }
        self.recompute_adapted();
        true
    }

    /// Refresh the cached `adapted` embedding from `anchor_centroid` +
    /// `evidence`. `None` whenever there is no evidence yet.
    fn recompute_adapted(&mut self) {
        let Some(evidence) = self.evidence.as_deref() else {
            self.adapted = None;
            return;
        };
        let lambda = self.config.adapt_residual_lambda;
        self.adapted = Some(match self.anchor_centroid.as_deref() {
            Some(centroid) => centroid
                .iter()
                .zip(evidence.iter())
                .map(|(&c, &e)| lambda * c + (1.0 - lambda) * e)
                .collect(),
            // Unreachable in the normal flow: `can_auto_learn` rejects
            // every candidate while there are no anchors, so `evidence`
            // is never set without a centroid. Kept total for safety.
            None => evidence.to_vec(),
        });
    }

    // ---- JSON persistence ------------------------------------------

    /// Serialise the pool to a JSON string (schema version 2).
    ///
    /// # Errors
    /// Returns [`PersistenceError::Json`] if `serde_json` rejects the
    /// payload — should not happen for the in-memory shape, but the
    /// error path is preserved for symmetry with [`Self::from_json`].
    pub fn to_json(&self) -> Result<String, PersistenceError> {
        let payload = PoolPayload::from_pool(self);
        serde_json::to_string(&payload).map_err(PersistenceError::Json)
    }

    /// Deserialise a pool from a JSON string. The pool's `config` is
    /// supplied by the caller — it controls the adaptation knobs, which
    /// are not part of the on-disk payload.
    ///
    /// Schema version 2 is the native format. Version 1 (the old
    /// `auto_learn` FIFO) is accepted and migrated: the FIFO's mean
    /// seeds `evidence`. Any other version is rejected.
    ///
    /// # Errors
    /// * [`PersistenceError::Json`]               — malformed JSON
    /// * [`PersistenceError::UnsupportedVersion`] — `version` is
    ///   neither 1 nor 2
    pub fn from_json(s: &str, config: EmbeddingPoolConfig) -> Result<Self, PersistenceError> {
        let payload: PoolPayload = serde_json::from_str(s).map_err(PersistenceError::Json)?;
        if payload.version != 1 && payload.version != PERSISTENCE_VERSION {
            return Err(PersistenceError::UnsupportedVersion(payload.version));
        }
        Ok(payload.into_pool(config))
    }

    /// Persist the pool as JSON to `path`. Convenience wrapper around
    /// [`Self::to_json`] + `fs::write`.
    ///
    /// # Errors
    /// Returns [`PersistenceError::Io`] for filesystem failures or
    /// [`PersistenceError::Json`] if serialisation fails.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), PersistenceError> {
        let body = self.to_json()?;
        fs::write(path, body).map_err(PersistenceError::Io)
    }

    /// Load a pool from a JSON file at `path`.
    ///
    /// # Errors
    /// Returns [`PersistenceError::Io`] for filesystem failures or
    /// [`PersistenceError::Json`] / [`PersistenceError::UnsupportedVersion`]
    /// for malformed payloads.
    pub fn load(
        path: impl AsRef<Path>,
        config: EmbeddingPoolConfig,
    ) -> Result<Self, PersistenceError> {
        let body = fs::read_to_string(path).map_err(PersistenceError::Io)?;
        Self::from_json(&body, config)
    }
}

/// On-disk representation of [`EmbeddingPool`]. Kept private to the
/// module so the public type remains free to evolve.
#[derive(Debug, Serialize, Deserialize)]
struct PoolPayload {
    version: u32,
    #[serde(default)]
    anchors: Vec<Vec<f32>>,
    /// Version 2: the running evidence EMA.
    #[serde(default)]
    evidence: Option<Vec<f32>>,
    /// Version 1 only: the legacy admitted-embedding FIFO. Ignored when
    /// `evidence` is present; otherwise migrated to `evidence` by mean.
    #[serde(default)]
    auto_learn: Vec<Vec<f32>>,
    #[serde(default)]
    metadata: EnrollmentMetadata,
}

impl PoolPayload {
    fn from_pool(pool: &EmbeddingPool) -> Self {
        Self {
            version: PERSISTENCE_VERSION,
            anchors: pool.anchors.clone(),
            evidence: pool.evidence.clone(),
            auto_learn: Vec::new(),
            metadata: pool.metadata,
        }
    }

    fn into_pool(self, config: EmbeddingPoolConfig) -> EmbeddingPool {
        // v2 carries `evidence` directly; v1 carried an `auto_learn`
        // FIFO instead — migrate it by taking its mean.
        let evidence = self.evidence.or_else(|| elementwise_mean(&self.auto_learn));
        let anchor_centroid = elementwise_mean(&self.anchors);
        let condition_centroids = condition_groups(&self.anchors);
        let mut pool = EmbeddingPool {
            config,
            anchors: self.anchors,
            evidence,
            metadata: self.metadata,
            anchor_centroid,
            condition_centroids,
            adapted: None,
        };
        pool.recompute_adapted();
        pool
    }
}

/// Errors returned by the [`EmbeddingPool`] JSON persistence path.
#[derive(Debug)]
pub enum PersistenceError {
    Io(std::io::Error),
    Json(serde_json::Error),
    UnsupportedVersion(u32),
}

impl std::fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Json(e) => write!(f, "JSON error: {e}"),
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported enrollment version: {v}")
            }
        }
    }
}

impl std::error::Error for PersistenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Json(e) => Some(e),
            Self::UnsupportedVersion(_) => None,
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_precision_loss)]
mod tests {
    use super::*;

    fn unit(v: &[f32]) -> Vec<f32> {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    #[test]
    fn anchor_distance_zero_for_identical() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        let anchor = unit(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        pool.add_anchors([anchor.clone()]);
        let d = pool.anchor_distance(&anchor).unwrap();
        assert!(d.abs() < 1e-6, "d={d}");
    }

    #[test]
    fn anchor_distance_none_when_empty() {
        let pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        assert!(pool.anchor_distance(&[1.0, 0.0]).is_none());
    }

    #[test]
    fn anchor_centroid_none_when_empty() {
        let pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        assert!(pool.anchor_centroid().is_none());
    }

    #[test]
    fn anchor_centroid_single_anchor_is_verbatim() {
        // Centroid of one anchor is that anchor byte-for-byte (each
        // f32 multiplied by 1.0). This is what keeps `pipeline_parity`
        // byte-equal under #117.
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        let anchor = unit(&[0.3, -0.7, 0.1, 0.64]);
        pool.add_anchors([anchor.clone()]);
        assert_eq!(pool.anchor_centroid().unwrap(), anchor.as_slice());
    }

    #[test]
    fn anchor_centroid_is_elementwise_mean() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        pool.add_anchors([vec![1.0_f32, 3.0, -1.0], vec![3.0_f32, 1.0, 1.0]]);
        assert_eq!(pool.anchor_centroid().unwrap(), &[2.0, 2.0, 0.0]);
    }

    #[test]
    fn add_anchors_recomputes_centroid() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        pool.add_anchors([vec![0.0_f32, 2.0]]);
        assert_eq!(pool.anchor_centroid().unwrap(), &[0.0, 2.0]);
        pool.add_anchors([vec![2.0_f32, 0.0]]);
        assert_eq!(pool.anchor_centroid().unwrap(), &[1.0, 1.0]);
    }

    #[test]
    fn match_score_single_anchor_equals_cos_similarity() {
        // Single anchor, no admitted evidence: match_score reduces to
        // cos_similarity against that anchor (the pre-#117 path).
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        let anchor = unit(&[1.0, 0.0, 0.0]);
        pool.add_anchors([anchor.clone()]);
        let emb = unit(&[0.8, 0.6, 0.0]);
        assert!((pool.match_score(&emb) - cos_similarity(&emb, &anchor)).abs() < 1e-6);
    }

    #[test]
    fn match_score_takes_best_condition_over_centroid() {
        // Two anchors pointing in different directions (the shape a
        // multi-condition profile has: morning voice + evening voice).
        // Their centroid sits on the
        // bisector and matches *neither* well. A probe equal to one
        // anchor must score ~1.0 because two conditions fit below the
        // hard cap and are retained separately.
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        let a = unit(&[1.0, 1.0, 0.0]);
        let b = unit(&[1.0, -1.0, 0.0]);
        pool.add_anchors([a.clone(), b]);
        let probe = a.clone();
        let centroid_score = cos_similarity(&probe, pool.anchor_centroid().unwrap());
        let score = pool.match_score(&probe);
        assert!((score - 1.0).abs() < 1e-6, "score={score}");
        assert!(
            score > centroid_score,
            "per-condition max {score} must beat the centroid {centroid_score}"
        );
    }

    #[test]
    fn many_similar_anchors_collapse_into_a_few_condition_means() {
        // The reason this matters: a max taken over a dozen raw anchors
        // hands an unrelated speaker a dozen chances to get lucky on one
        // noisy window, which is what let a same-sex impostor through.
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        // 12 anchors scattered around one direction, as one recording's
        // sliding windows are.
        let anchors: Vec<Vec<f32>> = (0..12)
            .map(|i| {
                let mut v = vec![0.0_f32; 8];
                v[0] = 1.0;
                v[1 + i % 6] = 0.12 * (1.0 + (i % 3) as f32);
                v
            })
            .collect();
        pool.add_anchors(anchors);
        assert!(
            pool.condition_centroids().len() <= MAX_CONDITION_GROUPS,
            "12 same-condition anchors must not stay 12 references, got {}",
            pool.condition_centroids().len()
        );
        assert!(!pool.condition_centroids().is_empty());
    }

    #[test]
    fn accepted_but_variable_enrollment_still_obeys_the_hard_group_cap() {
        // Every pair has cosine 0.60: comfortably above the GUI's 0.40
        // enrollment-consistency floor, but below CONDITION_MERGE_COS.
        // The old early break left all twelve as lottery tickets.
        let common = 0.60_f32.sqrt();
        let unique = 0.40_f32.sqrt();
        let anchors: Vec<Vec<f32>> = (0..12)
            .map(|i| {
                let mut v = vec![0.0_f32; 13];
                v[0] = common;
                v[i + 1] = unique;
                v
            })
            .collect();
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        pool.add_anchors(anchors);
        assert!(
            (1..=MAX_CONDITION_GROUPS).contains(&pool.condition_centroids().len()),
            "accepted enrollment must collapse to the hard cap, got {} groups",
            pool.condition_centroids().len()
        );
    }

    #[test]
    fn hard_group_cap_wins_over_many_distinct_registrations() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        let anchors: Vec<Vec<f32>> = (0..8)
            .map(|axis| {
                let mut v = vec![0.0_f32; 8];
                v[axis] = 1.0;
                v
            })
            .collect();
        pool.add_anchors(anchors);
        assert_eq!(pool.condition_centroids().len(), MAX_CONDITION_GROUPS);
    }

    #[test]
    fn distinct_registrations_survive_the_group_budget() {
        // Two acoustically distant registrations, far more than the
        // budget of one group they would get on count alone. Averaging
        // them together would match neither, so they must stay apart.
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        let mut anchors: Vec<Vec<f32>> = Vec::new();
        for i in 0..4_u8 {
            anchors.push(unit(&[1.0, 0.05 * f32::from(i), 0.0]));
            anchors.push(unit(&[0.0, 0.05 * f32::from(i), 1.0]));
        }
        pool.add_anchors(anchors);
        assert!(
            pool.condition_centroids().len() >= 2,
            "two distinct conditions must not be merged"
        );
        // A probe sitting on either condition scores near 1.0.
        for probe in [unit(&[1.0, 0.0, 0.0]), unit(&[0.0, 0.0, 1.0])] {
            let s = pool.match_score(&probe);
            assert!(s > 0.95, "score={s} for a probe on an enrolled condition");
        }
    }

    #[test]
    fn match_score_still_uses_centroid_when_it_wins() {
        // For a single-condition profile the centroid averages away
        // per-window noise and can beat every individual anchor; it
        // stays in the max so that gain isn't lost.
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        let a = unit(&[1.0, 0.20, 0.0]);
        let b = unit(&[1.0, -0.20, 0.0]);
        pool.add_anchors([a.clone(), b.clone()]);
        let probe = unit(&[1.0, 0.0, 0.0]);
        let best_anchor = cos_similarity(&probe, &a).max(cos_similarity(&probe, &b));
        let score = pool.match_score(&probe);
        assert!(
            score > best_anchor,
            "centroid score {score} should beat the best anchor {best_anchor}"
        );
    }

    #[test]
    fn match_score_zero_for_empty_pool() {
        let pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        assert_eq!(pool.match_score(&[1.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn match_score_floors_negative_cosine_at_zero() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        pool.add_anchors([vec![1.0, 0.0, 0.0]]);
        assert_eq!(pool.match_score(&[-1.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn can_auto_learn_rejects_far_embedding() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig {
            anchor_distance_threshold: 0.4,
            ..EmbeddingPoolConfig::default()
        });
        pool.add_anchors([unit(&[1.0, 0.0, 0.0])]);
        let near = unit(&[0.95, 0.05, 0.0]);
        let far = unit(&[0.0, 0.0, 1.0]);
        assert!(pool.can_auto_learn(&near));
        assert!(!pool.can_auto_learn(&far));
    }

    #[test]
    fn adapt_rejects_without_anchor() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        assert!(!pool.adapt(unit(&[1.0, 0.0])));
        assert!(pool.evidence().is_none());
        assert!(pool.adapted().is_none());
    }

    #[test]
    fn bootstrap_seed_rejected_when_flag_disabled() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig {
            allow_bootstrap_from_runtime: false,
            ..EmbeddingPoolConfig::default()
        });
        assert!(!pool.bootstrap_seed(unit(&[1.0, 0.0, 0.0]), 150.0, 30.0));
        assert!(pool.is_empty());
        assert_eq!(pool.metadata().f0_mu, 0.0);
    }

    #[test]
    fn bootstrap_seed_rejected_when_pool_nonempty() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig {
            allow_bootstrap_from_runtime: true,
            ..EmbeddingPoolConfig::default()
        });
        pool.add_anchors([unit(&[1.0, 0.0, 0.0])]);
        let before_count = pool.anchors().len();
        // The seed must not replace or duplicate the existing anchor.
        assert!(!pool.bootstrap_seed(unit(&[0.0, 1.0, 0.0]), 200.0, 50.0));
        assert_eq!(pool.anchors().len(), before_count);
    }

    #[test]
    fn bootstrap_seed_installs_first_anchor_and_f0() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig {
            allow_bootstrap_from_runtime: true,
            ..EmbeddingPoolConfig::default()
        });
        let seed = unit(&[1.0, 0.0, 0.0]);
        assert!(pool.bootstrap_seed(seed.clone(), 175.0, 35.0));
        assert_eq!(pool.anchors().len(), 1);
        assert_eq!(pool.metadata().f0_mu, 175.0);
        assert_eq!(pool.metadata().f0_sigma, 35.0);
        // The centroid of a one-anchor pool is the anchor verbatim, so
        // match_score against the seed is exactly 1.
        let score = pool.match_score(&seed);
        assert!((score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn bootstrap_then_adapt_accumulates_evidence() {
        // After bootstrap installs the first anchor, the normal
        // `adapt` path takes over and folds further high-confidence
        // candidates into evidence — i.e. bootstrap doesn't lock the
        // pool, it kickstarts it.
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig {
            allow_bootstrap_from_runtime: true,
            anchor_distance_threshold: 0.5,
            ..EmbeddingPoolConfig::default()
        });
        assert!(pool.bootstrap_seed(unit(&[1.0, 0.0, 0.0]), 150.0, 30.0));
        // A near candidate clears `can_auto_learn` because the anchor
        // distance check now has an anchor to compare against.
        let near = unit(&[0.95, 0.05, 0.0]);
        assert!(pool.adapt(near));
        assert!(pool.evidence().is_some());
    }

    #[test]
    fn adapt_rejects_far_candidate() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig {
            anchor_distance_threshold: 0.4,
            ..EmbeddingPoolConfig::default()
        });
        pool.add_anchors([unit(&[1.0, 0.0, 0.0])]);
        assert!(!pool.adapt(unit(&[0.0, 0.0, 1.0])));
        assert!(pool.adapted().is_none());
    }

    #[test]
    fn adapt_first_admission_seeds_evidence_verbatim() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig {
            anchor_distance_threshold: 1.0,
            ..EmbeddingPoolConfig::default()
        });
        let anchor = unit(&[1.0, 0.0, 0.0]);
        pool.add_anchors([anchor.clone()]);
        let cand = unit(&[0.0, 1.0, 0.0]);
        assert!(pool.adapt(cand.clone()));
        // evidence is the candidate verbatim on the first admission.
        for (e, c) in pool.evidence().unwrap().iter().zip(cand.iter()) {
            assert!((e - c).abs() < 1e-6);
        }
    }

    #[test]
    fn adapt_residual_pulls_toward_centroid() {
        // adapted = λ·centroid + (1-λ)·evidence. With one admission,
        // evidence == candidate, so adapted is exactly that blend.
        let lambda = 0.25_f32;
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig {
            anchor_distance_threshold: 1.0,
            adapt_residual_lambda: lambda,
            ..EmbeddingPoolConfig::default()
        });
        let centroid = vec![1.0_f32, 0.0, 0.0];
        pool.add_anchors([centroid.clone()]);
        let cand = vec![0.0_f32, 1.0, 0.0];
        assert!(pool.adapt(cand.clone()));
        let adapted = pool.adapted().unwrap();
        for ((a, &c), &v) in adapted.iter().zip(centroid.iter()).zip(cand.iter()) {
            let want = lambda * c + (1.0 - lambda) * v;
            assert!((a - want).abs() < 1e-6, "got {a}, want {want}");
        }
    }

    #[test]
    fn adapt_ema_blends_later_admissions() {
        let eta = 0.5_f32;
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig {
            anchor_distance_threshold: 1.0,
            adapt_rate: eta,
            ..EmbeddingPoolConfig::default()
        });
        pool.add_anchors([vec![1.0_f32, 0.0]]);
        pool.adapt(vec![0.0_f32, 1.0]); // evidence = [0, 1]
        pool.adapt(vec![1.0_f32, 1.0]); // evidence = 0.5·[0,1] + 0.5·[1,1] = [0.5, 1]
        let evidence = pool.evidence().unwrap();
        assert!((evidence[0] - 0.5).abs() < 1e-6, "ev0={}", evidence[0]);
        assert!((evidence[1] - 1.0).abs() < 1e-6, "ev1={}", evidence[1]);
    }

    #[test]
    fn match_score_maxes_centroid_with_adapted() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig {
            anchor_distance_threshold: 1.0,
            adapt_residual_lambda: 0.1,
            ..EmbeddingPoolConfig::default()
        });
        pool.add_anchors([unit(&[1.0, 0.0, 0.0])]);
        let probe = unit(&[0.0, 1.0, 0.0]);
        // Orthogonal to the anchor -> centroid score ~0, no evidence yet.
        assert!(pool.match_score(&probe) < 1e-6);
        // Admit the probe: adapted moves toward it (minus the λ residual),
        // so match_score against the probe jumps well above 0.
        assert!(pool.adapt(probe.clone()));
        assert!(
            pool.match_score(&probe) > 0.8,
            "score={}",
            pool.match_score(&probe)
        );
    }

    #[test]
    fn iter_yields_anchors_then_adapted() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig {
            anchor_distance_threshold: 1.0,
            ..EmbeddingPoolConfig::default()
        });
        pool.add_anchors([unit(&[1.0, 0.0])]);
        assert_eq!(pool.iter().count(), 1); // anchor only
        pool.adapt(unit(&[0.0, 1.0]));
        assert_eq!(pool.iter().count(), 2); // anchor + adapted
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn set_f0_stats_round_trip() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        pool.set_f0_stats(180.0, 25.0);
        let m = pool.metadata();
        assert_eq!(m.f0_mu, 180.0);
        assert_eq!(m.f0_sigma, 25.0);
    }

    // -----------------------------------------------------------------
    // JSON persistence
    // -----------------------------------------------------------------

    #[test]
    fn json_round_trip_preserves_state() {
        let cfg = EmbeddingPoolConfig {
            anchor_distance_threshold: 1.0,
            ..EmbeddingPoolConfig::default()
        };
        let mut pool = EmbeddingPool::new(cfg);
        pool.add_anchors([unit(&[1.0, 0.0])]);
        pool.set_f0_stats(180.0, 25.0);
        assert!(pool.adapt(unit(&[0.0, 1.0])));

        let body = pool.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["version"], 2);
        assert_eq!(parsed["anchors"].as_array().unwrap().len(), 1);
        assert_eq!(parsed["evidence"].as_array().unwrap().len(), 2);

        let restored = EmbeddingPool::from_json(&body, cfg).unwrap();
        assert_eq!(restored.anchors().len(), 1);
        assert_eq!(restored.metadata().f0_mu, 180.0);
        assert_eq!(restored.metadata().f0_sigma, 25.0);
        let (a, b) = (pool.evidence().unwrap(), restored.evidence().unwrap());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
        // The cached adapted embedding is rebuilt on load.
        assert!(restored.adapted().is_some());
    }

    #[test]
    fn save_load_through_filesystem() {
        let cfg = EmbeddingPoolConfig::default();
        let mut pool = EmbeddingPool::new(cfg);
        pool.add_anchors([unit(&[1.0, 0.0])]);

        let dir = std::env::temp_dir().join(format!(
            "mellonella-pool-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("enrollment.json");

        pool.save(&path).unwrap();
        let restored = EmbeddingPool::load(&path, cfg).unwrap();
        assert_eq!(restored.anchors().len(), 1);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn from_json_rejects_unknown_version() {
        let cfg = EmbeddingPoolConfig::default();
        let body = r#"{"version": 99, "anchors": [], "metadata": {}}"#;
        match EmbeddingPool::from_json(body, cfg) {
            Err(PersistenceError::UnsupportedVersion(99)) => {}
            other => panic!("expected UnsupportedVersion(99), got {other:?}"),
        }
    }

    #[test]
    fn from_json_migrates_v1_fifo_to_evidence_mean() {
        // A version-1 payload carried an `auto_learn` FIFO instead of
        // `evidence`. On load the FIFO is migrated to `evidence` by
        // taking its element-wise mean.
        let saved = r#"{
            "version": 1,
            "anchors": [[1.0, 0.0]],
            "auto_learn": [[0.0, 2.0], [0.0, 4.0]],
            "metadata": {"f0_mu": 0.0, "f0_sigma": 0.0}
        }"#;
        let cfg = EmbeddingPoolConfig::default();
        let pool = EmbeddingPool::from_json(saved, cfg).unwrap();
        // mean([0,2], [0,4]) = [0, 3]
        let evidence = pool.evidence().unwrap();
        assert!((evidence[0] - 0.0).abs() < 1e-6);
        assert!((evidence[1] - 3.0).abs() < 1e-6);
        assert!(pool.adapted().is_some());
    }

    #[test]
    fn from_json_v1_without_fifo_has_no_evidence() {
        let saved = r#"{"version": 1, "anchors": [[1.0, 0.0]], "auto_learn": [], "metadata": {}}"#;
        let pool = EmbeddingPool::from_json(saved, EmbeddingPoolConfig::default()).unwrap();
        assert!(pool.evidence().is_none());
        assert!(pool.adapted().is_none());
    }
}
