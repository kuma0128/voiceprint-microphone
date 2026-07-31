//! Online model of the *other* people heard during a session.
//!
//! # Why this exists
//!
//! The gate's only evidence used to be `cos(window, enrolled profile)`
//! against a fixed threshold. That works when the other speaker is
//! acoustically distant, and fails when they are not: measured on
//! same-sex, same-language speaker pairs, an unrelated speaker's 1 s
//! windows reach a median of 0.35 and a maximum of 0.53 against the
//! enrolled profile, while the enrolled speaker's own 5th percentile is
//! only 0.43. The two distributions **overlap**, so no threshold on that
//! number alone can separate them — raising it cuts the enrolled speaker,
//! lowering it lets the friend through.
//!
//! What does separate them is a comparison the gate was throwing away:
//! a friend's window is far more similar to *the friend's other windows*
//! than it is to the enrolled speaker. Concretely, on the same fixtures:
//!
//! | quantity                                   | value            |
//! |--------------------------------------------|------------------|
//! | impostor window: `s_target - s_own_cluster` | p50 −0.37, max +0.05 |
//! | enrolled window: `s_target - s_others`      | p05 +0.19        |
//!
//! Those *do not* overlap. So this module accumulates the other speakers
//! it hears into a small set of centroids, and the gate additionally
//! requires the enrolled speaker to win by [`OtherSpeakerConfig::margin`].
//!
//! # Safety properties
//!
//! The failure this must never cause is learning the enrolled speaker
//! and then locking them out. Three guards make that impossible:
//!
//! 1. **Only rejected audio is learned.** A window is a candidate only
//!    while its target score is below the pass threshold — audio the gate
//!    is already refusing to pass.
//! 2. **Every cluster is re-checked against the profile.** A seed or an
//!    update that would put a centroid within
//!    [`OtherSpeakerConfig::guard_frac`] of the pass threshold is
//!    discarded, so no centroid can drift onto the enrolled speaker.
//! 3. **Two agreeing windows are needed to open a cluster**, after a run
//!    of consecutive rejected windows — an isolated bad window (a cough,
//!    a clipped syllable, the tail of a sentence) cannot seed anything.
//!
//! The model is session-scoped: [`OtherSpeakers::reset`] clears it when a
//! session starts. Persisting it across runs would buy a slightly faster
//! first block at the cost of a stale cluster outliving whatever produced
//! it, which is the wrong trade for a gate that must never lock its owner
//! out.

use crate::gating::cos_similarity;

/// Tuning for [`OtherSpeakers`].
#[derive(Debug, Clone, Copy)]
pub struct OtherSpeakerConfig {
    /// How far the enrolled speaker must out-score every known other
    /// speaker for the gate to open. `0.0` still demands a strict win.
    pub margin: f32,
    /// Learn only from windows scoring below `theta_pass * this`.
    pub seed_ceiling_frac: f32,
    /// Discard any centroid that scores at or above
    /// `theta_pass * this` against the enrolled profile.
    pub guard_frac: f32,
    /// Consecutive below-ceiling windows required before a new cluster
    /// may be opened.
    pub min_run: u32,
    /// Two candidate windows must be at least this similar to each other
    /// to open a cluster together.
    pub agree: f32,
    /// A window at least this similar to an existing cluster updates it
    /// instead of opening a new one.
    pub merge: f32,
    /// EMA step for folding a window into the cluster it matched.
    pub eta: f32,
    /// Cluster budget; the least-reinforced one is evicted when full.
    pub max_clusters: usize,
}

impl Default for OtherSpeakerConfig {
    fn default() -> Self {
        Self {
            // 0.0 disables the margin test, keeping `OtherSpeakers`
            // inert unless a caller opts in. The live GUI sets 0.08.
            margin: 0.0,
            seed_ceiling_frac: 1.0,
            guard_frac: 0.85,
            min_run: 2,
            agree: 0.45,
            merge: 0.45,
            eta: 0.2,
            max_clusters: 6,
        }
    }
}

/// Session-scoped centroids of the non-enrolled speakers heard so far.
#[derive(Debug, Clone, Default)]
pub struct OtherSpeakers {
    centroids: Vec<Vec<f32>>,
    /// Reinforcement count per centroid, used to pick an eviction victim.
    counts: Vec<u32>,
    /// Consecutive below-ceiling windows seen, and the previous one —
    /// a new cluster needs two that agree.
    run: u32,
    pending: Option<Vec<f32>>,
}

impl OtherSpeakers {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of distinct other speakers currently modelled.
    #[must_use]
    pub fn len(&self) -> usize {
        self.centroids.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.centroids.is_empty()
    }

    /// Forget every other speaker. Called when a session starts.
    pub fn reset(&mut self) {
        self.centroids.clear();
        self.counts.clear();
        self.run = 0;
        self.pending = None;
    }

    /// Best cosine similarity between `emb` and any known other speaker,
    /// or `0.0` when none are known yet.
    #[must_use]
    pub fn score(&self, emb: &[f32]) -> f32 {
        self.centroids
            .iter()
            .map(|c| cos_similarity(emb, c))
            .fold(0.0_f32, f32::max)
    }

    /// Fold one identity window into the model.
    ///
    /// `target_score` is that window's score against the enrolled
    /// profile and `theta_pass` the gate's pass threshold, so the model
    /// only ever learns from audio the gate is already rejecting.
    /// `profile_score` re-scores a candidate centroid against the
    /// enrolled profile — it is the guard that stops a centroid drifting
    /// onto the enrolled speaker, so it must be the same scoring
    /// function the gate uses.
    pub fn observe<F>(
        &mut self,
        emb: &[f32],
        target_score: f32,
        theta_pass: f32,
        config: &OtherSpeakerConfig,
        profile_score: F,
    ) where
        F: Fn(&[f32]) -> f32,
    {
        if target_score >= theta_pass * config.seed_ceiling_frac {
            self.run = 0;
            self.pending = None;
            return;
        }
        self.run = self.run.saturating_add(1);
        let guard = theta_pass * config.guard_frac;

        // Reinforce the cluster this window belongs to, if any. Tracking
        // a known speaker across the session matters more than opening
        // new clusters, so this runs before the seeding path.
        if let Some((idx, best)) = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, cos_similarity(emb, c)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
        {
            if best >= config.merge {
                let updated: Vec<f32> = self.centroids[idx]
                    .iter()
                    .zip(emb)
                    .map(|(&c, &x)| (1.0 - config.eta) * c + config.eta * x)
                    .collect();
                if profile_score(&updated) < guard {
                    self.centroids[idx] = updated;
                    self.counts[idx] = self.counts[idx].saturating_add(1);
                }
                return;
            }
        }

        if self.run < config.min_run {
            return;
        }
        match self.pending.take() {
            Some(prev) if cos_similarity(&prev, emb) >= config.agree => {
                let seed: Vec<f32> = prev
                    .iter()
                    .zip(emb)
                    .map(|(&a, &b)| 0.5 * (a + b))
                    .collect();
                if profile_score(&seed) < guard {
                    if self.centroids.len() >= config.max_clusters {
                        let victim = self
                            .counts
                            .iter()
                            .enumerate()
                            .min_by_key(|(_, &c)| c)
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        self.centroids.remove(victim);
                        self.counts.remove(victim);
                    }
                    self.centroids.push(seed);
                    self.counts.push(2);
                }
            }
            // Either nothing pending, or the two windows disagreed —
            // keep the newer one as the next candidate.
            _ => self.pending = Some(emb.to_vec()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(dim: usize, hot: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[hot] = 1.0;
        v
    }

    /// A window mostly along `hot` with a little of `other` mixed in, so
    /// repeated draws are similar to each other but not identical.
    fn near(dim: usize, hot: usize, other: usize, mix: f32) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[hot] = 1.0 - mix;
        v[other] = mix;
        v
    }

    #[test]
    fn empty_model_scores_zero_and_never_blocks() {
        let m = OtherSpeakers::new();
        assert!(m.is_empty());
        assert!((m.score(&unit(8, 0))).abs() < 1e-6);
    }

    #[test]
    fn learns_a_repeated_other_speaker() {
        let cfg = OtherSpeakerConfig::default();
        let mut m = OtherSpeakers::new();
        // The enrolled profile lives on axis 0; the other speaker on 1.
        let profile = |e: &[f32]| cos_similarity(e, &unit(8, 0));
        for k in 0..6_u8 {
            let w = near(8, 1, 2, 0.05 + 0.01 * f32::from(k));
            m.observe(&w, 0.20, 0.45, &cfg, profile);
        }
        assert_eq!(m.len(), 1, "one repeated speaker is one cluster");
        assert!(
            m.score(&near(8, 1, 2, 0.07)) > 0.9,
            "a further window from that speaker must match the cluster"
        );
        assert!(
            m.score(&unit(8, 0)) < 0.2,
            "the enrolled direction must not match it"
        );
    }

    #[test]
    fn never_learns_windows_the_gate_would_pass() {
        let cfg = OtherSpeakerConfig::default();
        let mut m = OtherSpeakers::new();
        let profile = |e: &[f32]| cos_similarity(e, &unit(8, 0));
        for _ in 0..10 {
            // Above theta_pass — the gate accepts these, so they are the
            // enrolled speaker as far as this model is concerned.
            m.observe(&unit(8, 1), 0.60, 0.45, &cfg, profile);
        }
        assert!(m.is_empty());
    }

    #[test]
    fn guard_rejects_a_seed_that_looks_like_the_enrolled_speaker() {
        let cfg = OtherSpeakerConfig::default();
        let mut m = OtherSpeakers::new();
        // Profile scoring that calls everything a match: every candidate
        // centroid is above the guard, so nothing may be learned.
        let profile = |_: &[f32]| 1.0_f32;
        for _ in 0..10 {
            m.observe(&unit(8, 1), 0.10, 0.45, &cfg, profile);
        }
        assert!(
            m.is_empty(),
            "the guard must veto every seed near the enrolled profile"
        );
    }

    #[test]
    fn isolated_bad_windows_do_not_seed_a_cluster() {
        let cfg = OtherSpeakerConfig::default();
        let mut m = OtherSpeakers::new();
        let profile = |e: &[f32]| cos_similarity(e, &unit(8, 0));
        // One rejected window, then an accepted one, repeatedly: the run
        // never reaches `min_run` with a pending candidate to agree with.
        for _ in 0..10 {
            m.observe(&unit(8, 1), 0.10, 0.45, &cfg, profile);
            m.observe(&unit(8, 0), 0.90, 0.45, &cfg, profile);
        }
        assert!(m.is_empty());
    }

    #[test]
    fn disagreeing_windows_do_not_seed_a_cluster() {
        let cfg = OtherSpeakerConfig::default();
        let mut m = OtherSpeakers::new();
        let profile = |e: &[f32]| cos_similarity(e, &unit(8, 0));
        // Alternating orthogonal directions never agree with each other.
        for k in 0..12 {
            let w = unit(8, 1 + usize::from(k % 2 == 0));
            m.observe(&w, 0.10, 0.45, &cfg, profile);
        }
        assert!(m.is_empty());
    }

    #[test]
    fn separate_speakers_get_separate_clusters() {
        let cfg = OtherSpeakerConfig::default();
        let mut m = OtherSpeakers::new();
        let profile = |e: &[f32]| cos_similarity(e, &unit(8, 0));
        for _ in 0..4 {
            m.observe(&near(8, 1, 5, 0.05), 0.20, 0.45, &cfg, profile);
        }
        for _ in 0..4 {
            m.observe(&near(8, 3, 6, 0.05), 0.20, 0.45, &cfg, profile);
        }
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn cluster_budget_is_enforced() {
        let cfg = OtherSpeakerConfig {
            max_clusters: 2,
            ..OtherSpeakerConfig::default()
        };
        let mut m = OtherSpeakers::new();
        let profile = |e: &[f32]| cos_similarity(e, &unit(16, 0));
        for axis in [1_usize, 3, 5, 7] {
            for _ in 0..4 {
                m.observe(&near(16, axis, axis + 8, 0.05), 0.20, 0.45, &cfg, profile);
            }
        }
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn reset_clears_everything() {
        let cfg = OtherSpeakerConfig::default();
        let mut m = OtherSpeakers::new();
        let profile = |e: &[f32]| cos_similarity(e, &unit(8, 0));
        for _ in 0..4 {
            m.observe(&near(8, 1, 2, 0.05), 0.20, 0.45, &cfg, profile);
        }
        assert_eq!(m.len(), 1);
        m.reset();
        assert!(m.is_empty());
        assert!(m.score(&near(8, 1, 2, 0.05)).abs() < 1e-6);
    }
}
