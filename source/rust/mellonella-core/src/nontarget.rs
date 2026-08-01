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
//! requires the enrolled speaker to win by
//! [`crate::gating::GateConfig::other_speaker_margin`].
//!
//! # Safety properties
//!
//! The failure this must never cause is learning the enrolled speaker
//! and then locking them out. Three guards make that unlikely, and a
//! fourth provides a bounded recovery path if the guards are defeated:
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
//! 4. **Consistent target evidence wins.** Every window that clears the
//!    enrolled-speaker threshold and is blocked *only* by a cluster is
//!    evidence against that cluster; every window that does not cancels
//!    part of it. Once
//!    [`OtherSpeakerConfig::conflict_forget_windows`] windows' worth has
//!    accumulated, the cluster is discarded. An impostor's isolated
//!    upward spikes stay blocked; a mislearned owner cannot be locked
//!    out indefinitely.
//!
//! Guard 4 cancels rather than *resets*, and that distinction is the
//! whole safety property. A speaker whose acoustic condition has
//! degraded — new microphone, a cold, a noisy room — does not sit
//! cleanly above the threshold once they recover, they straddle it. A
//! rule that demanded consecutive conflicts could be starved for the
//! entire session by every second window falling short, which is exactly
//! the permanent lockout guard 4 exists to prevent. Cancelling instead
//! turns the question into a *rate*, and the two rates are far apart:
//! against the 0.45 threshold a same-sex other speaker clears it in
//! about one window in twenty (measured p50 0.35, p95 0.45, max 0.53),
//! while the enrolled speaker's own 5th percentile is 0.43 — they clear
//! it nearly always. See [`CONFLICT_WEIGHT`] for where that leaves the
//! break-even.
//!
//! That bound holds only while the threshold this module is handed is
//! the one the gate actually admits on, and while both target and other
//! scores are raw cosine values. It is why
//! [`crate::gating::GateConfig::other_speaker_margin`] is incompatible
//! with `adaptive_theta` and `use_as_norm`: an adapted threshold can
//! sink below the level at which a target score identifies the speaker,
//! while AS-Norm changes only the target score to a z-score. Streaming
//! pipeline construction and live gate updates reject both combinations
//! before they can reach this module.
//!
//! The model is session-scoped: [`OtherSpeakers::reset`] clears it when a
//! session starts. Persisting it across runs would buy a slightly faster
//! first block at the cost of a stale cluster outliving whatever produced
//! it, which is the wrong trade for a gate that must never lock its owner
//! out.

use crate::gating::cos_similarity;

/// Evidence a conflicting window contributes, against the one unit that
/// every non-conflicting window cancels.
///
/// This sets the break-even rate of the recovery guard: conflicting
/// windows have to outnumber non-conflicting ones by more than
/// `1 : CONFLICT_WEIGHT - 1` — i.e. more than one window in three —
/// before a cluster is forgotten. Both populations sit far from that
/// line. Measured against the 0.45 threshold, a same-sex other speaker
/// clears it in about one window in twenty (p95 0.45) and the enrolled
/// speaker in nearly all of them (p05 0.43), so the guard neither
/// erodes a genuine cluster nor stalls on an owner whose degraded
/// condition leaves them only straddling the threshold. A weight of `1`
/// would put the break-even at exactly one in two, which a straddling
/// owner can sit on forever.
const CONFLICT_WEIGHT: u32 = 2;

/// Tuning for [`OtherSpeakers`].
#[derive(Debug, Clone, Copy)]
pub struct OtherSpeakerConfig {
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
    /// `0` disables learning and clears any pending seed.
    pub max_clusters: usize,
    /// Windows that clear the target threshold but would be blocked by
    /// the same other-speaker cluster before that cluster is forgotten.
    ///
    /// This is the recovery path for a low-scoring acoustic condition of
    /// the enrolled speaker being mislearned as somebody else. It is a
    /// count of *unanimous* windows: interleaved windows that do not
    /// conflict cancel part of the evidence, so a cluster contradicted
    /// only intermittently takes proportionally longer to forget and one
    /// contradicted at a real other speaker's rate is never forgotten at
    /// all. See [`CONFLICT_WEIGHT`]. Values below `1` are treated as `1`.
    pub conflict_forget_windows: u32,
}

impl Default for OtherSpeakerConfig {
    fn default() -> Self {
        Self {
            seed_ceiling_frac: 1.0,
            guard_frac: 0.85,
            min_run: 2,
            agree: 0.45,
            merge: 0.45,
            eta: 0.2,
            max_clusters: 6,
            conflict_forget_windows: 3,
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
    /// Which cluster the accumulated contradiction is about, and how
    /// much of it there is in [`CONFLICT_WEIGHT`] units. Indices are
    /// positional, so anything that removes a centroid must clear this.
    conflict_cluster: Option<usize>,
    conflict_evidence: u32,
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
        self.forget_conflict();
    }

    /// Drop the accumulated contradiction. Called whenever the centroid
    /// indices shift, since [`Self::conflict_cluster`] is positional and
    /// a stale index would credit one cluster's evidence to another.
    fn forget_conflict(&mut self) {
        self.conflict_cluster = None;
        self.conflict_evidence = 0;
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

    /// Score `emb` for the gate, forgetting a cluster that persistently
    /// conflicts with otherwise-valid enrolled-speaker evidence.
    ///
    /// A low target score is not proof that a window came from somebody
    /// else: a new microphone or acoustic condition can make the owner
    /// miss the enrollment profile several times and seed a cluster. If
    /// later windows clear `theta_pass` but the same cluster would still
    /// block them by `margin`, count that against the cluster. Once
    /// [`OtherSpeakerConfig::conflict_forget_windows`] windows' worth has
    /// accumulated, discard the suspect cluster and return the next-best
    /// other score.
    ///
    /// Windows that do not conflict cancel one unit of that evidence
    /// rather than clearing it, so a real impostor's isolated target
    /// spikes stay pinned near zero and remain blocked, while an owner
    /// who merely straddles the threshold still recovers. `theta_pass`
    /// must be the threshold the gate itself admits on, on the same
    /// scale as `target_score` — see the module docs on `adaptive_theta`.
    pub fn score_for_gate(
        &mut self,
        emb: &[f32],
        target_score: f32,
        theta_pass: f32,
        margin: f32,
        config: &OtherSpeakerConfig,
    ) -> f32 {
        let Some((idx, best)) = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, cos_similarity(emb, c)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
        else {
            self.forget_conflict();
            return 0.0;
        };

        // The cluster is only contradicted when it is the *sole* reason
        // this window would be refused: the enrolled speaker cleared the
        // threshold on their own and only the margin stands in the way.
        let conflicts = target_score >= theta_pass && target_score - best < margin;
        if !conflicts {
            self.conflict_evidence = self.conflict_evidence.saturating_sub(1);
            if self.conflict_evidence == 0 {
                self.conflict_cluster = None;
            }
            return best.max(0.0);
        }

        if self.conflict_cluster == Some(idx) {
            self.conflict_evidence = self.conflict_evidence.saturating_add(CONFLICT_WEIGHT);
        } else {
            self.conflict_cluster = Some(idx);
            self.conflict_evidence = CONFLICT_WEIGHT;
        }
        let forget_at = config
            .conflict_forget_windows
            .max(1)
            .saturating_mul(CONFLICT_WEIGHT);
        if self.conflict_evidence < forget_at {
            return best.max(0.0);
        }

        self.centroids.remove(idx);
        self.counts.remove(idx);
        self.forget_conflict();
        self.score(emb)
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
        if config.max_clusters == 0 {
            self.run = 0;
            self.pending = None;
            return;
        }
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
                let seed: Vec<f32> = prev.iter().zip(emb).map(|(&a, &b)| 0.5 * (a + b)).collect();
                if profile_score(&seed) < guard {
                    if self.centroids.len() >= config.max_clusters {
                        let victim = self
                            .counts
                            .iter()
                            .enumerate()
                            .min_by_key(|(_, &c)| c)
                            .map_or(0, |(i, _)| i);
                        self.centroids.remove(victim);
                        self.counts.remove(victim);
                        self.forget_conflict();
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

    /// Unit vector whose cosine against axis 0 is exactly `target_score`.
    fn at_target_score(target_score: f32) -> Vec<f32> {
        vec![target_score, (1.0 - target_score * target_score).sqrt()]
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
    fn zero_cluster_budget_disables_learning_without_panicking() {
        let cfg = OtherSpeakerConfig {
            max_clusters: 0,
            ..OtherSpeakerConfig::default()
        };
        let mut m = OtherSpeakers::new();
        let profile = |e: &[f32]| cos_similarity(e, &unit(8, 0));
        for _ in 0..10 {
            m.observe(&unit(8, 1), 0.10, 0.45, &cfg, profile);
        }
        assert!(m.is_empty());
    }

    #[test]
    fn conflict_forget_window_overflow_is_saturated() {
        let cfg = OtherSpeakerConfig {
            conflict_forget_windows: u32::MAX,
            ..OtherSpeakerConfig::default()
        };
        let mut m = OtherSpeakers::new();
        let target = unit(2, 0);
        let profile = |e: &[f32]| cos_similarity(e, &target);
        let other = at_target_score(0.30);
        for _ in 0..3 {
            m.observe(&other, 0.30, 0.45, &cfg, profile);
        }
        assert_eq!(m.len(), 1);
        // This used to overflow while calculating the forget threshold
        // in debug builds. One conflict must not remove the cluster.
        let score = m.score_for_gate(&at_target_score(0.50), 0.50, 0.45, 0.08, &cfg);
        assert!(score > 0.90);
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn persistent_target_conflict_forgets_a_mislearned_owner_cluster() {
        let cfg = OtherSpeakerConfig::default();
        let mut m = OtherSpeakers::new();
        let target = unit(2, 0);
        let profile = |e: &[f32]| cos_similarity(e, &target);

        // A changed acoustic condition of the enrolled speaker can score
        // below both the pass threshold and the centroid guard. Repeated
        // windows therefore look exactly like a learnable other speaker.
        let low_owner = at_target_score(0.30);
        for _ in 0..3 {
            m.observe(&low_owner, 0.30, 0.45, &cfg, profile);
        }
        assert_eq!(m.len(), 1, "the synthetic owner condition was mislearned");

        // The same voice moves closer to the enrolled profile and now
        // clears the ordinary threshold, while remaining extremely close
        // to the cluster it seeded. The margin blocks the first windows,
        // but consistent target evidence must evict the suspect cluster.
        let recovered_owner = at_target_score(0.50);
        for _ in 1..cfg.conflict_forget_windows {
            let other = m.score_for_gate(&recovered_owner, 0.50, 0.45, 0.08, &cfg);
            assert!(
                other > 0.90,
                "the conflicting cluster should still block briefly"
            );
        }
        let other = m.score_for_gate(&recovered_owner, 0.50, 0.45, 0.08, &cfg);
        assert!(
            other.abs() < f32::EPSILON,
            "the mislearned owner cluster must be forgotten"
        );
        assert!(m.is_empty());
    }

    #[test]
    fn isolated_target_spikes_do_not_forget_a_real_other_speaker() {
        let cfg = OtherSpeakerConfig::default();
        let mut m = OtherSpeakers::new();
        let target = unit(2, 0);
        let profile = |e: &[f32]| cos_similarity(e, &target);
        let other_speaker = at_target_score(0.30);
        for _ in 0..3 {
            m.observe(&other_speaker, 0.30, 0.45, &cfg, profile);
        }
        assert_eq!(m.len(), 1);

        // A real other speaker reaches the enrolled threshold in bursts
        // separated by ordinary windows — measured p95 0.45 against a
        // 0.45 threshold, so about one window in twenty. Two adjacent
        // spikes per six is six times that rate and still below the
        // one-in-three break-even, so the surrounding windows cancel the
        // evidence faster than the spikes build it. Run it long enough
        // that any net upward drift would show.
        let spike = at_target_score(0.50);
        for _ in 0..20 {
            for _ in 0..2 {
                assert!(m.score_for_gate(&spike, 0.50, 0.45, 0.08, &cfg) > 0.90);
            }
            for _ in 0..4 {
                assert!(m.score_for_gate(&other_speaker, 0.30, 0.45, 0.08, &cfg) > 0.90);
            }
        }
        assert_eq!(
            m.len(),
            1,
            "isolated spikes must not erase the impostor model"
        );
    }

    #[test]
    fn an_owner_straddling_the_threshold_is_not_locked_out_for_the_session() {
        let cfg = OtherSpeakerConfig::default();
        let mut m = OtherSpeakers::new();
        let target = unit(2, 0);
        let profile = |e: &[f32]| cos_similarity(e, &target);

        let low_owner = at_target_score(0.30);
        for _ in 0..3 {
            m.observe(&low_owner, 0.30, 0.45, &cfg, profile);
        }
        assert_eq!(m.len(), 1, "the synthetic owner condition was mislearned");

        // The owner's condition improves, but not cleanly: every second
        // window still falls just short of the threshold. Demanding
        // consecutive conflicts would never complete here, so the
        // cluster would block them for the rest of the session.
        let over = at_target_score(0.50);
        let under = at_target_score(0.44);
        let mut windows = 0_u32;
        while !m.is_empty() {
            let (emb, score) = if windows % 2 == 0 {
                (&over, 0.50)
            } else {
                (&under, 0.44)
            };
            m.score_for_gate(emb, score, 0.45, 0.08, &cfg);
            windows += 1;
            assert!(
                windows < 100,
                "a straddling owner must not be blocked indefinitely"
            );
        }
        // Bounded, and long enough that a genuine impostor at their own
        // much lower rate never gets there.
        assert!(
            windows <= 4 * cfg.conflict_forget_windows,
            "recovery took {windows} windows, expected a small multiple of \
             {}",
            cfg.conflict_forget_windows
        );
    }

    #[test]
    fn evicting_a_cluster_does_not_hand_its_evidence_to_another() {
        let cfg = OtherSpeakerConfig {
            max_clusters: 2,
            ..OtherSpeakerConfig::default()
        };
        let mut m = OtherSpeakers::new();
        let profile = |e: &[f32]| cos_similarity(e, &unit(16, 0));
        for axis in [1_usize, 3] {
            for _ in 0..4 {
                m.observe(&near(16, axis, axis + 8, 0.05), 0.20, 0.45, &cfg, profile);
            }
        }
        assert_eq!(m.len(), 2);

        // Bank contradiction against the cluster at index 1, one window
        // short of forgetting it.
        let contested = near(16, 3, 11, 0.05);
        for _ in 1..cfg.conflict_forget_windows {
            m.score_for_gate(&contested, 0.50, 0.45, 0.08, &cfg);
        }

        // A third speaker evicts a centroid, shifting every index after
        // it. The banked evidence describes a cluster that has moved, so
        // it must not carry over and retire whatever now sits there.
        for _ in 0..4 {
            m.observe(&near(16, 5, 13, 0.05), 0.20, 0.45, &cfg, profile);
        }
        assert_eq!(m.len(), 2);
        let survivors = m.len();
        m.score_for_gate(&near(16, 5, 13, 0.05), 0.50, 0.45, 0.08, &cfg);
        assert_eq!(
            m.len(),
            survivors,
            "evidence banked before the eviction must not retire a cluster"
        );
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
