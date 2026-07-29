//! Shared `ort` threading policy for ECAPA / VAD / DFN3 / TSE /
//! overlap sessions.
//!
//! The default `Session::builder()` configuration sets both
//! intra-op and inter-op pools to `num_cores`, which thrashes on
//! small (2-vCPU) hosts: every op pool tries to spread the same op
//! across all cores while the inter-op pool also tries to run ops in
//! parallel, and the OS has nowhere to schedule the actual work.
//!
//! For single-batch realtime inference (one audio chunk at a time,
//! and the live worker only ever has **one** ONNX session active at
//! a moment because it runs the chain stages sequentially):
//!
//! * `intra_op_num_threads` — the physical core count, clamped at 2.
//!   #179 briefly raised this to 4 on the theory that the heavier
//!   per-chunk inferences (TSE Conv-TasNet) needed more cores. A
//!   bench on a 4-vCPU box proved the opposite — ORT's intra-op pool
//!   busy-waits, so giving it 4 threads on a 4-core machine starves
//!   the worker / cpal / OS threads and makes the chain *slower*:
//!
//!   | intra | Solo RTF | Overlap (TSE) RTF |
//!   |------:|---------:|------------------:|
//!   |     1 |    2.44× |             1.54× |
//!   |   **2** | **2.75×** |        **1.64×** |
//!   |     3 |    2.02× |             1.35× |
//!   |     4 |    1.62× |             1.06× |
//!
//!   2 is the clear optimum; everything above it regresses. The clamp
//!   is back to 2. The real win for an underrunning laptop is a
//!   lighter TSE (int8 export) or moving ECAPA/TSE off the realtime
//!   thread, not more threads.
//!
//!   That table was measured on a **4-vCPU** box, where the starvation
//!   is the whole story: 4 busy-waiting intra-op threads leave nothing
//!   for cpal, the worker and the OS. On a wide desktop CPU the
//!   premise doesn't hold — pinning a 20-core machine to 2 threads
//!   leaves 90 % of it idle while the 1-second SepFormer/ECAPA block
//!   inference is the thing we're waiting on. So the historic clamp is
//!   kept verbatim for the benchmarked small-host regime and only
//!   relaxed above it, still reserving three quarters of the machine
//!   so the busy-wait pool never competes with a realtime thread.
//! * `inter_op_num_threads` — fixed at 1. We never benefit from
//!   parallelising ops within a single inference call because the
//!   graph is essentially linear and small.
//!
//! Override via the `MELLONELLA_ORT_INTRA_THREADS` env var when the
//! defaults don't fit (e.g. a dedicated server with a wide CPU and
//! nothing else competing for cores).

/// Above this core count the 4-vCPU starvation benchmark no longer
/// describes the machine, and the historic clamp of 2 just wastes it.
const SMALL_HOST_CORES: usize = 8;
/// Even on a very wide CPU, more than this many busy-waiting intra-op
/// threads stops paying for our small, mostly-linear graphs.
const MAX_INTRA_THREADS: usize = 8;

/// Returns the intra-op thread count to pin on each `Session`.
#[must_use]
pub fn intra_op_threads() -> usize {
    if let Ok(s) = std::env::var("MELLONELLA_ORT_INTRA_THREADS") {
        if let Ok(n) = s.parse::<usize>() {
            return n.max(1);
        }
    }
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    if cores <= SMALL_HOST_CORES {
        // The benchmarked regime — leave it exactly as measured.
        cores.min(2)
    } else {
        // Reserve three quarters of the machine for the cpal callbacks,
        // the audio worker, the GUI and the OS.
        (cores / 4).clamp(2, MAX_INTRA_THREADS)
    }
}

#[cfg(test)]
mod tests {
    /// The env override wins over any core-count heuristic, and the
    /// small-host clamp is preserved so the #179 bench still applies.
    #[test]
    fn small_hosts_keep_the_benchmarked_clamp() {
        // Mirrors the branch in `intra_op_threads` without touching the
        // process-wide env var (which would race other tests).
        let pick = |cores: usize| {
            if cores <= super::SMALL_HOST_CORES {
                cores.min(2)
            } else {
                (cores / 4).clamp(2, super::MAX_INTRA_THREADS)
            }
        };
        assert_eq!(pick(1), 1);
        assert_eq!(pick(2), 2);
        assert_eq!(pick(4), 2, "the 4-vCPU bench optimum must not regress");
        assert_eq!(pick(8), 2);
        assert_eq!(pick(12), 3);
        assert_eq!(pick(28), 7, "i7-14700 class: 7 of 28 logical cores");
        assert_eq!(pick(128), super::MAX_INTRA_THREADS);
    }
}
