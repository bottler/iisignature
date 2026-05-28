#ifndef IISIGNATURE_THREADING_H
#define IISIGNATURE_THREADING_H

// Tiny threading primitives shared by parallel kernels. Header-only,
// no Python / NumPy dependency, so it can be reused from a C++ caller.
//
// Two pieces:
//   * iisignature_num_threads() / iisignature_default_thread_count()
//     -- resolve the worker count once from IISIGNATURE_NUM_THREADS,
//        then OMP_NUM_THREADS, then sched_getaffinity (Linux) /
//        std::thread::hardware_concurrency.
//   * parallel_for(n, nThreads, errOut, body)
//     -- static block partitioning across std::thread workers; catches
//        per-worker std::exception and surfaces the first message.

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

// Default-pool resolution: pool size is resolved (once, cached) from
// three sources, in priority order:
//   1. IISIGNATURE_NUM_THREADS env var (iisignature-specific override)
//   2. OMP_NUM_THREADS env var (scientific-Python convention, what
//      threadpoolctl.threadpool_limits() sets under the hood)
//   3. all cores the process is allowed to use (affinity-aware on Linux)
//
// Default is "all cores" so that a fresh user gets parallelism for free,
// matching numpy/scipy/sklearn behaviour. Sub-paths shorter than nThreads
// fall back to the sequential code path inside the caller, so short-call
// workloads don't pay thread-spawn overhead.
inline int iisignature_default_thread_count() {
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0) {
    int n = CPU_COUNT(&set);
    if (n > 0)
      return n;
  }
#endif
  unsigned hw = std::thread::hardware_concurrency();
  return hw ? (int)hw : 1;
}

inline int iisignature_num_threads() {
  static int cached = []() {
    auto parse = [](const char *s) -> int {
      if (!s || !*s)
        return 0;
      int v = std::atoi(s);
      return v > 0 ? v : 0;
    };
    int v = parse(std::getenv("IISIGNATURE_NUM_THREADS"));
    if (!v)
      v = parse(std::getenv("OMP_NUM_THREADS"));
    if (!v)
      v = iisignature_default_thread_count();
    int hw = iisignature_default_thread_count();
    if (hw > 0 && v > hw)
      v = hw;
    return v;
  }();
  return cached;
}

// Run body(t, i) for i in [0, n) across `nThreads` workers using static block
// partitioning. `t` is the thread id in [0, nThreads). Caller is expected to
// have released the GIL. Each worker catches std::exception so a single
// failure doesn't abort other threads; the first exception message is
// returned via `errOut`.
template <typename Body>
void parallel_for(int n, int nThreads, std::string &errOut, Body &&body) {
  if (n <= 0)
    return;
  if (nThreads > n)
    nThreads = n;
  if (nThreads <= 1) {
    try {
      for (int i = 0; i < n; ++i)
        body(0, i);
    } catch (std::exception &e) {
      errOut = e.what();
    }
    return;
  }
  std::vector<std::thread> workers;
  workers.reserve(nThreads - 1);
  std::atomic<bool> failed{false};
  std::string firstErr;
  std::mutex errMu;
  auto run = [&](int t) {
    int chunk = n / nThreads;
    int rem = n % nThreads;
    int begin = t * chunk + std::min(t, rem);
    int end = begin + chunk + (t < rem ? 1 : 0);
    try {
      for (int i = begin; i < end; ++i) {
        if (failed.load(std::memory_order_relaxed))
          return;
        body(t, i);
      }
    } catch (std::exception &e) {
      bool expected = false;
      if (failed.compare_exchange_strong(expected, true)) {
        std::lock_guard<std::mutex> lk(errMu);
        firstErr = e.what();
      }
    }
  };
  for (int t = 1; t < nThreads; ++t)
    workers.emplace_back(run, t);
  run(0);
  for (auto &w : workers)
    w.join();
  if (!firstErr.empty())
    errOut = firstErr;
}

#endif
