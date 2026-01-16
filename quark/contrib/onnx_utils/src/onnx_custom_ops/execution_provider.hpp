// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once
#include <chrono>
#include <cstring>
#include <iterator>
#include <memory>
#include <utility>

#define RYZENAI_EP_PERFORMANCE_COUNTERS_DEBUG 0

#if defined(_MSC_VER)
#define RYZENAI_EP_FORCE_INLINE __forceinline
#define RYZENAI_EP_NOINLINE __declspec(noinline)
#define RYZENAI_EP_EXPORT_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define RYZENAI_EP_FORCE_INLINE inline __attribute__((always_inline))
#define RYZENAI_EP_NOINLINE
#define RYZENAI_EP_EXPORT_API __attribute__((visibility("default")))
#else
#define RYZENAI_EP_FORCE_INLINE inline
#define RYZENAI_EP_NOINLINE
#define RYZENAI_EP_EXPORT_API
#endif

namespace ryzenai {

enum class Metric : uint32_t {
  KernelExecution,
  Casting,
  Tiling,
  MemCpy,
  Transpose,
  KVCacheUpdate,
  SplitQKV,
  XRTBOSync,
  Max
};

static const char* const MetricNames[] = {"kernel exec", "casting",
                                          "tiling",      "memcpy",
                                          "transpose",   "kvcupdate",
                                          "splitqkv",    "xrtbosync"};

static_assert(std::size(MetricNames) == static_cast<size_t>(Metric::Max));

namespace CPUGate {
struct Interface;
struct OpParams;
struct Op;
}  // namespace CPUGate

template <typename F>
struct Defer {
  Defer(F f) noexcept : f_(std::move(f)) {}

  ~Defer() noexcept {
    try {
      f_();
    } catch (...) {
    }
  }

 private:
  F f_;
};

struct IExecutionProvider {
  virtual ~IExecutionProvider() = default;

  using Clock = std::chrono::high_resolution_clock;

  virtual const std::string& GetSessionID() const = 0;

  virtual ryzenai::CPUGate::Interface& GetCPUGate() = 0;

  virtual bool IsPerformanceCountersEnabled() = 0;

  virtual void RecordDuration(Metric, Clock::duration) = 0;
  virtual void QueryPerformanceCounters(
    const char* const** names, uint32_t* count, uint64_t* values
  ) const = 0;
};

// must be called only from kernel constructor
IExecutionProvider* GetCurrentExecutionProvider();

struct ExecutionProviderExtensions {
  RYZENAI_EP_FORCE_INLINE bool IsEPAvailable() const noexcept { return ep_; };

  template <typename Func>
  RYZENAI_EP_FORCE_INLINE void RecordDuration(
    Metric metric, Func&& func
  ) const {
    if (!profiling_) return func();

#if RYZENAI_EP_PERFORMANCE_COUNTERS_DEBUG
    // first we want to make sure there are no overlaps

    static thread_local bool currentThreadIsProfiled = false;

    if (std::exchange(currentThreadIsProfiled, true)) abort();

    Defer resetCurrentThreadIsProfiled{[&] {
      currentThreadIsProfiled = false;
    }};

    // since currently we do not care about background thread activity, we want
    // to make sure we're not called from non-main thread

    static std::optional<std::thread::id> profiledTID;

    if (!profiledTID.has_value())
      profiledTID = std::this_thread::get_id();
    else if (profiledTID.value() != std::this_thread::get_id())
      abort();
#endif

    const auto start = IExecutionProvider::Clock::now();

    func();

    ep_->RecordDuration(metric, IExecutionProvider::Clock::now() - start);
  }

  RYZENAI_EP_FORCE_INLINE void MemCpy(
    void* dst, const void* src, size_t size
  ) const {
    RecordDuration(Metric::MemCpy, [=]() { std::memcpy(dst, src, size); });
  }

  std::shared_ptr<CPUGate::Op> CreateCPUOp(CPUGate::OpParams&& params);

 private:
  IExecutionProvider* const ep_ = GetCurrentExecutionProvider();
  const bool profiling_ = ep_ ? ep_->IsPerformanceCountersEnabled() : false;
};

};  // namespace ryzenai
