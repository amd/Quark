// Copyright (c) 2024 Advanced Micro Devices, Inc.

/**
 * @brief By inheriting from the JitNode class, all constructed instances of the
 * child class are added to a list. Since ORT constructs custom operators in
 * order of the nodes' in the model, the model can be topologically sorted to
 * ensure execution order matches the construction order.
 *
 * JIT loading works in two stages: reading and loading. Reading brings
 * in the data from file to memory in one of the shared buffers. Each child
 * class creates a number of these shared buffers which work as a circular
 * buffer. Loading loads one of these initialized shared buffers to the device
 * (in the NPU case, this binds the buffer to the const xrt::BO).
 *
 * The readAhead value (defaulting to kJitReadAhead but user configurable)
 * defines how many shared buffers to create i.e. how far ahead to prefetch data
 * and keep it read in memory. Lowering this value directly lowers the memory
 * requirements of the application. You can set it to a negative number to
 * disable JIT loading (all weights are read at construction time).
 *
 * The loadAhead value (defaulting to kJitLoadAhead) defines how far in advance
 * to load buffers to the device.
 *
 * For the NPU, loadAhead must be 0 or 1, indicating that the node will either
 * load its own weights prior to execution or it will load the next node's
 * weights after its own execution. The readAhead value must be at least 1 and
 * it must be strictly larger than the loadAhead value.
 *
 */

#pragma once

#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../session_state.hpp"

namespace ryzenai {

namespace RyzenMM {
struct BufferRef;
}
namespace onnx_utils {

/// @brief How far to read ahead from file to memory. Must be >=1 and >loadAhead
constexpr int kJitReadAhead = 3;
/// @brief How far to load ahead from memory to device. Must be 0 or 1 for NPU
constexpr int kJitLoadAhead = 1;

enum class JitState { Unloaded, Reading, Read, Loading, Loaded, Unloading };

template <typename T>
class JitNode {
 public:
  explicit JitNode(
    JitNode<T>* op_inter,
    const std::unordered_map<std::string, std::string>& session_configs
  );
  virtual ~JitNode() = default;

  /**
   * @brief Preload the next set of weights. This must be called after the
   * current operator has finished execution.
   */
  void loadData() const;
  /**
   * @brief Read the data for the first readAhead() nodes and load the data
   * for the first kJitLoadAhead nodes. This should be called in the op
   * constructor.
   */
  void loadFirstData();
  void readData() const;

  void unloadData() const;

  int weightsReady() const;

 protected:
  virtual void readDataImpl(int) = 0;
  virtual void loadDataImpl(int) = 0;
  virtual void unloadDataImpl(int) = 0;

  int readAhead() const;
  /**
   * @brief Helper function for child classes to initialize their JIT buffers.
   * This must be called from the class constructor.
   *
   * @param buffers vector of buffers
   * @param sizes vector of buffer sizes
   */
  void initializeSharedBuffers(std::vector<RyzenMM::BufferRef>& buffers) const;

  bool isJitEnabled() const;

 private:
  int idx_;
  int bo_idx_;

  struct State {
    int offset = 0;
    std::vector<JitNode<T>*> topo_order = {};
    std::vector<JitState> jit_state = {};
    std::vector<std::future<void>> jit_futures = {};
    int load_offset = 0;
    int read_offset = 0;
    int read_ahead = kJitReadAhead;
    bool jit_enable = true;
  };

  SessionState<State> ss_;

  void setReadAhead(
    const std::unordered_map<std::string, std::string>& session_configs
  );

  void loadJitData(int idx, bool is_constructor);
};

}  // namespace onnx_utils
}  // namespace ryzenai
