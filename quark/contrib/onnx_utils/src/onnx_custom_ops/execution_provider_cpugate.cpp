// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "execution_provider_cpugate.hpp"

#include <condition_variable>
#include <deque>
#include <iostream>
#include <thread>

#include "custom_ops.hpp"
#include "execution_provider.hpp"

extern "C" RYZENAI_EP_EXPORT_API OrtStatus* ORT_API_CALL
RyzenAI_RegisterCPUGateCustomOps(
  OrtSessionOptions* options, const OrtApiBase* api_base
);

namespace ryzenai::CPUGate {
using InitFunction = std::function<void(const OrtKernelInfo*)>;

struct Manager;

struct RemoteOp : Op {
  RemoteOp(Manager* m, OpParams p);

  void Invoke(
    std::vector<const OrtValue*> input, std::vector<OrtValue*> output
  ) override;

  Manager* manager = nullptr;
  OpParams params;

  std::vector<const char*> type_constraint_names;
  std::vector<ONNXTensorElementDataType> type_constraint_values;

  Ort::Op ort_op;
};

struct Manager : Interface {
  ~Manager() {
    {
      std::unique_lock<std::mutex> lock(guard_);
      stopping_ = true;
    }

    cv_.notify_all();

    if (thread_.joinable()) thread_.join();
  }

  std::shared_ptr<Op> CreateOp(OpParams params) override {
    auto op = std::make_shared<RemoteOp>(this, std::move(params));

    RunOnKernelConstruction([op](const OrtKernelInfo* info) {
      op->ort_op = Ort::Op::Create(
        info, op->params.op_name.c_str(), op->params.domain.c_str(),
        op->params.version, op->type_constraint_names.data(),
        op->type_constraint_values.data(), op->params.type_constraint.size(),
        op->params.attrs.data(), op->params.attrs.size(),
        op->params.input_count, op->params.output_count
      );
    });

    return op;
  }

  void RunOnKernelConstruction(InitFunction init) {
    std::lock_guard<std::mutex> lock(guard_);
    init_functions_.emplace_back(init);
  }

  void Initialize() override {
    std::unique_lock<std::mutex> lock(guard_);

    if (session_ || init_functions_.size() == 0) return;

    Ort::SessionOptions so;

    so.AddConfigEntry(
      "manager", std::to_string(reinterpret_cast<std::uintptr_t>(this)).c_str()
    );

    so.DisableCpuMemArena();
    so.DisableMemPattern();

    RegisterCustomOpsOverride = &RyzenAI_RegisterCPUGateCustomOps;
    Defer resetRegisterCustomOpsOverride{[]() {
      RegisterCustomOpsOverride = nullptr;
    }};
#ifdef _WIN32
    so.RegisterCustomOpsLibrary(L"onnxruntime_providers_ryzenai.dll");
#else
    so.RegisterCustomOpsLibrary("libonnxruntime_providers_ryzenai.so");
#endif

    so.AddConfigEntry("session.inter_op.allow_spinning", "0");
    so.AddConfigEntry("session.intra_op.allow_spinning", "0");

#if 0
    // this approach does not work due to probably a bug in model editor
    // specific op resolving
    //
    // "SetOpSchemaFromRegistryForNode(node);" call in "Status
    // Graph::VerifyNodeAndOpMatch(const ResolveOptions & options)" at
    // "onnxruntime\core\graph\graph.cc" fails to lookup previously registered
    // custom ops; basically with this kind of model custom ops are ignored or
    // not loaded somehow
    //
    // as a workaround we'll load model from blob
    //

    std::vector<Ort::Model::DomainOpsetPair> ops;
    ops.emplace_back("", 23);
    ops.emplace_back("com.ryzenai.cpu", 1);
    Ort::Model model{ops};

    Ort::TensorTypeAndShapeInfo tt(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1});
    Ort::TypeInfo tinfo = Ort::TypeInfo::CreateTensorInfo(tt.GetConst());

    std::vector<Ort::ValueInfo> inputs;
    inputs.emplace_back("i", tinfo.GetConst());

    std::vector<Ort::ValueInfo> outputs;
    outputs.emplace_back("o", tinfo.GetConst());

    Ort::Graph graph;

    graph.SetInputs(inputs);
    graph.SetOutputs(outputs);

    std::vector<Ort::OpAttr> attrs;
    Ort::Node node("Gate", "com.ryzenai.cpu", "theGate", {"i"}, {"o"}, attrs);

    graph.AddNode(node);
    model.AddGraph(graph);

    session_ = std::make_shared<Ort::Session>(env_, model, so);

    // old working model generation code for reference
    //
    /*
      auto mdl = ONNX_NAMESPACE::Model::Create("RyzenCPUGate", false, {});
      auto& gr = mdl->MainGraph();

      const auto uint8 = ONNX_NAMESPACE::TypeProto::Create();

      uint8->mutable_tensor_type()->set_elem_type(
        ONNX_NAMESPACE::TensorProto_DataType_UINT8
      );

      NodeArg* const i = &gr.GetOrCreateNodeArg("i", uint8.get());
      NodeArg* const o = &gr.GetOrCreateNodeArg("o", uint8.get());

      gr.AddNode(
        "theGate", "Gate", "", {&i, 1}, {&o, 1}, {}, "com.ryzenai.cpu"
      );

      gr.SetInputs({&i, 1});
      gr.SetOutputs({&o, 1});

      auto proto = mdl->ToProto();
      auto op_imp = proto->add_opset_import();

      *op_imp->mutable_domain() = "com.ryzenai.cpu";
      op_imp->set_version(1);

      return proto->SerializeAsString();
    */

#else
    static const unsigned char model_blob[] = {
      0x08, 0x09, 0x12, 0x03, 0x52, 0x41, 0x49, 0x3A, 0x4C, 0x0A, 0x26, 0x0A,
      0x01, 0x69, 0x12, 0x01, 0x6F, 0x1A, 0x07, 0x74, 0x68, 0x65, 0x47, 0x61,
      0x74, 0x65, 0x22, 0x04, 0x47, 0x61, 0x74, 0x65, 0x3A, 0x0F, 0x63, 0x6F,
      0x6D, 0x2E, 0x72, 0x79, 0x7A, 0x65, 0x6E, 0x61, 0x69, 0x2E, 0x63, 0x70,
      0x75, 0x12, 0x0C, 0x52, 0x79, 0x7A, 0x65, 0x6E, 0x43, 0x50, 0x55, 0x47,
      0x61, 0x74, 0x65, 0x5A, 0x09, 0x0A, 0x01, 0x69, 0x12, 0x04, 0x0A, 0x02,
      0x08, 0x02, 0x62, 0x09, 0x0A, 0x01, 0x6F, 0x12, 0x04, 0x0A, 0x02, 0x08,
      0x02, 0x42, 0x13, 0x0A, 0x0F, 0x63, 0x6F, 0x6D, 0x2E, 0x72, 0x79, 0x7A,
      0x65, 0x6E, 0x61, 0x69, 0x2E, 0x63, 0x70, 0x75, 0x10, 0x01
    };

    // following python code can be used to regenerate the blob
    //

    /*
    import onnx

    m = onnx.ModelProto(
        ir_version=9,
        producer_name="RAI",
        graph=onnx.GraphProto(
            name="RyzenCPUGate",
            node=[
                onnx.NodeProto(
                    name="theGate",
                    op_type="Gate",
                    domain="com.ryzenai.cpu",
                    input=["i"],
                    output=["o"],
                )
            ],
            input=[
                onnx.ValueInfoProto(
                    name="i",
                    type=onnx.TypeProto(
                        tensor_type=onnx.TypeProto.Tensor(elem_type=2)
                    ),
                )
            ],
            output=[
                onnx.ValueInfoProto(
                    name="o",
                    type=onnx.TypeProto(
                        tensor_type=onnx.TypeProto.Tensor(elem_type=2)
                    ),
                )
            ],
        ),
        opset_import=[
            onnx.OperatorSetIdProto(domain="com.ryzenai.cpu", version=1)
        ],
    ).SerializeToString()

    print(", ".join(f"0x{b:02X}" for b in m))
    */

    session_ =
      std::make_shared<Ort::Session>(env_, model_blob, sizeof(model_blob), so);
#endif

    // At this point OnKernelConstructed has been called by the Gate kernel. So,
    // all cpu Ort::Op objects are constructed.

    // Now we need to find OrtKernelContext* from cpu session, that we're gonna
    // use to invoke already constructed Ort::Op objects. To do so we're gonna
    // create a thread that runs our model, main kernel of which is going to
    // call our OnKernelComputeCalled. The method is going to store the context
    // and block the thread to keep context object alive. It's important because
    // the context object is stack allocated and gets destroyed once,
    // Kernel::Compute exits.

    thread_ = std::thread{[session = session_] {
      constexpr int64_t zero = 0;
      constexpr const char* input_names[] = {"i"};
      constexpr const char* output_names[] = {"o"};
      Ort::Value tensors[1] = {Ort::Value::CreateTensor(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
        nullptr, 0, &zero, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
      )};

      session->Run({}, input_names, tensors, 1, output_names, tensors, 1);
    }};

    // OnKernelComputeCalled is expected to notify us when kernel_ctx_ is set
    // and it's safe to proceed.
    if (std::cv_status::timeout == cv_.wait_for(lock, std::chrono::seconds(5)))
      throw std::runtime_error{"CPU Gate failed to initialize"};
  }

  void OnKernelConstructed(const OrtKernelInfo* info) {
    if (guard_.try_lock())
      throw std::runtime_error{
        "Was supposed to be locked in Manager::Initialize()"
      };
    for (const auto& init : init_functions_) init(info);
    init_functions_.clear();
  }

  void OnKernelComputeCalled(OrtKernelContext* context) {
    std::unique_lock<std::mutex> lock(guard_);
    kernel_ctx_ = context;
    cv_.notify_one();
    while (!stopping_) cv_.wait(lock);
    kernel_ctx_ = nullptr;
  }

  void InvokeOrtOp(
    Ort::Op& op, std::vector<const OrtValue*> input,
    std::vector<OrtValue*> output
  ) {
    std::lock_guard<std::mutex> lock(guard_);

    op.Invoke(
      kernel_ctx_, input.data(), input.size(), output.data(), output.size()
    );
  }

 private:
  Ort::Env env_{ORT_LOGGING_LEVEL_ERROR, "RyzenAICpuGateEnv"};

  std::shared_ptr<Ort::Session> session_;
  std::deque<InitFunction> init_functions_;
  std::thread thread_;
  std::condition_variable cv_;
  bool stopping_{false};
  OrtKernelContext* kernel_ctx_ = nullptr;
  std::mutex guard_;
};

RemoteOp::RemoteOp(Manager* m, OpParams p) : manager(m), params(std::move(p)) {
  type_constraint_names.reserve(params.type_constraint.size());
  type_constraint_values.reserve(params.type_constraint.size());

  for (const auto& [key, val] : params.type_constraint) {
    type_constraint_names.push_back(key.c_str());
    type_constraint_values.push_back(val);
  }
}

void RemoteOp::Invoke(
  std::vector<const OrtValue*> input, std::vector<OrtValue*> output
) {
  manager->InvokeOrtOp(ort_op, std::move(input), std::move(output));
}

std::shared_ptr<Interface> CreateInstance() {
  return std::make_shared<Manager>();
}

struct Kernel {
  Kernel(const OrtApi&, const OrtKernelInfo* info, Manager* manager)
    : manager_(manager) {
    manager_->OnKernelConstructed(info);
  }

  void Compute(OrtKernelContext* context) {
    manager_->OnKernelComputeCalled(context);
  }

 private:
  Manager* manager_;
};

struct CustomOperator
  : Ort::CustomOpBase<CustomOperator, ryzenai::CPUGate::Kernel> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    const auto str = last_session_options.GetConfigEntry("manager");
    const auto ptr =
      reinterpret_cast<Manager*>(uintptr_t(std::atoll(str.c_str())));

    return new ryzenai::CPUGate::Kernel(api, info, ptr);
  };

  const char* GetName() const noexcept { return "Gate"; }
  const char* GetExecutionProviderType() const noexcept {
    return "CPUExecutionProvider";
  }

  size_t GetInputTypeCount() const noexcept { return 1; }
  size_t GetOutputTypeCount() const noexcept { return 1; }

  ONNXTensorElementDataType GetInputType(size_t) const noexcept {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  ONNXTensorElementDataType GetOutputType(size_t) const noexcept {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }

  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(
    size_t
  ) const noexcept {
    return INPUT_OUTPUT_VARIADIC;
  }
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(
    size_t
  ) const noexcept {
    return INPUT_OUTPUT_VARIADIC;
  }

  bool GetVariadicInputHomogeneity() const noexcept { return false; }
  bool GetVariadicOutputHomogeneity() const noexcept { return false; }

  Ort::ConstSessionOptions last_session_options;
};
}  // namespace ryzenai::CPUGate

extern "C" {
RYZENAI_EP_EXPORT_API OrtStatus* ORT_API_CALL RyzenAI_RegisterCPUGateCustomOps(
  OrtSessionOptions* options, const OrtApiBase* api_base
) {
  static Ort::CustomOpDomain domain{"com.ryzenai.cpu"};
  static ryzenai::CPUGate::CustomOperator cpu_gate_op;

  cpu_gate_op.last_session_options = Ort::ConstSessionOptions{options};

  OrtStatus* result = nullptr;

  try {
    Ort::UnownedSessionOptions session_options(options);
    domain.Add(&cpu_gate_op);
    session_options.Add(domain);
  } catch (const std::exception& e) {
    Ort::Status status{e};
    result = status.release();
  }

  return nullptr;
}
}
