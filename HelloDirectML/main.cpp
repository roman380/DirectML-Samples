// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"

using winrt::com_ptr;
using winrt::check_hresult;
using winrt::check_bool;

namespace D3D
{
    class Context
    {
    public:
        com_ptr<ID3D12Device> m_Device;
        com_ptr<ID3D12CommandQueue> m_CommandQueue;
        com_ptr<ID3D12CommandAllocator> m_CommandAllocator;
        com_ptr<ID3D12GraphicsCommandList> m_CommandList;

    public:
        void Create()
        {
            WINRT_ASSERT(!m_Device);
            WINRT_ASSERT(!m_CommandQueue && !m_CommandAllocator && !m_CommandList);
            #if defined(_DEBUG)
                com_ptr<ID3D12Debug> Debug;
                if(FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(Debug.put()))))
                    winrt::throw_hresult(DXGI_ERROR_SDK_COMPONENT_MISSING);
                Debug->EnableDebugLayer();
            #endif
            com_ptr<IDXGIFactory4> DxgiFactory;
            check_hresult(CreateDXGIFactory1(IID_PPV_ARGS(DxgiFactory.put())));
            for(UINT DxgiAdapterIndex = 0; ; DxgiAdapterIndex++)
            {
                com_ptr<IDXGIAdapter> DxgiAdapter;
                check_hresult(DxgiFactory->EnumAdapters(DxgiAdapterIndex, DxgiAdapter.put()));
                const HRESULT CreateDeviceResult = D3D12CreateDevice(DxgiAdapter.get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(m_Device.put()));
                if(CreateDeviceResult == DXGI_ERROR_UNSUPPORTED) 
                    continue;
                check_hresult(CreateDeviceResult);
                break;
            }
            D3D12_COMMAND_QUEUE_DESC CommandQueueDesc { D3D12_COMMAND_LIST_TYPE_DIRECT };
            CommandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
            check_hresult(m_Device->CreateCommandQueue(&CommandQueueDesc, IID_PPV_ARGS(m_CommandQueue.put())));
            check_hresult(m_Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(m_CommandAllocator.put())));
            check_hresult(m_Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_CommandAllocator.get(), nullptr, IID_PPV_ARGS(m_CommandList.put())));
        }
        com_ptr<ID3D12Resource> CreateResource(const D3D12_HEAP_PROPERTIES& HeapProperties, const D3D12_RESOURCE_DESC& ResourceDesc, D3D12_RESOURCE_STATES InitialState = D3D12_RESOURCE_STATE_COMMON) const
        {
            WINRT_ASSERT(m_Device);
            com_ptr<ID3D12Resource> Resource;
            check_hresult(m_Device->CreateCommittedResource(&HeapProperties, D3D12_HEAP_FLAG_NONE, &ResourceDesc, InitialState, nullptr, IID_PPV_ARGS(Resource.put())));
            return Resource;
        }
        com_ptr<ID3D12Resource> CreateResource(D3D12_HEAP_TYPE HeadType, const D3D12_RESOURCE_DESC& ResourceDesc, D3D12_RESOURCE_STATES InitialState = D3D12_RESOURCE_STATE_COMMON) const
        {
            return CreateResource(CD3DX12_HEAP_PROPERTIES(HeadType), ResourceDesc, InitialState);
        }
        com_ptr<ID3D12Resource> CreateBufferResource(D3D12_HEAP_TYPE HeadType, UINT64 Width, D3D12_RESOURCE_STATES InitialState = D3D12_RESOURCE_STATE_COMMON) const
        {
            return CreateResource(CD3DX12_HEAP_PROPERTIES(HeadType), CD3DX12_RESOURCE_DESC::Buffer(Width), InitialState);
        }
        void ExecuteCommandListAndWait() const
        {
            WINRT_ASSERT(m_Device && m_CommandQueue && m_CommandAllocator && m_CommandList);
            check_hresult(m_CommandList->Close());
            ID3D12CommandList* CommandLists[] { m_CommandList.get() };
            m_CommandQueue->ExecuteCommandLists(static_cast<UINT>(std::size(CommandLists)), CommandLists);
            check_hresult(m_CommandList->Reset(m_CommandAllocator.get(), nullptr));
            com_ptr<ID3D12Fence> Fence;
            check_hresult(m_Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(Fence.put())));
            winrt::handle FenceEvent { CreateEvent(nullptr, TRUE, FALSE, nullptr) };
            check_bool(bool { FenceEvent });
            check_hresult(Fence->SetEventOnCompletion(1, FenceEvent.get()));
            check_hresult(m_CommandQueue->Signal(Fence.get(), 1));
            WINRT_VERIFY(WaitForSingleObjectEx(FenceEvent.get(), INFINITE, FALSE) == WAIT_OBJECT_0);
        }
        void ResourceBarrier(D3D12_RESOURCE_BARRIER const& Value) const
        {
            WINRT_ASSERT(m_CommandList);
            m_CommandList->ResourceBarrier(1, &Value);
        }
    };

    class DescriptorHeap
    {
    public:
        com_ptr<ID3D12DescriptorHeap> m_Value;

    public:
        DescriptorHeap(Context const& Context, UINT DescriptorCount)
        {
            WINRT_ASSERT(Context.m_Device && Context.m_CommandList);
            const D3D12_DESCRIPTOR_HEAP_DESC DescriptorHeapDesc { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, DescriptorCount, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE };
            check_hresult(Context.m_Device->CreateDescriptorHeap(&DescriptorHeapDesc, IID_PPV_ARGS(m_Value.put())));
            ID3D12DescriptorHeap* DescriptorHeaps[] { m_Value.get() };
            Context.m_CommandList->SetDescriptorHeaps(static_cast<UINT>(std::size(DescriptorHeaps)), DescriptorHeaps);
        }
        ID3D12DescriptorHeap* operator -> () const 
        {
            WINRT_ASSERT(m_Value);
            return m_Value.get();
        }
    };
}

namespace DML
{
    inline UINT64 CalculateBufferTensorSize(DML_TENSOR_DATA_TYPE DataType, UINT DimensionCount, _In_reads_(DimensionCount) const UINT* Sizes, _In_reads_opt_(DimensionCount) const UINT* Strides = nullptr)
    {
        WINRT_ASSERT(DimensionCount && Sizes);
        UINT ElementSize = 0;
        switch(DataType)
        {
        case DML_TENSOR_DATA_TYPE_FLOAT32:
        case DML_TENSOR_DATA_TYPE_UINT32:
        case DML_TENSOR_DATA_TYPE_INT32:
            ElementSize = 4;
            break;
        case DML_TENSOR_DATA_TYPE_FLOAT16:
        case DML_TENSOR_DATA_TYPE_UINT16:
        case DML_TENSOR_DATA_TYPE_INT16:
            ElementSize = 2;
            break;
        case DML_TENSOR_DATA_TYPE_UINT8:
        case DML_TENSOR_DATA_TYPE_INT8:
            ElementSize = 1;
            break;
        default:
            WINRT_ASSERT(FALSE);
            return 0;
        }
        UINT64 Size = 0;
        if(!Strides)
        {
            Size = Sizes[0];
            for(UINT DimensionIndex = 1; DimensionIndex < DimensionCount; ++DimensionIndex)
                Size *= Sizes[DimensionIndex];
            Size *= ElementSize;
        }
        else
        {
            UINT LastIndex = 0;
            for(UINT DimensionIndex = 0; DimensionIndex < DimensionCount; ++DimensionIndex)
                LastIndex += (Sizes[DimensionIndex] - 1) * Strides[DimensionIndex];
            Size = (LastIndex + 1) * ElementSize;
        }
        return (Size + 3) & ~3;
    }
    inline UINT64 CalculateBufferTensorSize(DML_BUFFER_TENSOR_DESC const& BufferTensorDesc)
    {
        return CalculateBufferTensorSize(BufferTensorDesc.DataType, BufferTensorDesc.DimensionCount, BufferTensorDesc.Sizes, BufferTensorDesc.Strides);
    }

    class BindingTable
    {
    public:
        DML_BINDING_TABLE_DESC m_Desc { };
        com_ptr<IDMLBindingTable> m_Value;

    public:
        BindingTable(com_ptr<IDMLDevice> const& Device, D3D::DescriptorHeap const& DescriptorHeap, UINT DescriptorCount, IDMLDispatchable* Dispatchable)
        {
            WINRT_ASSERT(Device);
            WINRT_ASSERT(Dispatchable);
            m_Desc.Dispatchable = Dispatchable;
            m_Desc.CPUDescriptorHandle = DescriptorHeap->GetCPUDescriptorHandleForHeapStart();
            m_Desc.GPUDescriptorHandle = DescriptorHeap->GetGPUDescriptorHandleForHeapStart();
            m_Desc.SizeInDescriptors = DescriptorCount;
            check_hresult(Device->CreateBindingTable(&m_Desc, IID_PPV_ARGS(m_Value.put())));
        }
        void Reset(IDMLDispatchable* Dispatchable)
        {
            WINRT_ASSERT(Dispatchable);
            WINRT_ASSERT(m_Value);
            m_Desc.Dispatchable = Dispatchable;
            check_hresult(m_Value->Reset(&m_Desc));
        }
        void BindInput(DML_BINDING_DESC const& Input) const
        {
            WINRT_ASSERT(m_Value);
            m_Value->BindInputs(1, &Input);
        }
        template <size_t t_Count>
        void BindInputs(DML_BINDING_DESC const (&Inputs)[t_Count])
        {
            WINRT_ASSERT(m_Value);
            m_Value->BindInputs(static_cast<UINT>(t_Count), Inputs);
        }
        void BindOutput(DML_BINDING_DESC const& Output) const
        {
            WINRT_ASSERT(m_Value);
            m_Value->BindOutputs(1, &Output);
        }
        template <size_t t_Count>
        void BindOutputs(DML_BINDING_DESC const (&Outputs)[t_Count])
        {
            WINRT_ASSERT(m_Value);
            m_Value->BindOutputs(static_cast<UINT>(t_Count), Outputs);
        }
        IDMLBindingTable* operator -> () const 
        {
            WINRT_ASSERT(m_Value);
            return m_Value.get();
        }
    };

    class BufferBindingDesc : public DML_BINDING_DESC
    {
    public:
        DML_BUFFER_BINDING m_BufferBinding;

    public:
        BufferBindingDesc(com_ptr<ID3D12Resource> const& Buffer, UINT64 Size) :
            m_BufferBinding { Buffer.get(), 0, Size }
        {
            Type = DML_BINDING_TYPE_BUFFER;
            Desc = &m_BufferBinding;
        }
    };

    class OperatorBuffers
    {
    public:
        UINT64 m_TemporaryResourceSize;
        com_ptr<ID3D12Resource> m_TemporaryBuffer;
        UINT64 m_PersistentResourceSize;
        com_ptr<ID3D12Resource> m_PersistentBuffer;

    public:
        void Create(D3D::Context const& Context, const DML_BINDING_PROPERTIES& InitializeProperties, const DML_BINDING_PROPERTIES& ExecuteProperties)
        {
            // The temporary resource is scratch memory (used internally by DirectML), whose contents you don't need to define.
            // The persistent resource is long-lived, and you need to initialize it using the IDMLOperatorInitializer.
            m_TemporaryResourceSize = std::max(InitializeProperties.TemporaryResourceSize, ExecuteProperties.TemporaryResourceSize);
            if(m_TemporaryResourceSize)
                m_TemporaryBuffer = Context.CreateBufferResource(D3D12_HEAP_TYPE_DEFAULT, m_TemporaryResourceSize);
            m_PersistentResourceSize = ExecuteProperties.PersistentResourceSize;
            if(m_PersistentResourceSize)
                m_PersistentBuffer = Context.CreateBufferResource(D3D12_HEAP_TYPE_DEFAULT, m_PersistentResourceSize);
        }
        void BindToInitialize(BindingTable const& BindingTable) const
        {
            WINRT_ASSERT(BindingTable.m_Value);
            if(m_TemporaryResourceSize)
                BindingTable->BindTemporaryResource(&DML::BufferBindingDesc(m_TemporaryBuffer, m_TemporaryResourceSize));
            if(m_PersistentResourceSize)
                BindingTable.BindOutput(DML::BufferBindingDesc(m_PersistentBuffer, m_PersistentResourceSize)); // Persistent is Initializer Output
        }
        void BindToExecute(BindingTable const& BindingTable) const
        {
            WINRT_ASSERT(BindingTable.m_Value);
            if(m_TemporaryResourceSize)
                BindingTable->BindTemporaryResource(&DML::BufferBindingDesc(m_TemporaryBuffer, m_TemporaryResourceSize));
            if(m_PersistentResourceSize)
                BindingTable->BindPersistentResource(&DML::BufferBindingDesc(m_PersistentBuffer, m_PersistentResourceSize));
        }
    };

    template <size_t t_Count>
    class Operators
    {
    public:
        com_ptr<IDMLOperator> m_Operators[t_Count];
        com_ptr<IDMLCompiledOperator> m_CompiledOperators[t_Count];
        com_ptr<IDMLOperatorInitializer> m_OperatorInitializer;
        DML_BINDING_PROPERTIES m_InitializeBindingProperties;
        DML_BINDING_PROPERTIES m_ExecuteBindingProperties[t_Count];
        UINT m_DescriptorCount;
        OperatorBuffers m_Buffers;

    public:
        Operators() = default;
        void Compile(com_ptr<IDMLDevice> const& Device, DML_EXECUTION_FLAGS Flags = DML_EXECUTION_FLAG_NONE)
        {
            WINRT_ASSERT(Device);
            // Compile the operator into an object that can be dispatched to the GPU. In this step, DirectML performs operator
            // fusion and just-in-time (JIT) compilation of shader bytecode, then compiles it into a Direct3D 12 pipeline state object (PSO).
            // The resulting compiled operator is a baked, optimized form of an operator suitable for execution on the GPU.
            IDMLCompiledOperator* CompiledOperators[t_Count];
            for(size_t Index = 0; Index < t_Count; Index++)
            {
                check_hresult(Device->CompileOperator(m_Operators[Index].get(), Flags, IID_PPV_ARGS(m_CompiledOperators[Index].put())));
                CompiledOperators[Index] = m_CompiledOperators[Index].get();
            }
            WINRT_ASSERT(!m_OperatorInitializer);
            check_hresult(Device->CreateOperatorInitializer(static_cast<UINT>(std::size(CompiledOperators)), CompiledOperators, IID_PPV_ARGS(m_OperatorInitializer.put())));
            // Query the operator for the required size (in descriptors) of its binding table.
            // You need to initialize an operator exactly once before it can be executed, and the two stages require different numbers of descriptors for binding. 
            // For simplicity, we create a single descriptor heap that's large enough to satisfy them both.
            m_InitializeBindingProperties = m_OperatorInitializer->GetBindingProperties();
            m_ExecuteBindingProperties[0] = m_CompiledOperators[0]->GetBindingProperties();
            m_DescriptorCount = std::max(m_InitializeBindingProperties.RequiredDescriptorCount, m_ExecuteBindingProperties[0].RequiredDescriptorCount);
        }
        void CreateBuffers(D3D::Context const& Context)
        {
            m_Buffers.Create(Context, m_InitializeBindingProperties, m_ExecuteBindingProperties[0]);
        }
    };

    class CommandRecorder
    {
    public:
        com_ptr<IDMLCommandRecorder> m_Value;

    public:
        CommandRecorder(com_ptr<IDMLDevice> const& Device)
        {
            WINRT_ASSERT(Device);
            check_hresult(Device->CreateCommandRecorder(IID_PPV_ARGS(m_Value.put())));
        }
        void RecordDispatch(BindingTable const& BindingTable, D3D::Context& Context) const
        {
            WINRT_ASSERT(BindingTable.m_Desc.Dispatchable && Context.m_CommandList);
            WINRT_ASSERT(m_Value);
            m_Value->RecordDispatch(Context.m_CommandList.get(), BindingTable.m_Desc.Dispatchable, BindingTable.m_Value.get());
        }
    };
}

int wmain(int argc, char** argv)
{
    D3D::Context D3dContext;
    D3dContext.Create();

    DML_CREATE_DEVICE_FLAGS DmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_NONE;
    #if defined(_DEBUG)
        DmlCreateDeviceFlags |= DML_CREATE_DEVICE_FLAG_DEBUG;
    #endif
    com_ptr<IDMLDevice> DmlDevice;
    check_hresult(DMLCreateDevice(D3dContext.m_Device.get(), DmlCreateDeviceFlags, IID_PPV_ARGS(DmlDevice.put())));

    constexpr UINT g_TensorSize[4] { 1, 2, 3, 4 };
    constexpr UINT g_TensorElementCount = g_TensorSize[0] * g_TensorSize[1] * g_TensorSize[2] * g_TensorSize[3];

    DML_BUFFER_TENSOR_DESC DmlBufferTensorDesc { };
    {
        DmlBufferTensorDesc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
        DmlBufferTensorDesc.Flags = DML_TENSOR_FLAG_NONE;
        DmlBufferTensorDesc.DimensionCount = static_cast<UINT>(std::size(g_TensorSize));
        DmlBufferTensorDesc.Sizes = g_TensorSize;
        DmlBufferTensorDesc.Strides = nullptr;
        DmlBufferTensorDesc.TotalTensorSizeInBytes = DML::CalculateBufferTensorSize(DmlBufferTensorDesc);
    }
    const UINT64 TensorBufferSize { DmlBufferTensorDesc.TotalTensorSizeInBytes };

    DML::Operators<1> Operators;
    {
        DML_TENSOR_DESC TensorDesc { DML_TENSOR_TYPE_BUFFER, &DmlBufferTensorDesc };
        DML_ELEMENT_WISE_ADD_OPERATOR_DESC AddOperatorDesc { &TensorDesc, &TensorDesc, &TensorDesc};
        DML_OPERATOR_DESC OperatorDesc { DML_OPERATOR_ELEMENT_WISE_ADD, &AddOperatorDesc };
        check_hresult(DmlDevice->CreateOperator(&OperatorDesc, IID_PPV_ARGS(Operators.m_Operators[0].put())));
    }
    Operators.Compile(DmlDevice);
    Operators.CreateBuffers(D3dContext);

    D3D::DescriptorHeap DescriptorHeap(D3dContext, Operators.m_DescriptorCount);
    DML::BindingTable BindingTable(DmlDevice, DescriptorHeap, Operators.m_DescriptorCount, Operators.m_OperatorInitializer.get());

    Operators.m_Buffers.BindToInitialize(BindingTable);

    DML::CommandRecorder CommandRecorder { DmlDevice };
    CommandRecorder.RecordDispatch(BindingTable, D3dContext);

    // Close the Direct3D 12 command list, and submit it for execution as you would any other command list. You could in principle record the execution into the same command list as the initialization, 
    // but you need only to Initialize once, and typically you want to Execute an operator more frequently than that.
//    D3dContext.ExecuteCommandListAndWait();

    com_ptr<ID3D12Resource> InputBuffer { D3dContext.CreateResource(D3D12_HEAP_TYPE_DEFAULT, CD3DX12_RESOURCE_DESC::Buffer(TensorBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_COPY_DEST) };
    com_ptr<ID3D12Resource> OutputBuffer { D3dContext.CreateResource(D3D12_HEAP_TYPE_DEFAULT, CD3DX12_RESOURCE_DESC::Buffer(TensorBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_UNORDERED_ACCESS) };
    #pragma region Upload
    com_ptr<ID3D12Resource> UploadBuffer { D3dContext.CreateBufferResource(D3D12_HEAP_TYPE_UPLOAD, TensorBufferSize, D3D12_RESOURCE_STATE_GENERIC_READ) };
    {
        std::wcout << std::fixed; std::wcout.precision(3);
        std::array<FLOAT, g_TensorElementCount> InputArray;
        {
            std::wcout << L"input tensor: ";
            for(auto& element: InputArray)
            {
                element = 1.50f;
                std::wcout << element << L' ';
            };
            std::wcout << std::endl;
        }
        D3D12_SUBRESOURCE_DATA SubresourceData { InputArray.data(), static_cast<LONG_PTR>(TensorBufferSize), static_cast<LONG_PTR>(TensorBufferSize) };
        UpdateSubresources(D3dContext.m_CommandList.get(), InputBuffer.get(), UploadBuffer.get(), 0, 0, 1, &SubresourceData);
        D3dContext.ResourceBarrier(CD3DX12_RESOURCE_BARRIER::Transition(InputBuffer.get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
    }
    #pragma endregion

    BindingTable.Reset(Operators.m_CompiledOperators[0].get());
    DML_BINDING_DESC Inputs[] { DML::BufferBindingDesc(InputBuffer, TensorBufferSize), DML::BufferBindingDesc(InputBuffer, TensorBufferSize) };
    BindingTable.BindInputs(Inputs);
    DML_BINDING_DESC Ouputs[] { DML::BufferBindingDesc(OutputBuffer, TensorBufferSize) };
    BindingTable.BindOutputs(Ouputs);
    Operators.m_Buffers.BindToExecute(BindingTable);
    CommandRecorder.RecordDispatch(BindingTable, D3dContext);
//    D3dContext.ExecuteCommandListAndWait();

    #pragma region Readback
    {
        com_ptr<ID3D12Resource> ReadbackBuffer { D3dContext.CreateBufferResource(D3D12_HEAP_TYPE_READBACK, TensorBufferSize, D3D12_RESOURCE_STATE_COPY_DEST) };
        D3dContext.ResourceBarrier(CD3DX12_RESOURCE_BARRIER::Transition(OutputBuffer.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE));
        D3dContext.m_CommandList->CopyResource(ReadbackBuffer.get(), OutputBuffer.get());
        D3dContext.ExecuteCommandListAndWait();
        {
            D3D12_RANGE Range { 0, TensorBufferSize };
            FLOAT* Data;
            check_hresult(ReadbackBuffer->Map(0, &Range, reinterpret_cast<void**>(&Data)));
            std::wcout << L"output tensor: ";
            for(size_t Index = 0; Index < g_TensorElementCount; ++Index, ++Data)
                std::wcout << *Data << L' ';
            std::wcout << std::endl;
            D3D12_RANGE WriteRange { 0, 0 };
            ReadbackBuffer->Unmap(0, &WriteRange);
        }
    }
    #pragma endregion 
}
