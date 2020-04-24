// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"

using winrt::com_ptr;
using winrt::check_hresult;
using winrt::check_bool;

namespace D3D
{
    class Device
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
        com_ptr<IDMLBindingTable> m_Value;

    public:
        BindingTable(com_ptr<IDMLDevice> const& Device, DML_BINDING_TABLE_DESC const& Desc)
        {
            WINRT_ASSERT(Device);
            check_hresult(Device->CreateBindingTable(&Desc, IID_PPV_ARGS(m_Value.put())));
        }
        void RecordDispatch(com_ptr<IDMLCommandRecorder>& CommandRecorder, D3D::Device& Device, IDMLDispatchable* Dispatchable) const
        {
            WINRT_ASSERT(CommandRecorder && Device.m_CommandList && Dispatchable);
            WINRT_ASSERT(m_Value);
            CommandRecorder->RecordDispatch(Device.m_CommandList.get(), Dispatchable, m_Value.get());
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
        OperatorBuffers(D3D::Device const & Device, const DML_BINDING_PROPERTIES& InitializeProperties, const DML_BINDING_PROPERTIES& ExecuteProperties)
        {
            m_TemporaryResourceSize = std::max(InitializeProperties.TemporaryResourceSize, ExecuteProperties.TemporaryResourceSize);
            if(m_TemporaryResourceSize)
                m_TemporaryBuffer = Device.CreateBufferResource(D3D12_HEAP_TYPE_DEFAULT, m_TemporaryResourceSize);
            m_PersistentResourceSize = ExecuteProperties.PersistentResourceSize;
            if(m_PersistentResourceSize)
                m_PersistentBuffer = Device.CreateBufferResource(D3D12_HEAP_TYPE_DEFAULT, m_PersistentResourceSize);
        }
        void InitializeBind(BindingTable const& BindingTable) const
        {
            WINRT_ASSERT(BindingTable.m_Value);
            if(m_TemporaryResourceSize)
                BindingTable->BindTemporaryResource(&DML::BufferBindingDesc(m_TemporaryBuffer, m_TemporaryResourceSize));
            if(m_PersistentResourceSize)
                BindingTable->BindOutputs(1, &DML::BufferBindingDesc(m_PersistentBuffer, m_PersistentResourceSize)); // Persistent is Initializer Output
        }
        void ExecuteBind(BindingTable const& BindingTable) const
        {
            WINRT_ASSERT(BindingTable.m_Value);
            if(m_TemporaryResourceSize)
                BindingTable->BindTemporaryResource(&DML::BufferBindingDesc(m_TemporaryBuffer, m_TemporaryResourceSize));
            if(m_PersistentResourceSize)
                BindingTable->BindPersistentResource(&DML::BufferBindingDesc(m_PersistentBuffer, m_PersistentResourceSize));
        }
    };
}

int wmain(int argc, char** argv)
{
    D3D::Device Device;
    Device.Create();

    DML_CREATE_DEVICE_FLAGS DmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_NONE;
    #if defined(_DEBUG)
        DmlCreateDeviceFlags |= DML_CREATE_DEVICE_FLAG_DEBUG;
    #endif
    com_ptr<IDMLDevice> DmlDevice;
    check_hresult(DMLCreateDevice(Device.m_Device.get(), DmlCreateDeviceFlags, IID_PPV_ARGS(DmlDevice.put())));

    constexpr UINT g_TensorSize[4] { 1, 2, 3, 4 };
    constexpr UINT g_TensorElementCount = g_TensorSize[0] * g_TensorSize[1] * g_TensorSize[2] * g_TensorSize[3];

    DML_BUFFER_TENSOR_DESC DmlBufferTensorDesc { };
    DmlBufferTensorDesc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
    DmlBufferTensorDesc.Flags = DML_TENSOR_FLAG_NONE;
    DmlBufferTensorDesc.DimensionCount = static_cast<UINT>(std::size(g_TensorSize));
    DmlBufferTensorDesc.Sizes = g_TensorSize;
    DmlBufferTensorDesc.Strides = nullptr;
    DmlBufferTensorDesc.TotalTensorSizeInBytes = DML::CalculateBufferTensorSize(DmlBufferTensorDesc);

    com_ptr<IDMLOperator> DmlOperator;
    {
        // Create DirectML operator(s). Operators represent abstract functions such as "multiply", "reduce", "convolution", or even compound operations such as recurrent neural nets.
        // This example creates an instance of the Identity operator, which applies the function f(x) = x for all elements in a tensor.
        DML_TENSOR_DESC TensorDesc { DML_TENSOR_TYPE_BUFFER, &DmlBufferTensorDesc };
        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC IdentityOperatorDesc { };
        IdentityOperatorDesc.InputTensor = &TensorDesc;
        IdentityOperatorDesc.OutputTensor = &TensorDesc;
        DML_OPERATOR_DESC OperatorDesc { DML_OPERATOR_ELEMENT_WISE_IDENTITY, &IdentityOperatorDesc };
        check_hresult(DmlDevice->CreateOperator(&OperatorDesc, IID_PPV_ARGS(DmlOperator.put())));
    }

    // Compile the operator into an object that can be dispatched to the GPU. In this step, DirectML performs operator
    // fusion and just-in-time (JIT) compilation of shader bytecode, then compiles it into a Direct3D 12 pipeline state object (PSO).
    // The resulting compiled operator is a baked, optimized form of an operator suitable for execution on the GPU.

    com_ptr<IDMLCompiledOperator> DmlCompiledOperator;
    check_hresult(DmlDevice->CompileOperator(DmlOperator.get(), DML_EXECUTION_FLAG_NONE, IID_PPV_ARGS(DmlCompiledOperator.put())));
    com_ptr<IDMLOperatorInitializer> DmlOperatorInitializer;
    IDMLCompiledOperator* DmlCompiledOperators[] { DmlCompiledOperator.get() };
    check_hresult(DmlDevice->CreateOperatorInitializer(static_cast<UINT>(std::size(DmlCompiledOperators)), DmlCompiledOperators, IID_PPV_ARGS(DmlOperatorInitializer.put())));

    // Query the operator for the required size (in descriptors) of its binding table.
    // You need to initialize an operator exactly once before it can be executed, and the two stages require different numbers of descriptors for binding. 
    // For simplicity, we create a single descriptor heap that's large enough to satisfy them both.
    const DML_BINDING_PROPERTIES InitializeBindingProperties = DmlOperatorInitializer->GetBindingProperties();
    const DML_BINDING_PROPERTIES ExecuteBindingProperties = DmlCompiledOperator->GetBindingProperties();
    const UINT DescriptorCount = std::max(InitializeBindingProperties.RequiredDescriptorCount, ExecuteBindingProperties.RequiredDescriptorCount);

    const D3D12_DESCRIPTOR_HEAP_DESC DescriptorHeapDesc { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, DescriptorCount, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE };
    com_ptr<ID3D12DescriptorHeap> DescriptorHeap;
    check_hresult(Device.m_Device->CreateDescriptorHeap(&DescriptorHeapDesc, IID_PPV_ARGS(DescriptorHeap.put())));
    ID3D12DescriptorHeap* DescriptorHeaps[] { DescriptorHeap.get() };
    Device.m_CommandList->SetDescriptorHeaps(static_cast<UINT>(std::size(DescriptorHeaps)), DescriptorHeaps);
    DML_BINDING_TABLE_DESC DmlBindingTableDesc { DmlOperatorInitializer.get(), DescriptorHeap->GetCPUDescriptorHandleForHeapStart(), DescriptorHeap->GetGPUDescriptorHandleForHeapStart(), DescriptorCount };
    DML::BindingTable BindingTable(DmlDevice, DmlBindingTableDesc);

    // The temporary resource is scratch memory (used internally by DirectML), whose contents you don't need to define.
    // The persistent resource is long-lived, and you need to initialize it using the IDMLOperatorInitializer.
    DML::OperatorBuffers OperatorBuffers(Device, InitializeBindingProperties, ExecuteBindingProperties);
    OperatorBuffers.InitializeBind(BindingTable);

    // The command recorder is a stateless object that records Dispatches into an existing Direct3D 12 command list.
    com_ptr<IDMLCommandRecorder> CommandRecorder;
    check_hresult(DmlDevice->CreateCommandRecorder(IID_PPV_ARGS(CommandRecorder.put())));
    BindingTable.RecordDispatch(CommandRecorder, Device, DmlOperatorInitializer.get());

    // Close the Direct3D 12 command list, and submit it for execution as you would any other command list. You could in principle record the execution into the same command list as the initialization, 
    // but you need only to Initialize once, and typically you want to Execute an operator more frequently than that.
//    Device.ExecuteCommandListAndWait();

    Device.m_CommandList->SetDescriptorHeaps(static_cast<UINT>(std::size(DescriptorHeaps)), DescriptorHeaps);

    // Reset the binding table to bind for the operator we want to execute (it was previously used to bind for the initializer).
    DmlBindingTableDesc.Dispatchable = DmlCompiledOperator.get();
    check_hresult(BindingTable.m_Value->Reset(&DmlBindingTableDesc));

    OperatorBuffers.ExecuteBind(BindingTable);

    const UINT64 TensorBufferSize { DmlBufferTensorDesc.TotalTensorSizeInBytes };

    com_ptr<ID3D12Resource> InputBuffer { Device.CreateResource(D3D12_HEAP_TYPE_DEFAULT, CD3DX12_RESOURCE_DESC::Buffer(TensorBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_COPY_DEST) };
    com_ptr<ID3D12Resource> OutputBuffer { Device.CreateResource(D3D12_HEAP_TYPE_DEFAULT, CD3DX12_RESOURCE_DESC::Buffer(TensorBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_UNORDERED_ACCESS) };

    #pragma region Upload
    com_ptr<ID3D12Resource> UploadBuffer { Device.CreateBufferResource(D3D12_HEAP_TYPE_UPLOAD, TensorBufferSize, D3D12_RESOURCE_STATE_GENERIC_READ) };
    {
        std::wcout << std::fixed; std::wcout.precision(2);
        std::array<FLOAT, g_TensorElementCount> InputArray;
        {
            std::wcout << L"input tensor: ";
            for(auto& element: InputArray)
            {
                element = 1.618f;
                std::wcout << element << L' ';
            };
            std::wcout << std::endl;
        }
        D3D12_SUBRESOURCE_DATA SubresourceData { InputArray.data(), static_cast<LONG_PTR>(TensorBufferSize), SubresourceData.RowPitch };
        UpdateSubresources(Device.m_CommandList.get(), InputBuffer.get(), UploadBuffer.get(), 0, 0, 1, &SubresourceData);
        Device.m_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(InputBuffer.get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
    }
    #pragma endregion
    BindingTable->BindInputs(1, &DML::BufferBindingDesc(InputBuffer, TensorBufferSize));
    BindingTable->BindOutputs(1, &DML::BufferBindingDesc(OutputBuffer, TensorBufferSize));
    BindingTable.RecordDispatch(CommandRecorder, Device, DmlCompiledOperator.get());
//    Device.ExecuteCommandListAndWait();
    #pragma region Readback
    {
        com_ptr<ID3D12Resource> ReadbackBuffer { Device.CreateBufferResource(D3D12_HEAP_TYPE_READBACK, TensorBufferSize, D3D12_RESOURCE_STATE_COPY_DEST) };
        Device.m_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(OutputBuffer.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE));
        Device.m_CommandList->CopyResource(ReadbackBuffer.get(), OutputBuffer.get());
        Device.ExecuteCommandListAndWait();
        {
            D3D12_RANGE Range { 0, TensorBufferSize };
            FLOAT* Data;
            check_hresult(ReadbackBuffer->Map(0, &Range, reinterpret_cast<void**>(&Data)));
            std::wcout << L"output tensor: ";
            for(size_t Index { 0 }; Index < g_TensorElementCount; ++Index, ++Data)
                std::wcout << *Data << L' ';
            std::wcout << std::endl;
            D3D12_RANGE WriteRange { 0, 0 };
            ReadbackBuffer->Unmap(0, &WriteRange);
        }
    }
    #pragma endregion 
}
