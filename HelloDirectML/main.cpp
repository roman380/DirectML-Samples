// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"

using winrt::com_ptr;
using winrt::check_hresult;
using winrt::check_bool;
using winrt::handle;

void CreateDevice(com_ptr<ID3D12Device>& Device, com_ptr<ID3D12CommandQueue>& CommandQueue, com_ptr<ID3D12CommandAllocator>& CommandAllocator, com_ptr<ID3D12GraphicsCommandList>& CommandList)
{
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
        const HRESULT CreateDeviceResult = D3D12CreateDevice(DxgiAdapter.get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(Device.put()));
        if(CreateDeviceResult == DXGI_ERROR_UNSUPPORTED) 
            continue;
        check_hresult(CreateDeviceResult);
        break;
    }
    D3D12_COMMAND_QUEUE_DESC CommandQueueDesc { D3D12_COMMAND_LIST_TYPE_DIRECT };
    CommandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    check_hresult(Device->CreateCommandQueue(&CommandQueueDesc, IID_PPV_ARGS(CommandQueue.put())));
    check_hresult(Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(CommandAllocator.put())));
    check_hresult(Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, CommandAllocator.get(), nullptr, IID_PPV_ARGS(CommandList.put())));
}

void CloseExecuteResetWait(com_ptr<ID3D12Device> const& Device, com_ptr<ID3D12CommandQueue> const& CommandQueue, com_ptr<ID3D12CommandAllocator> const& CommandAllocator, com_ptr<ID3D12GraphicsCommandList> const& CommandList)
{
    check_hresult(CommandList->Close());
    ID3D12CommandList* CommandLists[] { CommandList.get() };
    CommandQueue->ExecuteCommandLists(static_cast<UINT>(std::size(CommandLists)), CommandLists);
    check_hresult(CommandList->Reset(CommandAllocator.get(), nullptr));
    com_ptr<ID3D12Fence> Fence;
    check_hresult(Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(Fence.put())));
    handle FenceEvent { CreateEvent(nullptr, TRUE, FALSE, nullptr) };
    check_bool(bool { FenceEvent });
    check_hresult(Fence->SetEventOnCompletion(1, FenceEvent.get()));
    check_hresult(CommandQueue->Signal(Fence.get(), 1));
    WaitForSingleObjectEx(FenceEvent.get(), INFINITE, FALSE);
}

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

int wmain(int argc, char** argv)
{
    com_ptr<ID3D12Device> Device;
    com_ptr<ID3D12CommandQueue> CommandQueue;
    com_ptr<ID3D12CommandAllocator> CommandAllocator;
    com_ptr<ID3D12GraphicsCommandList> CommandList;
    CreateDevice(Device, CommandQueue, CommandAllocator, CommandList);

    DML_CREATE_DEVICE_FLAGS DmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_NONE;
    #if defined(_DEBUG)
        DmlCreateDeviceFlags |= DML_CREATE_DEVICE_FLAG_DEBUG;
    #endif
    com_ptr<IDMLDevice> DmlDevice;
    check_hresult(DMLCreateDevice(Device.get(), DmlCreateDeviceFlags, IID_PPV_ARGS(DmlDevice.put())));

    constexpr UINT g_TensorSize[4] { 1, 2, 3, 4 };
    constexpr UINT g_TensorElementCount = g_TensorSize[0] * g_TensorSize[1] * g_TensorSize[2] * g_TensorSize[3];

    DML_BUFFER_TENSOR_DESC DmlBufferTensorDesc { };
    DmlBufferTensorDesc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
    DmlBufferTensorDesc.Flags = DML_TENSOR_FLAG_NONE;
    DmlBufferTensorDesc.DimensionCount = static_cast<UINT>(std::size(g_TensorSize));
    DmlBufferTensorDesc.Sizes = g_TensorSize;
    DmlBufferTensorDesc.Strides = nullptr;
    DmlBufferTensorDesc.TotalTensorSizeInBytes = CalculateBufferTensorSize(DmlBufferTensorDesc);

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
    check_hresult(Device->CreateDescriptorHeap(&DescriptorHeapDesc, IID_PPV_ARGS(DescriptorHeap.put())));
    ID3D12DescriptorHeap* DescriptorHeaps[] { DescriptorHeap.get() };
    CommandList->SetDescriptorHeaps(static_cast<UINT>(std::size(DescriptorHeaps)), DescriptorHeaps);
    DML_BINDING_TABLE_DESC DmlBindingTableDesc { DmlOperatorInitializer.get(), DescriptorHeap->GetCPUDescriptorHandleForHeapStart(), DescriptorHeap->GetGPUDescriptorHandleForHeapStart(), DescriptorCount };
    com_ptr<IDMLBindingTable> DmlBindingTable;
    check_hresult(DmlDevice->CreateBindingTable(&DmlBindingTableDesc, IID_PPV_ARGS(DmlBindingTable.put())));

    // The temporary resource is scratch memory (used internally by DirectML), whose contents you don't need to define.
    // The persistent resource is long-lived, and you need to initialize it using the IDMLOperatorInitializer.

    com_ptr<ID3D12Resource> TemporaryBuffer;
    const UINT64 TemporaryResourceSize = std::max(InitializeBindingProperties.TemporaryResourceSize, ExecuteBindingProperties.TemporaryResourceSize);
    if(TemporaryResourceSize)
    {
        check_hresult(Device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(TemporaryResourceSize), D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(TemporaryBuffer.put())));
        DML_BUFFER_BINDING BufferBinding { TemporaryBuffer.get(), 0, TemporaryResourceSize };
        DML_BINDING_DESC BindingDesc { DML_BINDING_TYPE_BUFFER, &BufferBinding };
        DmlBindingTable->BindTemporaryResource(&BindingDesc);
    }
    com_ptr<ID3D12Resource> PersistentBuffer;
    const UINT64 PersistentResourceSize = ExecuteBindingProperties.PersistentResourceSize;
    if(PersistentResourceSize)
    {
        check_hresult(Device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(PersistentResourceSize), D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(PersistentBuffer.put())));
        DML_BUFFER_BINDING BufferBinding { PersistentBuffer.get(), 0, PersistentResourceSize };
        DML_BINDING_DESC BindingDesc { DML_BINDING_TYPE_BUFFER, &BufferBinding };
        DmlBindingTable->BindOutputs(1, &BindingDesc); // Persistent is Initializer Output
    }

    // The command recorder is a stateless object that records Dispatches into an existing Direct3D 12 command list.
    com_ptr<IDMLCommandRecorder> CommandRecorder;
    check_hresult(DmlDevice->CreateCommandRecorder(IID_PPV_ARGS(CommandRecorder.put())));
    CommandRecorder->RecordDispatch(CommandList.get(), DmlOperatorInitializer.get(), DmlBindingTable.get());

    // Close the Direct3D 12 command list, and submit it for execution as you would any other command list. You could in principle record the execution into the same command list as the initialization, 
    // but you need only to Initialize once, and typically you want to Execute an operator more frequently than that.
    CloseExecuteResetWait(Device, CommandQueue, CommandAllocator, CommandList);

    CommandList->SetDescriptorHeaps(static_cast<UINT>(std::size(DescriptorHeaps)), DescriptorHeaps);

    // Reset the binding table to bind for the operator we want to execute (it was previously used to bind for the initializer).
    DmlBindingTableDesc.Dispatchable = DmlCompiledOperator.get();
    check_hresult(DmlBindingTable->Reset(&DmlBindingTableDesc));

    if(TemporaryResourceSize)
    {
        DML_BUFFER_BINDING BufferBinding { TemporaryBuffer.get(), 0, TemporaryResourceSize };
        DML_BINDING_DESC BindingDesc { DML_BINDING_TYPE_BUFFER, &BufferBinding };
        DmlBindingTable->BindTemporaryResource(&BindingDesc);
    }
    if(PersistentResourceSize)
    {
        DML_BUFFER_BINDING BufferBinding { PersistentBuffer.get(), 0, PersistentResourceSize };
        DML_BINDING_DESC BindingDesc { DML_BINDING_TYPE_BUFFER, &BufferBinding };
        DmlBindingTable->BindPersistentResource(&BindingDesc);
    }

    const UINT64 TensorBufferSize { DmlBufferTensorDesc.TotalTensorSizeInBytes };

    #pragma region Initialize, Upload, Input Buffer
    com_ptr<ID3D12Resource> InputBuffer;
    com_ptr<ID3D12Resource> UploadBuffer;
    {
        check_hresult(Device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(TensorBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(InputBuffer.put())));
        check_hresult(Device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(TensorBufferSize), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(UploadBuffer.put())));
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
            D3D12_SUBRESOURCE_DATA SubresourceData { InputArray.data(), static_cast<LONG_PTR>(TensorBufferSize), SubresourceData.RowPitch };
            UpdateSubresources(CommandList.get(), InputBuffer.get(), UploadBuffer.get(), 0, 0, 1, &SubresourceData);
            CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(InputBuffer.get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        }
    }
    {
        DML_BUFFER_BINDING BufferBinding { InputBuffer.get(), 0, TensorBufferSize };
        DML_BINDING_DESC BindingDesc { DML_BINDING_TYPE_BUFFER, &BufferBinding };
        DmlBindingTable->BindInputs(1, &BindingDesc);
    }
    #pragma endregion
    com_ptr<ID3D12Resource> OutputBuffer;
    {
        check_hresult(Device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(TensorBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(OutputBuffer.put())));
        DML_BUFFER_BINDING BufferBinding { OutputBuffer.get(), 0, TensorBufferSize };
        DML_BINDING_DESC BindingDesc { DML_BINDING_TYPE_BUFFER, &BufferBinding };
        DmlBindingTable->BindOutputs(1, &BindingDesc);
    }
    CommandRecorder->RecordDispatch(CommandList.get(), DmlCompiledOperator.get(), DmlBindingTable.get());
    CloseExecuteResetWait(Device, CommandQueue, CommandAllocator, CommandList);
    #pragma region Readback Buffer
    // The output buffer now contains the result of the identity operator, so read it back if you want the CPU to access it.
    {
        com_ptr<ID3D12Resource> ReadbackBuffer;
        check_hresult(Device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK), D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(TensorBufferSize), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(ReadbackBuffer.put())));
        CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(OutputBuffer.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE));
        CommandList->CopyResource(ReadbackBuffer.get(), OutputBuffer.get());
        CloseExecuteResetWait(Device, CommandQueue, CommandAllocator, CommandList);
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
