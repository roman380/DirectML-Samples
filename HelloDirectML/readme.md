# HelloDirectML sample

A minimal but complete DirectML sample that demonstrates how to initialize D3D12 and DirectML, create and compile an operator, execute the operator on the GPU, and retrieve the results.

This sample executes the DirectML "add" and "multiply" operators over a 1x2x3x4 tensor. The addition and multiplication are element wise and hence 1.5 is expected to become (1.5 + 1.5) * (1.5 + 1.5).The expected output is:

```
input tensor: 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5
output tensor: 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0
```

The operators use intermediate tensor with a UAV resource barrier to synchronize dispatch between the two operators.

When built using the "Debug" configuration, the sample enables the D3D12 and DirectML debug layers, which require the Graphics Tools feature-on-demand (FOD) to be installed. For more information, see [Using the DirectML debug layer](https://docs.microsoft.com/windows/desktop/direct3d12/dml-debug-layer).
