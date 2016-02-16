Tensor library
==============

# Overview
To create a `Tensor`:

```swift
let ntensor = Tensor<NativeStorage<Double>>(shape: Extent(5, 5))
let ctensor = Tensor<CBlasStorage<Double>>(shape: Extent(5, 5))
```

where `NativeStorage` lays out the memory for the `Tensor` in a row-major format, and `CBlasStorage` las out the memory for the `Tensor` in column-major format. New storage types can be added by implementing the protocol `Storage`. The `Storage` protocol determines how memory is allocated, how to index into the memory, and is used for dispatching to the correct methods.

An example of using storage types for dispatch:

```swift
let ntensor2 = ntensor + ntensor
let ctensor2 = ctensor + ctensor
```

In the first case, the native (unaccelerated) version of `Tensor` addition is used, whereas in the second the CBlas accelerated version will be used.

The `Tensor` class has several specialized sub-classes: `Matrix`, `Vector`, `ColumnVector`, and `RowVector`. The specializations allow custom constructors and methods to allow the classes to be more easily used.
