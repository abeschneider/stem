//
//  cblas.swift
//  stem
//
//  Created by Abe Schneider on 11/13/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation

#if os(OSX)
    import Accelerate
#else
    // ?
#endif

open class CBlasStorage<T:NumericType>: Storage {
    public typealias ElementType = T
    
    var array:SharedArray<T>
    
    open var size:Int { return array.memory.count }
    open var order:DimensionOrder { return .rowMajor }
    
    public required init(size:Int, value:ElementType=0) {
        array = SharedArray<ElementType>(count: size, repeatedValue: value)
    }
    
    public required init(array:[T]) {
        self.array = SharedArray<T>(array, offset: 0)
    }
    
    public required init(storage:CBlasStorage, offset:Int=0) {
        array = SharedArray<ElementType>(storage.array.memory, offset: offset)
    }
    
    public required init(storage:CBlasStorage, copy:Bool) {
        if copy {
            array = SharedArray<ElementType>(count: storage.size, repeatedValue: ElementType(0))
            array.copy(storage.array)
        } else {
            array = SharedArray<ElementType>(storage.array.memory, offset: 0)
        }
    }

    open subscript(index:Int) -> T {
        get { return array[index] }
        set { array[index] = newValue }
    }
    
    open func calculateOrder(_ dims:Int) -> [Int] {
        return (0..<dims).map { $0 }
    }
    
    open func calculateOrder(_ values:[Int]) -> [Int] {
        return values
    }
}

func add(
    left:Tensor<CBlasStorage<Double>>,
    right:Tensor<CBlasStorage<Double>>,
    result:Tensor<CBlasStorage<Double>>)
{
    // use accelerated methods if they're both vectors
    if isVector(left) && isVector(right) {
        let v1Ptr = UnsafePointer<Double>(left.storage.array.memory) + left.calculateOffset()
        let v2Ptr = UnsafePointer<Double>(right.storage.array.memory) + right.calculateOffset()
        let resultPtr = UnsafeMutablePointer<Double>(mutating: result.storage.array.memory) + result.calculateOffset()
        let numElements = Int32(left.shape.elements)

        // result += left
        cblas_daxpy(numElements, 1.0, v2Ptr, Int32(right.stride[0]), resultPtr, Int32(result.stride[0]))
        
        // result += right
        cblas_daxpy(numElements, 1.0, v1Ptr, Int32(left.stride[0]), resultPtr, Int32(result.stride[0]))
    } else {
        // TODO: should be able to accelerate matrix/vector broadcasts (for other ops as well)
        let (l, r) = broadcast(left, right)
        elementwiseBinaryOp(l, r, result: result, op: { $0 + $1 })
    }
}

func +(left:Tensor<CBlasStorage<Double>>,
       right:Tensor<CBlasStorage<Double>>) -> Tensor<CBlasStorage<Double>>
{
    let result = Tensor<CBlasStorage<Double>>(Extent(left.shape[0]))
    add(left: left, right: right, result: result)
    
    return result
}

func iadd(
    left:Tensor<CBlasStorage<Double>>,
    right:Tensor<CBlasStorage<Double>>)
{
    if left.shape.span == 1 && right.shape.span == 1 {
        let numElements = Int32(right.shape.elements)
        let v2Ptr = UnsafePointer<Double>(right.storage.array.memory) + right.calculateOffset()
        let leftPtr = UnsafeMutablePointer<Double>(mutating: left.storage.array.memory) + left.calculateOffset()
        
        cblas_daxpy(numElements, 1.0, v2Ptr, Int32(right.stride[0]), leftPtr, Int32(left.stride[0]))
    } else {
        let (l, r) = broadcast(left, right)
        elementwiseBinaryOp(l, r, result: left, op: { $0 + $1 })
    }
}

func +=(left:Tensor<CBlasStorage<Double>>,
       right:Tensor<CBlasStorage<Double>>) -> Tensor<CBlasStorage<Double>>
{
    iadd(left: left, right: right)
    return left
}

//func dot<T:NumericType>(
//    _ left:Tensor<CBlasStorage<T>>,
//    _ right:Tensor<CBlasStorage<T>>,
//    result:Tensor<CBlasStorage<T>>,
//    alpha:Double=1.0,
//    beta:Double=1.0)
//{
//    // specializations provided below
//    assertionFailure("No implemention for given type")
//}

func dot(
    _ left:Tensor<CBlasStorage<Double>>,
    _ right:Tensor<CBlasStorage<Double>>,
    result:Tensor<CBlasStorage<Double>>,
    alpha:Double=1.0,
    beta:Double=1.0)
{
    // inner dimensions must match
    precondition(left.shape[1] == right.shape[0])
    
    if isVector(right) {
        precondition(isVector(result))

        // TODO: check if this is correct (or should it be .RowVector)?
        let leftTransposed = (left.type == .columnVector)
        
        cblas_dgemv(CblasColMajor,
                    leftTransposed ? CblasTrans : CblasNoTrans,
                    Int32(left.shape[0]),
                    Int32(left.shape[1]),
                    alpha,
                    UnsafePointer<Double>(left.storage.array.memory) + left.calculateOffset(),
                    Int32(left.shape[0]),
                    UnsafePointer<Double>(right.storage.array.memory) + right.calculateOffset(),
                    Int32(right.stride[0]),
                    beta,
                    UnsafeMutablePointer<Double>(mutating: result.storage.array.memory) + result.calculateOffset(),
                    Int32(result.stride[0]))
    } else if isMatrix(right) {
        precondition(isMatrix(result))
        
        cblas_dgemm(CblasColMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    Int32(left.shape[0]),   // M: A.rows
                    Int32(right.shape[1]),  // N: B.cols
                    Int32(right.shape[0]),  // K: B.rows
                    alpha,
                    UnsafePointer<Double>(left.storage.array.memory) + left.calculateOffset(),  // A
                    Int32(left.shape[0]),   // A.rows
                    UnsafePointer<Double>(right.storage.array.memory) + right.calculateOffset(),// B
                    Int32(right.shape[0]),  // B.rows
                    beta,
                    UnsafeMutablePointer<Double>(mutating: result.storage.array.memory)
                        + result.calculateOffset(),                                             // C
                    Int32(left.shape[0]))   // C.rows
    }
}

public func outer(
    left:Tensor<CBlasStorage<Double>>,      // Vector
    right:Tensor<CBlasStorage<Double>>,     // Vector
    result:Tensor<CBlasStorage<Double>>)
{
    precondition(isVector(left))
    precondition(isVector(right))
    precondition(left.shape[0] == result.shape[0])
    precondition(right.shape[0] == result.shape[1])
    
    cblas_dger( CblasColMajor,
                Int32(result.shape[0]),
                Int32(result.shape[1]),
                1.0,
                UnsafePointer<Double>(left.storage.array.memory),
                Int32(left.stride[0]),
                UnsafePointer<Double>(right.storage.array.memory),
                Int32(right.stride[0]),
                UnsafeMutablePointer<Double>(mutating: result.storage.array.memory),
                Int32(result.stride[0]))
}

public func conv2d
    (_ input:Tensor<CBlasStorage<Double>>,
     kernels:Tensor<CBlasStorage<Double>>,
     stride:[Int]=[1, 1],
     padding:[Int]=[0, 0],
     paddingValue:Double=0,
     flip:Bool=true,
     result:Tensor<CBlasStorage<Double>>)
{
    typealias T = Tensor<CBlasStorage<Double>>
    
    let unfoldedKernel = unroll(kernels: kernels, flip: flip)
    let unfoldedInput = unroll(tensor: input, kernelShape: Extent(kernels.shape[2], kernels.shape[3]), padding: padding)
    
    let output = T(Extent(unfoldedInput.shape[0], unfoldedKernel.shape[1]))
    dot(unfoldedInput, unfoldedKernel, result: output)
    
    for i in 0..<kernels.shape[0] {
        result[i, all, all] = output[all, i].reshape(Extent(result.shape[1], result.shape[2]))
    }
}

public func conv2d(
    _ input:Tensor<CBlasStorage<Double>>,
    kernels:Tensor<CBlasStorage<Double>>,
    stride:[Int]=[1, 1],
    padding:[Int]=[0, 0],
    paddingValue:Double=0,
    flip:Bool=true) -> Tensor<CBlasStorage<Double>>
{
    let outputShape = calculateConv2DSize(input: input, kernels: kernels, stride: stride, padding: padding)
    let out = Tensor<CBlasStorage<Double>>(outputShape)
    
    conv2d(input, kernels: kernels, stride: stride, padding: padding, flip: flip, result: out)
    return out
}

//func addvv(
//    left left:Tensor<CBlasStorage<Double>>,
//    right:Tensor<CBlasStorage<Double>>,
//    result:Tensor<CBlasStorage<Double>>,
//    alpha:Double=1.0)
//{
////    assert(left.shape[0] == right.shape[0])
////    if left.shape[0] != right.shape[0] {
////        throw TensorError.SizeMismatch(lhs: left.shape, rhs: right.shape)
////    }
//    precondition(left.shape[0] == right.shape[0], "Number of rows must match")
//
//    let v1Ptr = UnsafePointer<Double>(left.storage.array.memory) + left.calculateOffset()
//    let v2Ptr = UnsafePointer<Double>(right.storage.array.memory) + right.calculateOffset()
//    let resultPtr = UnsafeMutablePointer<Double>(result.storage.array.memory) + result.calculateOffset()
//    
//    let numElements = Int32(left.shape.elements)
//    
//    // result += left
//    cblas_daxpy(numElements, 1.0, v2Ptr, Int32(right.stride[0]), resultPtr, Int32(result.stride[0]))
//    
//    // result += right
//    cblas_daxpy(numElements, 1.0, v1Ptr, Int32(left.stride[0]), resultPtr, Int32(result.stride[0]))
//}
//
//func addmv(
//    left left:Tensor<CBlasStorage<Double>>,
//    right:Tensor<CBlasStorage<Double>>,
//    result:Tensor<CBlasStorage<Double>>,
//    alpha:Double=1.0)
//{
//    // NxM + N
////    assert(left.shape[0] == right.shape[0])
//    precondition(left.shape[0] == right.shape[0], "Number of rows must match")
////    if left.shape[0] != right.shape[0] {
////        throw TensorError.SizeMismatch(lhs: left.shape, rhs: right.shape)
////    }
//    
//    let numElements = Int32(right.shape.elements)
//    
//    let v2Ptr = UnsafePointer<Double>(right.storage.array.memory) + right.calculateOffset()
//    
//    let cols = left.shape[1]
//    for i in 0..<cols {
//        let v1Ptr = UnsafePointer<Double>(left.storage.array.memory) + left.calculateOffset([0, i])
//        let resultPtr = UnsafeMutablePointer<Double>(result.storage.array.memory) + result.calculateOffset([0, i])
//
//        // result += left
//        cblas_daxpy(numElements, alpha, v2Ptr, Int32(right.stride[0]), resultPtr, Int32(result.stride[1]))
//        
//        // result += right
//        cblas_daxpy(numElements, alpha, v1Ptr, Int32(left.stride[1]), resultPtr, Int32(result.stride[1]))
//    }
//}
//
//// TODO: combine with above
//func add(
//    left left:Matrix<CBlasStorage<Double>>,
//    right:RowVector<CBlasStorage<Double>>,
//    result:Matrix<CBlasStorage<Double>>,
//    alpha:Double=1.0)
//{
//    // NxM + 1xM
////    if left.shape[1] != right.shape[1] {
////        throw TensorError.SizeMismatch(lhs: left.shape, rhs: right.shape)
////    }
//    precondition(left.shape[1] == right.shape[1], "Number of columns must match")
//    
//    let numElements = Int32(right.shape.elements)
//    
//    let v2Ptr = UnsafePointer<Double>(right.storage.array.memory) + right.calculateOffset()
//    
//    let rows = left.shape[0]
//    for i in 0..<rows {
//        let v1Ptr = UnsafePointer<Double>(left.storage.array.memory) + left.calculateOffset([i, 0])
//        let resultPtr = UnsafeMutablePointer<Double>(result.storage.array.memory) + result.calculateOffset([i, 0])
//        
//        // result += left
//        cblas_daxpy(numElements, alpha, v1Ptr, Int32(left.stride[0]), resultPtr, Int32(result.stride[0]))
//        
//        // result += right
//        cblas_daxpy(numElements, alpha, v2Ptr, Int32(right.stride[0]), resultPtr, Int32(result.stride[0]))
//    }
//}

//func iadd(
//    left left:Matrix<CBlasStorage<Double>>,
//    right:RowVector<CBlasStorage<Double>>,
//    alpha:Double=1.0)
//{
//    // NxM + 1xM
////    if left.shape[1] != right.shape[1] {
////        throw TensorError.SizeMismatch(lhs: left.shape, rhs: right.shape)
////    }
//    precondition(left.shape[1] == right.shape[1], "Number of columns must match")
//    
//    let numElements = Int32(right.shape.elements)
//    
//    let v2Ptr = UnsafePointer<Double>(right.storage.array.memory) + right.calculateOffset()
//    
//    let rows = left.shape[0]
//    for i in 0..<rows {
//        let leftPtr = UnsafeMutablePointer<Double>(left.storage.array.memory) + left.calculateOffset([i, 0])
//        
//        // result += right
//        cblas_daxpy(numElements, alpha, v2Ptr, Int32(right.stride[1]), leftPtr, Int32(left.stride[0]))
//    }
//}
//
//func iadd(
//    left left:Matrix<CBlasStorage<Double>>,
//    right:ColumnVector<CBlasStorage<Double>>,
//    alpha:Double=1.0)
//{
//    // NxM + Nx1
////    if left.shape[0] != right.shape[0] {
////        throw TensorError.SizeMismatch(lhs: left.shape, rhs: right.shape)
////    }
//    precondition(left.shape[0] == right.shape[0], "Number of rows must match")
//    
//    let numElements = Int32(right.shape.elements)
//    
//    let v2Ptr = UnsafePointer<Double>(right.storage.array.memory) + right.calculateOffset()
//    
//    let cols = left.shape[1]
//    for i in 0..<cols {
//        let leftPtr = UnsafeMutablePointer<Double>(left.storage.array.memory) + left.calculateOffset([0, i])
//        
//        // result += right
//        cblas_daxpy(numElements, alpha, v2Ptr, Int32(right.stride[0]), leftPtr, Int32(left.stride[1]))
//    }
//}

