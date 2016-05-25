//
//  cblas.swift
//  stem
//
//  Created by Abe Schneider on 11/13/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation
import Accelerate

public class CBlasStorage<T:NumericType>: Storage {
    public typealias ElementType = T
    
    var array:SharedArray<T>
    
    public var size:Int { return array.memory.count }
    public var order:DimensionOrder { return .RowMajor }
    
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

    public subscript(index:Int) -> T {
        get { return array[index] }
        set { array[index] = newValue }
    }
    
    public func calculateOrder(dims:Int) -> [Int] {
        return (0..<dims).map { $0 }
    }
    
    public func calculateOrder(values:[Int]) -> [Int] {
        return values
    }
}

func add(
    left left:Tensor<CBlasStorage<Double>>,
    right:Tensor<CBlasStorage<Double>>,
    result:Tensor<CBlasStorage<Double>>)
{
    // use accelerated methods if they're both vectors
    if left.shape.span == 1 && right.shape.span == 1 {
        let v1Ptr = UnsafePointer<Double>(left.storage.array.memory) + left.calculateOffset()
        let v2Ptr = UnsafePointer<Double>(right.storage.array.memory) + right.calculateOffset()
        let resultPtr = UnsafeMutablePointer<Double>(result.storage.array.memory) + result.calculateOffset()
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
    left left:Tensor<CBlasStorage<Double>>,
    right:Tensor<CBlasStorage<Double>>)
{
    if left.shape.span == 1 && right.shape.span == 1 {
        let numElements = Int32(right.shape.elements)
        let v2Ptr = UnsafePointer<Double>(right.storage.array.memory) + right.calculateOffset()
        let leftPtr = UnsafeMutablePointer<Double>(left.storage.array.memory) + left.calculateOffset()
        
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

func dot(
    left left:Tensor<CBlasStorage<Double>>, // Matrix
    right:Tensor<CBlasStorage<Double>>,     // Vector
    result:Tensor<CBlasStorage<Double>>,    // Vector
    alpha:Double=1.0,
    beta:Double=1.0)
{
    precondition(isMatrix(left.type))
    precondition(isVector(right.type))
    precondition(isVector(result.type))
    precondition(left.shape[1] == right.shape[0])
    
    // TODO: check if this is correct (or should it be .RowVector)?
    let leftTransposed = (left.type == .ColumnVector)
    
    cblas_dgemv(CblasColMajor,
                leftTransposed ? CblasTrans : CblasNoTrans,
                Int32(left.shape[0]),
                Int32(left.shape[1]),
                alpha,
                UnsafePointer<Double>(left.storage.array.memory) + left.calculateOffset(),
                Int32(left.stride[0]),
                UnsafePointer<Double>(right.storage.array.memory) + right.calculateOffset(),
                Int32(right.stride[0]),
                beta,
                UnsafeMutablePointer<Double>(result.storage.array.memory) + result.calculateOffset(),
                Int32(result.stride[0]))
}

public func outer(
    left left:Tensor<CBlasStorage<Double>>, // Vector
    right:Tensor<CBlasStorage<Double>>,     // Vector
    result:Tensor<CBlasStorage<Double>>)
{
    precondition(isVector(left.type))
    precondition(isVector(right.type))
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
                UnsafeMutablePointer<Double>(result.storage.array.memory),
                Int32(result.stride[0]))
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

