//
//  cblas.swift
//  stem
//
//  Created by Abe Schneider on 11/13/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation
import Accelerate

// TODO: need some way to set dimIndex from storage level (maybe some function
// to return the index as a default value?)
public class CBlasStorage<T:NumericType>: Storage {
    public typealias ElementType = T
    
    var array:SharedArray<T>
    
    public var size:Int { return array.memory.count }
    public var order:DimensionOrder { return .RowMajor }
    
    public required init(size:Int) {
        array = SharedArray<ElementType>(count: size, repeatedValue: ElementType(0))
    }
    
    public required init(array:[T]) {
        self.array = SharedArray<T>(array)
    }
    
    public required init(storage:CBlasStorage) {
        array = SharedArray<ElementType>(storage.array.memory)
    }
    
    public required init(storage:CBlasStorage, copy:Bool) {
        if copy {
            array = SharedArray<ElementType>(count: storage.size, repeatedValue: ElementType(0))
            array.copy(storage.array)
        } else {
            array = SharedArray<ElementType>(storage.array.memory)
        }
    }

    public subscript(index:Int) -> T {
        get { return array[index] }
        set { array[index] = newValue }
    }
    
    public func calculateStride(shape:Extent) -> [Int] {
        var stride = Array<Int>(count:shape.count, repeatedValue: 0)
        
        var mult = 1
        let last = shape.count-1
        stride[last] = 1
        
        var j = 0
        for i in last.stride(to: 0, by: -1) {
            stride[i-1] = shape[last-i]*mult
            mult *= shape[last-i]
            j += 1
        }
        
        return stride.reverse()
    }

    public func calculateOrder(dims:Int) -> [Int] {
        return (0..<dims).map { $0 }
    }
}

func add(
    left left:Vector<CBlasStorage<Double>>,
    right:Vector<CBlasStorage<Double>>,
    result:Vector<CBlasStorage<Double>>,
    alpha:Double=1.0)
{
//    assert(left.shape[0] == right.shape[0])
//    if left.shape[0] != right.shape[0] {
//        throw TensorError.SizeMismatch(lhs: left.shape, rhs: right.shape)
//    }
    precondition(left.shape[0] == right.shape[0], "Number of rows must match")

    let v1Ptr = UnsafePointer<Double>(left.storage.array.memory) + left.calculateOffset()
    let v2Ptr = UnsafePointer<Double>(right.storage.array.memory) + right.calculateOffset()
    let resultPtr = UnsafeMutablePointer<Double>(result.storage.array.memory) + result.calculateOffset()
    
    let numElements = Int32(left.shape.elements)
    
    // result += left
    cblas_daxpy(numElements, 1.0, v2Ptr, Int32(right.stride[0]), resultPtr, Int32(result.stride[0]))
    
    // result += right
    cblas_daxpy(numElements, 1.0, v1Ptr, Int32(left.stride[0]), resultPtr, Int32(result.stride[0]))
}

func add(
    left left:Matrix<CBlasStorage<Double>>,
    right:ColumnVector<CBlasStorage<Double>>,
    result:Matrix<CBlasStorage<Double>>,
    alpha:Double=1.0)
{
    // NxM + N
//    assert(left.shape[0] == right.shape[0])
    precondition(left.shape[0] == right.shape[0], "Number of rows must match")
//    if left.shape[0] != right.shape[0] {
//        throw TensorError.SizeMismatch(lhs: left.shape, rhs: right.shape)
//    }
    
    let numElements = Int32(right.shape.elements)
    
    let v2Ptr = UnsafePointer<Double>(right.storage.array.memory) + right.calculateOffset()
    
    let cols = left.shape[1]
    for i in 0..<cols {
        let v1Ptr = UnsafePointer<Double>(left.storage.array.memory) + left.calculateOffset([0, i])
        let resultPtr = UnsafeMutablePointer<Double>(result.storage.array.memory) + result.calculateOffset([0, i])

        // result += left
        cblas_daxpy(numElements, alpha, v2Ptr, Int32(right.stride[0]), resultPtr, Int32(result.stride[1]))
        
        // result += right
        cblas_daxpy(numElements, alpha, v1Ptr, Int32(left.stride[1]), resultPtr, Int32(result.stride[1]))
    }
}

func add(
    left left:Matrix<CBlasStorage<Double>>,
    right:RowVector<CBlasStorage<Double>>,
    result:Matrix<CBlasStorage<Double>>,
    alpha:Double=1.0)
{
    // NxM + 1xM
//    if left.shape[1] != right.shape[1] {
//        throw TensorError.SizeMismatch(lhs: left.shape, rhs: right.shape)
//    }
    precondition(left.shape[1] == right.shape[1], "Number of columns must match")
    
    let numElements = Int32(right.shape.elements)
    
    let v2Ptr = UnsafePointer<Double>(right.storage.array.memory) + right.calculateOffset()
    
    let rows = left.shape[0]
    for i in 0..<rows {
        let v1Ptr = UnsafePointer<Double>(left.storage.array.memory) + left.calculateOffset([i, 0])
        let resultPtr = UnsafeMutablePointer<Double>(result.storage.array.memory) + result.calculateOffset([i, 0])
        
        // result += left
        cblas_daxpy(numElements, alpha, v1Ptr, Int32(left.stride[0]), resultPtr, Int32(result.stride[0]))
        
        // result += right
        cblas_daxpy(numElements, alpha, v2Ptr, Int32(right.stride[0]), resultPtr, Int32(result.stride[0]))
    }
}

func iadd(
    left left:Matrix<CBlasStorage<Double>>,
    right:RowVector<CBlasStorage<Double>>,
    alpha:Double=1.0)
{
    // NxM + 1xM
//    if left.shape[1] != right.shape[1] {
//        throw TensorError.SizeMismatch(lhs: left.shape, rhs: right.shape)
//    }
    precondition(left.shape[1] == right.shape[1], "Number of columns must match")
    
    let numElements = Int32(right.shape.elements)
    
    let v2Ptr = UnsafePointer<Double>(right.storage.array.memory) + right.calculateOffset()
    
    let rows = left.shape[0]
    for i in 0..<rows {
        let leftPtr = UnsafeMutablePointer<Double>(left.storage.array.memory) + left.calculateOffset([i, 0])
        
        // result += right
        cblas_daxpy(numElements, alpha, v2Ptr, Int32(right.stride[1]), leftPtr, Int32(left.stride[0]))
    }
}

func iadd(
    left left:Matrix<CBlasStorage<Double>>,
    right:ColumnVector<CBlasStorage<Double>>,
    alpha:Double=1.0)
{
    // NxM + Nx1
//    if left.shape[0] != right.shape[0] {
//        throw TensorError.SizeMismatch(lhs: left.shape, rhs: right.shape)
//    }
    precondition(left.shape[0] == right.shape[0], "Number of rows must match")
    
    let numElements = Int32(right.shape.elements)
    
    let v2Ptr = UnsafePointer<Double>(right.storage.array.memory) + right.calculateOffset()
    
    let cols = left.shape[1]
    for i in 0..<cols {
        let leftPtr = UnsafeMutablePointer<Double>(left.storage.array.memory) + left.calculateOffset([0, i])
        
        // result += right
        cblas_daxpy(numElements, alpha, v2Ptr, Int32(right.stride[0]), leftPtr, Int32(left.stride[1]))
    }
}

func +(
    left:Vector<CBlasStorage<Double>>,
    right:Vector<CBlasStorage<Double>>) -> Vector<CBlasStorage<Double>>
{
    let result = Vector<CBlasStorage<Double>>(rows: left.shape[0])
    add(left: left, right: right, result: result)
    
    return result
}

func dot(
    left left:Matrix<CBlasStorage<Double>>,
    right:Vector<CBlasStorage<Double>>,
    result:Vector<CBlasStorage<Double>>,
    alpha:Double=1.0,
    beta:Double=1.0,
    leftTransposed:Bool=false)
{
    assert(left.shape[1] == right.shape[0])
    
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

// TODO: completely untested
func dot(
    left left:Matrix<CBlasStorage<Double>>,
    right:RowVector<CBlasStorage<Double>>,
    result:ColumnVector<CBlasStorage<Double>>)
{
    return dot(left: left, right: right, result: result, alpha: 1.0, beta: 1.0, leftTransposed: false)
}

func dot(
    left left:Matrix<CBlasStorage<Double>>,
         right:ColumnVector<CBlasStorage<Double>>,
         result:RowVector<CBlasStorage<Double>>)
{
    return dot(left: left, right: right, result: result, alpha: 1.0, beta: 1.0, leftTransposed: true)
}

public func outer(
    left left:Vector<CBlasStorage<Double>>,
    right:Vector<CBlasStorage<Double>>,
    result:Tensor<CBlasStorage<Double>>)
{
    assert(left.shape[0] == result.shape[0])
    assert(right.shape[0] == result.shape[1])
    
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
