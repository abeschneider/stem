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
    
    public required init(size:Int) {
        array = SharedArray<ElementType>(count: size, repeatedValue: ElementType(0))
    }
    
    public required init(array:[T]) {
        self.array = SharedArray<T>(array)
    }
    
    public subscript(index:Int) -> T {
        get { return array[index] }
        set { array[index] = newValue }
    }
    
    public func calculateStride(shape:Extent) -> [Int] {
        var stride = Array<Int>(count:shape.dims, repeatedValue: 0)
        
        var mult = 1
        for i in 0..<shape.dims-1 {
            stride[i] = shape[i]*mult
            mult *= shape[i]
        }
        stride[shape.dims-1] = 1

        return stride
    }
}

// Accelerated versions of the tensor operations
// TODO: make in-place operation that saves results in `left` (iadd)
func add(
    left left:Vector<CBlasStorage<Double>>,
    right:Vector<CBlasStorage<Double>>,
    result:Vector<CBlasStorage<Double>>, alpha:Double=1.0)
{
    assert(left.shape == right.shape)

    let v1Ptr = UnsafePointer<Double>(left.storage.array.memory) + left.calculateOffset()
    let v2Ptr = UnsafePointer<Double>(right.storage.array.memory) + right.calculateOffset()
    let resultPtr = UnsafeMutablePointer<Double>(result.storage.array.memory) + result.calculateOffset()
    
    // need to figure out best way to determine which dimension this will go on .. also,
    // if it's not on the first dimension, then a stride needs to be added on
    let numElements = Int32(left.shape.elements)
    
    cblas_dcopy(numElements, v2Ptr, Int32(left.stride[0]), resultPtr, Int32(result.stride[0]))
    cblas_daxpy(numElements, alpha, v1Ptr, Int32(left.stride[0]), resultPtr, Int32(result.stride[0]))
}

func +(
    left:Vector<CBlasStorage<Double>>,
    right:Vector<CBlasStorage<Double>>) -> Vector<CBlasStorage<Double>>
{
    let result = Vector<CBlasStorage<Double>>(rows: left.shape[0])
    add(left: left, right: right, result: result)
    
    return result
}

/*func iadd(left left:CDVector, right:CDVector, alpha:Double=1.0) {
    assert(left.shape == right.shape)
    
    let v1Ptr = UnsafeMutablePointer<Double>(left.view.storage.array.memory) + left.view.calculateOffset()
    let v2Ptr = UnsafePointer<Double>(right.view.storage.array.memory) + right.view.calculateOffset()
    
    // need to figure out best way to determine which dimension this will go on .. also,
    // if it's not on the first dimension, then a stride needs to be added on
    let numElements = Int32(left.shape.elements)
    
    var leftStride:Int32
    if left.shape[0] > 1 && left.shape.dims() > 1 {
        // row major, so need to skip by number of columns
        leftStride = Int32(left.view.shape[1])
    } else {
        // column major, so use a stride of 1
        leftStride = 1
    }
    
    cblas_daxpy(numElements, alpha, v2Ptr, leftStride, v1Ptr, leftStride)
}

func +=(left:CDVector, right:CDVector) {
    iadd(left: left, right: right, alpha: 1.0)
}*/

func dot(
    left left:Matrix<CBlasStorage<Double>>,
    right:Vector<CBlasStorage<Double>>,
    result:Vector<CBlasStorage<Double>>,
    alpha:Double=1.0,
    beta:Double=1.0)
{
    assert(left.shape[1] == right.shape[0])
    
    cblas_dgemv(CblasColMajor,
                left.transposed ? CblasTrans : CblasNoTrans,
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
                Int32(left.view.stride[0]),
                UnsafePointer<Double>(right.storage.array.memory),
                Int32(right.view.stride[0]),
                UnsafeMutablePointer<Double>(result.storage.array.memory),
                Int32(result.view.stride[0]))
}
