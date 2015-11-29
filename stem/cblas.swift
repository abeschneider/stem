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
    
    public let order:MatrixOrder = .ColumnMajor
    var array:SharedArray<T>
    public var shape:Extent
    public var stride:[Int]
    
    public required init(shape:Extent) {
        self.shape = shape
        array = SharedArray<ElementType>(count: shape.elements, repeatedValue: ElementType(0))
        stride = Array<Int>(count:self.shape.dims(), repeatedValue: 0)
        
        var mult = 1
        for i in 0..<self.shape.dims()-1 {
            stride[i] = self.shape[i]*mult
            mult *= self.shape[i]
        }
        stride[self.shape.dims()-1] = 1
    }
    
    public required init(array:[T], shape:Extent) {
        self.shape = shape
        self.array = SharedArray<T>(array)
        stride = Array<Int>(count:self.shape.dims(), repeatedValue: 0)
        
        var mult = 1
        for i in 0..<self.shape.dims()-1 {
            stride[i] = self.shape[i]*mult
            mult *= self.shape[i]
        }
        stride[self.shape.dims()-1] = 1
    }
    
    public subscript(index:Int) -> T {
        get { return array[index] }
        set { array[index] = newValue }
    }
}

public typealias CDTensor = Tensor<CBlasStorage<Double>>
public typealias CFTensor = Tensor<CBlasStorage<Float>>
public typealias CDMatrix = Matrix<CBlasStorage<Double>>
public typealias CFMatrix = Matrix<CBlasStorage<Float>>
public typealias CDVector = Vector<CBlasStorage<Double>>
public typealias CFVector = Vector<CBlasStorage<Float>>

// Accelerated versions of the tensor operations
// TODO: make in-place operation that saves results in `left` (iadd)
func add(left left:CDVector, right:CDVector, alpha:Double, result:CDVector) {
    assert(left.shape == right.shape)

    let v1Ptr = UnsafePointer<Double>(left.view.storage.array.memory) + left.view.calculateOffset()
    let v2Ptr = UnsafePointer<Double>(right.view.storage.array.memory) + right.view.calculateOffset()
    let resultPtr = UnsafeMutablePointer<Double>(result.view.storage.array.memory) + result.view.calculateOffset()
    
    // need to figure out best way to determine which dimension this will go on .. also,
    // if it's not on the first dimension, then a stride needs to be added on
    let numElements = Int32(left.shape.elements)
    
    var leftStride:Int32
    if left.shape[0] > 1 && left.shape.dims() > 1 {
        // row major, so need to skip by number of columns
        leftStride = Int32(left.view.storage.shape[1])
    } else {
        // column major, so use a stride of 1
        leftStride = 1
    }
    
    var resultStride:Int32
    if result.shape[0] > 1 && result.shape.dims() > 1 {
        resultStride = Int32(result.view.storage.shape[1])
    } else {
        resultStride = 1
    }
    
    cblas_dcopy(numElements, v2Ptr, leftStride, resultPtr, resultStride)
    cblas_daxpy(numElements, alpha, v1Ptr, leftStride, resultPtr, resultStride)
}

func +(left:CDVector, right:CDVector) -> CDVector {
    let result = CDVector(rows: left.shape[0])
    add(left: left, right: right, alpha: 1.0, result: result)
    
    return result
}

func iadd(left left:CDVector, right:CDVector, alpha:Double) {
    assert(left.shape == right.shape)
    
    let v1Ptr = UnsafeMutablePointer<Double>(left.view.storage.array.memory) + left.view.calculateOffset()
    let v2Ptr = UnsafePointer<Double>(right.view.storage.array.memory) + right.view.calculateOffset()
    
    // need to figure out best way to determine which dimension this will go on .. also,
    // if it's not on the first dimension, then a stride needs to be added on
    let numElements = Int32(left.shape.elements)
    
    var leftStride:Int32
    if left.shape[0] > 1 && left.shape.dims() > 1 {
        // row major, so need to skip by number of columns
        leftStride = Int32(left.view.storage.shape[1])
    } else {
        // column major, so use a stride of 1
        leftStride = 1
    }
    
    cblas_daxpy(numElements, alpha, v2Ptr, leftStride, v1Ptr, leftStride)
}

func +=(left:CDVector, right:CDVector) {
    iadd(left: left, right: right, alpha: 1.0)
}

func dot(left left:CDMatrix, right:CDVector, result:CDVector, alpha:Double=1.0, beta:Double=1.0) {
    assert(left.shape[1] == right.shape[0])
    
    let leftStride = Int32(left.view.storage.shape[0])
    
    var rightStride:Int32
    if right.shape[0] > 1 && right.shape.dims() > 1 {
        // row major, so need to skip by number of columns
        rightStride = Int32(left.view.storage.shape[1])
    } else {
        // column major, so use a stride of 1
        rightStride = 1
    }
    
    var resultStride:Int32
    if result.shape[0] > 1 && result.shape.dims() > 1 {
        resultStride = Int32(result.view.storage.shape[1])
    } else {
        resultStride = 1
    }
    
    cblas_dgemv(CblasColMajor,
                left.transposed ? CblasTrans : CblasNoTrans,
                Int32(left.shape[0]),
                Int32(left.shape[1]),
                alpha,
                UnsafePointer<Double>(left.view.storage.array.memory) + left.view.calculateOffset(),
                leftStride,
                UnsafePointer<Double>(right.view.storage.array.memory) + right.view.calculateOffset(),
                rightStride,
                beta,
                UnsafeMutablePointer<Double>(result.view.storage.array.memory) + result.view.calculateOffset(),
                resultStride)
}


