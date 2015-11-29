//
//  native.swift
//  stem
//
//  Created by Abe Schneider on 11/14/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation

public class NativeStorage<T:NumericType>: Storage {
    public typealias ElementType = T
    
    public let order:MatrixOrder = .RowMajor
    var array:SharedArray<T>
    public var shape:Extent
    
    public var stride:[Int]
    
    public required init(shape:Extent) {
        self.shape = shape
        array = SharedArray<ElementType>(count: shape.elements, repeatedValue: ElementType(0))
        stride = Array<Int>(count:self.shape.dims(), repeatedValue: 0)
        
        var mult = 1
        stride[0] = 1
        for i in 1..<self.shape.dims() {
            stride[i] = self.shape[i]*mult
            mult *= self.shape[i]
        }
    }
    
    public required init(array:[T], shape:Extent) {
        self.shape = shape
        self.array = SharedArray<T>(array)
        stride = Array<Int>(count:self.shape.dims(), repeatedValue: 0)
        
        var mult = 1
        stride[0] = 1
        for i in 1..<self.shape.dims() {
            stride[i] = self.shape[i]*mult
            mult *= self.shape[i]
        }
    }
    
    public subscript(index:Int) -> T {
        get { return array[index] }
        set { array[index] = newValue }
    }
}

public typealias DTensor = Tensor<NativeStorage<Double>>
public typealias DMatrix = Matrix<NativeStorage<Double>>
public typealias DVector = Vector<NativeStorage<Double>>
public typealias FTensor = Tensor<NativeStorage<Float>>
public typealias FMatrix = Matrix<NativeStorage<Float>>
public typealias FVector = Vector<NativeStorage<Float>>
