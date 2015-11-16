//
//  native.swift
//  stem
//
//  Created by Abe Schneider on 11/14/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation

class NativeStorage<T:NumericType>: Storage {
    typealias ElementType = T
    
    let order:MatrixOrder = .RowMajor
    var array:SharedArray<T>
    var shape:Extent
    
    required init(shape:Extent) {
        self.shape = shape
        array = SharedArray<ElementType>(count: shape.elements, repeatedValue: ElementType(0))
    }
    
    required init(array:[T], shape:Extent) {
        self.shape = shape
        self.array = SharedArray<T>(array)
    }
    
    subscript(index:Int) -> T {
        get { return array[index] }
        set { array[index] = newValue }
    }
}

typealias DTensor = Tensor<StorageRowView<NativeStorage<Double>>>
typealias DMatrix = Matrix<StorageRowView<NativeStorage<Double>>>
typealias DVector = Vector<StorageRowView<NativeStorage<Double>>>
typealias FTensor = Tensor<StorageRowView<NativeStorage<Float>>>
typealias FMatrix = Matrix<StorageRowView<NativeStorage<Float>>>
typealias FVector = Vector<StorageRowView<NativeStorage<Float>>>
