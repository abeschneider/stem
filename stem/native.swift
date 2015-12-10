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
        stride[0] = 1
        for i in 1..<shape.dims {
            stride[i] = shape[i]*mult
            mult *= shape[i]
        }
        
        return stride
    }    
}
