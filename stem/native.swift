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
//    var broadcast:[Bool]
    
    public var size:Int { return array.memory.count }
    
    public required init(size:Int) {
        array = SharedArray<ElementType>(count: size, repeatedValue: ElementType(0))
    }
    
    public required init(array:[T]) {
        self.array = SharedArray<T>(array)
    }
    
    public required init(storage:NativeStorage) {
        array = SharedArray<ElementType>(storage.array.memory)
    }

    public required init(storage:NativeStorage, copy:Bool) {
        if copy {
            array = SharedArray<ElementType>(count: storage.size, repeatedValue: ElementType(0))
            array.copy(storage.array)
        } else {
            array = SharedArray<ElementType>(storage.array.memory)
        }
    }
    
    public func transform<NewStorageType:Storage>(fn:(el:ElementType) -> NewStorageType.ElementType) -> NewStorageType {
        let value:[NewStorageType.ElementType] = array.memory.map(fn)
        return NewStorageType(array:value)
    }
    
    public subscript(index:Int) -> T {
        get { return array[index] }
        set { array[index] = newValue }
    }
    
    public func calculateStride(shape:Extent) -> [Int] {
        var stride = Array<Int>(count:shape.count, repeatedValue: 0)
        
        var mult = 1
        stride[0] = 1
        for i in 1..<shape.count {
            stride[i] = shape[i]*mult
            mult *= shape[i]
        }
        
        return stride
    }
}

//public struct NativeIndex<T:NumericType> {
//    public var offset:Array<Int>
//    public var shape:Extent
//    public var overflow:Bool
//    
//    public init(_ s:Extent) {
//        shape = s
//        overflow = false
//
//        // TODO: is there a way to allocate an array without initializing it?
//        offset = Array<Int>(count:shape.count, repeatedValue: 0)
//        
//        var mult = 1
//        offset[0] = 1
//        for i in 1..<shape.count {
//            offset[i] = shape[i]*mult
//            mult *= shape[i]
//        }
//    }
//}
//
//public func +=<T:NumericType>(var index:NativeIndex<T>, size:Int) {
//    let last = index.shape.count-1
//    if index.offset[last] >= index.shape[last] {
//        var d:Int = index.shape.count - size
//        
//        // loop until we no longer overflow
//        while d >= 0 && index.offset[d] >= index.shape[d] {
//            // at the end, so return no results left
//            if d == 0 {
//                index.overflow = true
//            } else {
//                // reset current index
//                index.offset[d] = 0
//                
//                // increment next offset
//                index.offset[d-1] += 1
//                
//                // go to next dimension
//                d -= 1
//            }
//        }
//    }
//}
