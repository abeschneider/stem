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
    
    public var array:SharedArray<T>
    public var offset:Int
    
    public var size:Int { return array.memory.count }
    public var order:DimensionOrder { return .ColumnMajor }
    
    public required init(size:Int, value:ElementType=0) {
        array = SharedArray<ElementType>(count: size, repeatedValue: value)
        offset = 0
    }
    
    public required init(array:[T]) {
        self.array = SharedArray<T>(array)
        offset = 0
    }
    
    public required init(storage:NativeStorage, offset:Int=0) {
        array = storage.array
        self.offset = offset
    }

    public required init(storage:NativeStorage, copy:Bool) {
        offset = 0
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
        get { return array[index+offset] }
        set { array[index+offset] = newValue }
    }
    
    public func calculateOrder(dims:Int) -> [Int] {
        return (0..<dims).map { dims-$0-1 }
    }
    
    public func calculateOrder(values:[Int]) -> [Int] {
        return values.reverse()
    }
}
