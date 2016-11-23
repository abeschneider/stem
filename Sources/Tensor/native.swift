//
//  native.swift
//  stem
//
//  Created by Abe Schneider on 11/14/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation

open class NativeStorage<T:NumericType>: Storage {
    public typealias ElementType = T
    
    open var array:SharedArray<T>
    open var offset:Int
    
    open var size:Int { return array.memory.count }
    open var order:DimensionOrder { return .columnMajor }
    
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
    
    open func transform<NewStorageType:Storage>(_ fn:(_ el:ElementType) -> NewStorageType.ElementType) -> NewStorageType {
        let value:[NewStorageType.ElementType] = array.memory.map(fn)
        return NewStorageType(array:value)
    }
    
    open subscript(index:Int) -> T {
        get { return array[index+offset] }
        set { array[index+offset] = newValue }
    }
    
    open func calculateOrder(_ dims:Int) -> [Int] {
        return (0..<dims).map { dims-$0-1 }
    }
    
    open func calculateOrder(_ values:[Int]) -> [Int] {
        return values.reversed()
    }
}
