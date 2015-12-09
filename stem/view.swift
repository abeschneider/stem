//
//  view.swift
//  stem
//
//  Created by Abe Schneider on 11/11/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation

/*public struct StorageViewIndex<StorageType:Storage>: GeneratorType {
    typealias ViewType = StorageView<StorageType>
    
    var view:ViewType
    var indices:[Int]
    
    init(_ view:ViewType) {
        self.view = view
        indices = [Int](count: view.shape.dims(), repeatedValue: 0)
    }
    
    public mutating func next() -> Int? {
        let last = view.shape.dims()-1
        if indices[last] >= view.shape[last] {
            var d:Int = view.shape.dims() - 1
            while d >= 0 && indices[d] >= view.shape[d] {
                // at the end, so return no results left
                if d == 0 { return nil }
                
                // reset current index
                indices[d] = 0
                
                // increment next offset
                ++indices[d-1]
                
                // go to next dimension
                --d
            }
        }
        
        // TODO: This is expensive if the next index gives currentOffset+1.
        // Need to figure out a way of shortcuting this calculation when possible.
        let value = view.calculateOffset(indices)
        ++indices[last]
        return value
    }
}*/

public struct StorageView<StorageType:Storage> {
    public var shape:Extent
    
    // offset within storage
    public var offset:[Int]
    
    public init(shape:Extent, offset:[Int]) {
        self.shape = shape
        self.offset = offset
    }
}


// TODO: remove
//    public init(storage:StorageType, dimIndex:[Int]?=nil) {
//        self.storage = storage
//        
//        // inherit the shape from storage
////        shape = storage.shape
//        
////        window = storage.shape.map { 0..<$0 }
//        let dims = shape.dims()
//        offset = Array<Int>(count: dims, repeatedValue: 0)
//        
//        
//        if let idx = dimIndex {
//            self.dimIndex = idx
//        } else {
//            self.dimIndex = (0..<dims).map { dims-$0-1 }
//        }
//        
//        stride = storage.calculateStride(shape)
//    }
    
    /*public init(storage:StorageType, shape:Extent, offset:[Int]?=nil, dimIndex:[Int]?=nil) {
        self.storage = storage
        
        // shape is defined by the parameters (and not inherited from storage)
        self.shape = shape
        
        if let o = offset {
            self.offset = o
        } else {
            self.offset = Array<Int>(count: shape.dims(), repeatedValue: 0)
        }
        
        if let d = dimIndex {
            self.dimIndex = d
        } else {
            self.dimIndex = (0..<shape.dims()).map { shape.dims()-$0-1 }
        }
        
//        stride = storage.calculateStride(storage.shape)
        stride = storage.calculateStride(shape)
    }

    
    public init(storage:StorageType, window:[Range<Int>], dimIndex:[Int]?=nil) {
        self.storage = storage

        // shape is defined by the window (and not inherited from storage)
        shape = Extent(window.map { $0.last! - $0.first! + 1})
        offset = window.map { $0.first! }
        print("offset = \(offset)")
        
        let dims = shape.dims()
        if let d = dimIndex {
            self.dimIndex = d
        } else {
            self.dimIndex = (0..<dims).map { dims-$0-1 }
        }
        
        stride = storage.calculateStride(shape)
    }*/
    
    /*public func calculateOffset(indices:Int...) -> Int {
        return calculateOffset(indices)
    }
    
    public subscript(indices:[Int]) -> StorageType.ElementType {
        get { return storage[calculateOffset(indices)] }
        set { storage[calculateOffset(indices)] = newValue }
    }
    
    public subscript(indices:Int...) -> StorageType.ElementType {
        get { return storage[calculateOffset(indices)] }
        set { storage[calculateOffset(indices)] = newValue }
    }*/
    
    // generates indices of view in storage
//    public func storageIndices() -> GeneratorSequence<StorageViewIndex<StorageType>> {
//        return GeneratorSequence<StorageViewIndex<StorageType>>(StorageViewIndex<StorageType>(self))
//    }
