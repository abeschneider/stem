//
//  view.swift
//  stem
//
//  Created by Abe Schneider on 11/11/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation

public struct StorageViewIndex<StorageType:Storage>: GeneratorType {
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
}

public struct StorageView<StorageType:Storage> {
    public var storage:StorageType
    public var shape:Extent
    public var window:[Range<Int>]
    public var dimIndex:[Int]

    
    public init(storage:StorageType, dimIndex:[Int]?=nil) {
        self.storage = storage
        shape = storage.shape
        window = storage.shape.map { 0..<$0 }
        
        let dims = shape.dims()
        if let idx = dimIndex {
            self.dimIndex = idx
        } else {
            self.dimIndex = (0..<dims).map { dims-$0-1 }
        }
    }
    
    public init(storage:StorageType, window:[Range<Int>], dimIndex:[Int]?=nil) {
        self.storage = storage
        self.shape = Extent(window.map { $0.last! - $0.first! + 1})
        self.window = window

        let dims = shape.dims()
        if let idx = dimIndex {
            self.dimIndex = idx
        } else {
            self.dimIndex = (0..<dims).map { dims-$0-1 }
        }
    }
    
    public func calculateOffset(indices:[Int]) -> Int {
        var offset = 0
        for i in 0..<indices.count {
            offset += (indices[dimIndex[i]]+window[dimIndex[i]].first!)*storage.stride[i]
        }
        
        return offset
    }
    
    public func calculateOffset(indices:Int...) -> Int {
        return calculateOffset(indices)
    }
    
    public subscript(indices:[Int]) -> StorageType.ElementType {
        get { return storage[calculateOffset(indices)] }
        set { storage[calculateOffset(indices)] = newValue }
    }
    
    public subscript(indices:Int...) -> StorageType.ElementType {
        get { return storage[calculateOffset(indices)] }
        set { storage[calculateOffset(indices)] = newValue }
    }
    
    // generates indices of view in storage
    public func storageIndices() -> GeneratorSequence<StorageViewIndex<StorageType>> {
//        StorageViewIndex<StorageType> {
        return GeneratorSequence<StorageViewIndex<StorageType>>(StorageViewIndex<StorageType>(self))
    }
}