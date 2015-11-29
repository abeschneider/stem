//
//  view.swift
//  stem
//
//  Created by Abe Schneider on 11/11/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation

/*
protocol StorageViewIndex: GeneratorType {
    typealias ViewType:StorageView
    
    init(_ view:ViewType)
}

struct StorageViewColumnIndex<V:StorageView>: StorageViewIndex {
    typealias ViewType = V

    var view:V
    var indices:[Int]
    
    init(_ view:V) {
        self.view = view
        indices = [Int](count: view.shape.dims(), repeatedValue: 0)
    }

    mutating func next() -> Int? {
        // check if we are beyond the extent of dimension
        if indices[0] >= view.shape[0] {
            var d:Int = 0
            while d < view.shape.dims() && indices[d] >= view.shape[d] {
                // at the end, so return no results left
                if d == view.shape.dims()-1 { return nil }
                
                // reset current index
                indices[d] = 0
                
                // increment next offset
                ++indices[d+1]
                
                // go to next dimension
                ++d
            }
        }
        
        // TODO: calculate most of this in the above if-statement, and add indices[0]
        let value = view.calculateOffset(indices)
        ++indices[0]
        return value
    }
}

struct StorageViewRowIndex<V:StorageView>: StorageViewIndex {
    typealias ViewType = V
    
    var view:V
    var indices:[Int]
    
    init(_ view:V) {
        self.view = view
        indices = [Int](count: view.shape.dims(), repeatedValue: 0)
    }
    
    mutating func next() -> Int? {
        let last = view.shape.count - 1
        if indices[last] >= view.shape[last] {
            var d:Int = last
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
        
        let value = view.calculateOffset(indices)
        ++indices[last]
        return value
    }
}*/

public struct StorageViewIndex<StorageType:Storage>: GeneratorType {
    typealias ViewType = StorageView<StorageType>
    
    var view:ViewType
    var indices:[Int]
    
    init(_ view:ViewType) {
        self.view = view
        indices = [Int](count: view.shape.dims(), repeatedValue: 0)
    }
    
    public mutating func next() -> Int? {
        let last = view.shape.count - 1
        if indices[last] >= view.shape[last] {
            var d:Int = last
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
        
        let value = view.calculateOffset(indices)
        ++indices[last]
        return value

        
        // check if we are beyond the extent of dimension
//        if indices[view.dimIndex[0]] >= view.shape[view.dimIndex[0]] {
        /*if indices[0] >= view.shape[0] {
            var d:Int = 0
            while d < view.shape.dims() &&
                indices[d] >= view.shape[d]
            {
                // at the end, so return no results left
                if d == view.shape.dims()-1 { return nil }
                
                // reset current index
                indices[d] = 0
                
                // increment next offset
                ++indices[d+1]
                
                // go to next dimension
                ++d
            }
        }
        
        // TODO: calculate most of this in the above if-statement, and add indices[0]
        let value = view.calculateOffset(indices)
        ++indices[0]
        return value*/
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
        
        if let idx = dimIndex {
            self.dimIndex = idx
        } else {
            self.dimIndex = (0..<storage.shape.dims()).map { $0 }
        }
    }
    
    public init(storage:StorageType, window:[Range<Int>], dimIndex:[Int]?=nil) {
        self.storage = storage
        shape = Extent(window.map { $0.last! - $0.first! + 1})
        self.window = window
        
        if let idx = dimIndex {
            self.dimIndex = idx
        } else {
            self.dimIndex = (0..<storage.shape.dims()).map { $0 }
        }
    }
    
    public func calculateOffset(indices:[Int]) -> Int {
        var offset = 0
        for i in 0..<indices.count {
            offset += (indices[dimIndex[i]]+window[dimIndex[i]].first!)*storage.stride[i]
//            print("[dimIndex:\(dimIndex[i]), indices:\(indices[dimIndex[i]]), stride:\(stride[i]), offset:\(offset)]")

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
    public func storageIndices() -> AnyGenerator<Int> {
        var igen = StorageViewIndex(self)
        return anyGenerator { return igen.next() }
    }
}

/*public extension StorageView {
    subscript(indices:Int...) -> StorageType.ElementType {
        get { return self[indices] }
        set { self[indices] = newValue }
    }
    
    func calculateOffset(indices:Int...) -> Int {
        return calculateOffset(indices)
    }
}

public class StorageRowView<StorageType:Storage>: StorageView {
    public typealias TransposeType = StorageColumnView<StorageType>
    
    public var storage:StorageType
    public var window:[Range<Int>]
    public var shape:Extent
    
    public required init(storage:StorageType) {
        self.storage = storage
        self.window = storage.shape.map { Range<Int>(start: 0, end: $0) }
        shape = storage.shape
    }
    
    init(storage:StorageType, window:Range<Int>...) {
        self.storage = storage
        self.window = window
        shape = Extent(window.map { $0.last! - $0.first! + 1})
    }
    
    public required init(storage:StorageType, window:[Range<Int>]) {
        self.storage = storage
        self.window = window
        shape = Extent(window.map { $0.last! - $0.first! + 1})
    }
    
    public func calculateOffset(indices:[Int]) -> Int {
        var offset = indices[0]+window[0].first!
        var mult = storage.shape[0]
        for i in 1..<indices.count {
            offset += (indices[i]+window[i].first!)*mult
            mult *= storage.shape[i]
        }
        return offset
    }
    
    public subscript(indices:[Int]) -> StorageType.ElementType {
        get { return storage[calculateOffset(indices)] }
        set { storage[calculateOffset(indices)] = newValue }
    }
    
    public subscript(indices:Int...) -> StorageType.ElementType {
        get { return storage[calculateOffset(indices)] }
        set { storage[calculateOffset(indices)] = newValue }
    }
    
    public func storageIndices() -> AnyGenerator<Int> {
        var igen = StorageViewRowIndex(self)
        return anyGenerator { return igen.next() }
    }
    
    public func transpose() -> TransposeType {
        return TransposeType(storage: storage, window: window)
    }
}

public class StorageColumnView<StorageType:Storage>: StorageView {
    public typealias TransposeType = StorageRowView<StorageType>
    
    public var storage:StorageType
    public var window:[Range<Int>]
    public var shape:Extent
    
    public required init(storage:StorageType) {
        self.storage = storage
        window = storage.shape.map { Range<Int>(start: 0, end: $0) }
        shape = storage.shape
    }
    
    init(storage:StorageType, view:Range<Int>...) {
        self.storage = storage
        self.window = view
        shape = Extent(view.map { $0.last! - $0.first! + 1})
    }
    
    public required init(storage:StorageType, window:[Range<Int>]) {
        self.storage = storage
        self.window = window
        shape = Extent(window.map { $0.last! - $0.first! + 1})
    }
    
    public func calculateOffset(indices:[Int]) -> Int {
        let last = storage.shape.count - 1
        let index = last < indices.count ? indices[last] : 0
        var offset = index+window[last].first!
        var mult = storage.shape[last]
        for (var i=last-1; i >= 0; i--) {
            let index = i < indices.count ? indices[i] : 0
            offset += (index+window[i].first!)*mult
            mult *= storage.shape[i]
        }
        return offset
    }
    
    public subscript(indices:[Int]) -> StorageType.ElementType {
        get { return storage[calculateOffset(indices)] }
        set { storage[calculateOffset(indices)] = newValue }
    }
    
    public subscript(indices:Int...) -> StorageType.ElementType {
        get { return storage[calculateOffset(indices)] }
        set { storage[calculateOffset(indices)] = newValue }
    }
    
    public func storageIndices() -> AnyGenerator<Int> {
        var igen = StorageViewColumnIndex(self)
        return anyGenerator { return igen.next() }
    }
    
    public func transpose() -> TransposeType {
        return TransposeType(storage: storage, window: window)
    }
}*/

//func transposeView<S:Storage>(view view:StorageRowView<S>) -> StorageColumnView<S> {
//    return StorageColumnView<S>(storage: view.storage)
//}
//
//func transposeView<S:Storage>(view view:StorageColumnView<S>) -> StorageRowView<S> {
//    return StorageRowView<S>(storage: view.storage)
//}

// TODO: StorageFlatView