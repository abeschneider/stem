//
//  view.swift
//  stem
//
//  Created by Abe Schneider on 11/11/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation

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
}

protocol StorageView {
    typealias StorageType:Storage
    
    var shape:Extent { get }
    
    var storage:StorageType { get set }
    var window:[Range<Int>] { get }
    
    init(storage:StorageType)
    init(storage:StorageType, view:[Range<Int>])
    
    func calculateOffset(indices:[Int]) -> Int
    func calculateOffset(indices:Int...) -> Int
    
    subscript(indices:[Int]) -> StorageType.ElementType { get set }
    subscript(indices:Int...) -> StorageType.ElementType { get set }
    
    // generates indices of view in storage
    func storageIndices() -> AnyGenerator<Int>
}

extension StorageView {
    subscript(indices:Int...) -> StorageType.ElementType {
        get { return self[indices] }
        set { self[indices] = newValue }
    }
    
    func calculateOffset(indices:Int...) -> Int {
        return calculateOffset(indices)
    }
}

class StorageRowView<StorageType:Storage>: StorageView {
    var storage:StorageType
    var window:[Range<Int>]
    var shape:Extent
    
    required init(storage:StorageType) {
        self.storage = storage
        self.window = storage.shape.map { Range<Int>(start: 0, end: $0) }
        shape = storage.shape
    }
    
    init(storage:StorageType, view:Range<Int>...) {
        self.storage = storage
        self.window = view
        shape = Extent(view.map { $0.last! - $0.first! + 1})
    }
    
    required init(storage:StorageType, view:[Range<Int>]) {
        self.storage = storage
        self.window = view
        shape = Extent(view.map { $0.last! - $0.first! + 1})
    }
    
    func calculateOffset(indices:[Int]) -> Int {
        var offset = indices[0]+window[0].first!
        var mult = storage.shape[0]
        for i in 1..<indices.count {
            offset += (indices[i]+window[i].first!)*mult
            mult *= storage.shape[i]
        }
        return offset
    }
    
    subscript(indices:[Int]) -> StorageType.ElementType {
        get { return storage[calculateOffset(indices)] }
        set { storage[calculateOffset(indices)] = newValue }
    }
    
    subscript(indices:Int...) -> StorageType.ElementType {
        get { return storage[calculateOffset(indices)] }
        set { storage[calculateOffset(indices)] = newValue }
    }
    
    func storageIndices() -> AnyGenerator<Int> {
        var igen = StorageViewRowIndex(self)
        return anyGenerator { return igen.next() }
    }
}

class StorageColumnView<StorageType:Storage>: StorageView {
    var storage:StorageType
    var window:[Range<Int>]
    var shape:Extent
    
    required init(storage:StorageType) {
        self.storage = storage
        window = storage.shape.map { Range<Int>(start: 0, end: $0) }
        shape = storage.shape
    }
    
    init(storage:StorageType, view:Range<Int>...) {
        self.storage = storage
        self.window = view
        shape = Extent(view.map { $0.last! - $0.first! + 1})
    }
    
    required init(storage:StorageType, view:[Range<Int>]) {
        self.storage = storage
        self.window = view
        shape = Extent(view.map { $0.last! - $0.first! + 1})
    }
    
    func calculateOffset(indices:[Int]) -> Int {
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
    
    subscript(indices:[Int]) -> StorageType.ElementType {
        get { return storage[calculateOffset(indices)] }
        set { storage[calculateOffset(indices)] = newValue }
    }
    
    subscript(indices:Int...) -> StorageType.ElementType {
        get { return storage[calculateOffset(indices)] }
        set { storage[calculateOffset(indices)] = newValue }
    }
    
    func storageIndices() -> AnyGenerator<Int> {
        var igen = StorageViewColumnIndex(self)
        return anyGenerator { return igen.next() }
    }
}

// TODO: StorageFlatView