//
//  sharedarray.swift
//  stem
//
//  Created by Abe Schneider on 11/10/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation

// should add @_specialize for all NativeStorage types
// (other storage won't use SharedArray)
open class SharedArray<T>: Swift.Collection where T:Equatable {
    open var memory:Array<T>
    open var offset:Int
    
    open var startIndex:Int { return offset }
    open var endIndex:Int { return memory.count + offset }
    
    init(_ values:T...) {
        memory = Array<T>(values)
        offset = 0
    }
    
    init(_ values:[T], offset:Int=0) {
        memory = Array<T>(values)
        self.offset = offset
    }
    
    init(count:Int, repeatedValue:T) {
        memory = Array<T>(repeating: repeatedValue, count: count)
        offset = 0
    }
    
    open subscript(index:Int) -> T {
        get { return memory[index+offset] }
        set { memory[index+offset] = newValue }
    }
    
    func copy(_ array:SharedArray<T>) {
        memory = array.memory
        offset = array.offset
    }
    
    /// Returns the position immediately after the given index.
    ///
    /// - Parameter i: A valid index of the collection. `i` must be less than
    ///   `endIndex`.
    /// - Returns: The index value immediately after `i`.
    public func index(after i: Int) -> Int {
        return i+1
    }
}

extension SharedArray: CustomStringConvertible {
    public var description:String {
        let values = memory.map { String(describing: $0) }.joined(separator: ", ")
        return "[\(values)]"
    }
}
