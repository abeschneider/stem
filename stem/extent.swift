//
//  extent.swift
//  stem
//
//  Created by Abe Schneider on 11/10/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation

// TODO: look into adding this to Extent to allow
// broadcasting (currently directly supported in ops)
enum ExtentElement {
    case value(Int)
    case broadcast
}

public struct Extent: CollectionType {
    var values:[Int]
    var elements:Int
    var span:Int
    
    public var count:Int { return values.count }
    public var dims:[Int] { return values }
    
    init(_ ex: Int...) {
        values = ex
        elements = values.reduce(1, combine: *)
        span = (values.map { Int($0 > 1) }).reduce(0, combine: +)
    }
    
    init(_ dims:[Int]) {
        values = dims
        elements = values.reduce(1,combine: *)
        span = (values.map { Int($0 > 1) }).reduce(0, combine: +)
    }
    
    init(_ extent:Extent) {
        values = extent.values
        elements = values.reduce(1,combine: *)
        span = (values.map { Int($0 > 1) }).reduce(0, combine: +)
    }
    
    public var startIndex:Int { return 0 }
    public var endIndex:Int { return values.count }
    
    public subscript(index: Int) -> Int {
        get {
            // if we're beyond the bounds of the extent, always
            // return 1
            if index >= values.count {
                return 1
            }
            
            return values[index]
        }
        set {
            values[index] = newValue
            elements = values.reduce(1, combine: *)
        }
    }
    
    public func generate() -> AnyGenerator<Int> {
        var index:Int = 0
        return anyGenerator {
            if index >= self.values.count { return nil }
            return self.values[index++]
        }
    }
    
    func max() -> Int {
        var bestValue = values[0]
        var bestIndex = 0
        
        for i in 1..<values.count {
            if values[i] > bestValue {
                bestIndex = i
                bestValue = values[i]
            }
        }
        
        return bestIndex
    }
}

extension Extent: Equatable {}

public func ==(left:Extent, right:Extent) -> Bool{
    if left.elements != right.elements { return false }
    for i in 0..<left.count {
        if left[i] != right[i] { return false }
    }
    
    return true
}
