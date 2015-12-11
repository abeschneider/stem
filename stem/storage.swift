//
//  storage.swift
//  stem
//
//  Created by Abe Schneider on 11/11/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation

public protocol NumericType: AbsoluteValuable, Comparable {
    func +(lhs: Self, rhs: Self) -> Self
    func -(lhs: Self, rhs: Self) -> Self
    func *(lhs: Self, rhs: Self) -> Self
    func /(lhs: Self, rhs: Self) -> Self
    func %(lhs: Self, rhs: Self) -> Self

    init(_ v: Int)
}

// This isn't implemented by default??
extension Int: AbsoluteValuable {
    public static func abs(x:Int) -> Int {
        return abs(x)
    }
}

extension Int: NumericType {}
extension Double: NumericType {}
extension Float: NumericType {}

public protocol Storage {
    typealias ElementType:NumericType
    
    var size:Int { get }
    
    init(size:Int)
    init(array:[ElementType])
    init(storage:Self)
//    init<OtherStorageType:Storage>(storage:OtherStorageType)
    
    subscript(index:Int) -> ElementType {get set}
    
    // returns the stride for each dimension to traverse memory
    func calculateStride(shape:Extent) -> [Int]
}
