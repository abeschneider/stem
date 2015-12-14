//
//  numerictype.swift
//  stem
//
//  Created by Abe Schneider on 12/11/15.
//  Copyright © 2015 none. All rights reserved.
//

import Foundation

public protocol NumericType: AbsoluteValuable, Comparable, FloatingPointType {
    func +(lhs: Self, rhs: Self) -> Self
    func -(lhs: Self, rhs: Self) -> Self
    func *(lhs: Self, rhs: Self) -> Self
    func /(lhs: Self, rhs: Self) -> Self
    func %(lhs: Self, rhs: Self) -> Self
    func ^(lhs: Self, rhs: Self) -> Self
    
    init(_ v:Float)
    init(_ v:Double)
}

public func ^(lhs:Float, rhs:Float) -> Float {
    return pow(lhs, rhs)
}

public func ^(lhs:Double, rhs:Double) -> Double {
    return pow(lhs, rhs)
}


extension Float: NumericType {}
extension Double: NumericType {}

