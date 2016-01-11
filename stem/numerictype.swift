//
//  numerictype.swift
//  stem
//
//  Created by Abe Schneider on 12/11/15.
//  Copyright © 2015 none. All rights reserved.
//

import Foundation

public protocol NumericType: AbsoluteValuable, Comparable, FloatingPointType {
    func +(lhs:Self, rhs:Self) -> Self
    func -(lhs:Self, rhs:Self) -> Self
    func *(lhs:Self, rhs:Self) -> Self
    func /(lhs:Self, rhs:Self) -> Self
    func %(lhs:Self, rhs:Self) -> Self
    func ^(lhs:Self, rhs:Self) -> Self
    
    static func exp(value:Self) -> Self
    static func sqrt(value:Self) -> Self
    
    static func pow(value: Self, _ power: Self) -> Self
    
    
    init(_ v:Float)
    init(_ v:Double)
}

public func ^(lhs:Float, rhs:Float) -> Float {
    return pow(lhs, rhs)
}

public func ^(lhs:Double, rhs:Double) -> Double {
    return pow(lhs, rhs)
}


extension Float: NumericType {
    static public func exp(value:Float) -> Float {
        return Foundation.exp(value)
    }
    
    static public func sqrt(value:Float) -> Float {
        return Foundation.sqrtf(value)
    }
    
    static public func pow(value:Float, _ power:Float) -> Float {
        return powf(value, power)
    }
}
extension Double: NumericType {
    static public func exp(value: Double) -> Double {
        return Foundation.exp(value)
    }
    
    static public func sqrt(value:Double) -> Double {
        return Foundation.sqrt(value)
    }
    
    static public func pow(value:Double, _ power:Double) -> Double {
        return Foundation.pow(value, power)
    }
}

