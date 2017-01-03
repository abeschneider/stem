//
//  numerictype.swift
//  stem
//
//  Created by Abe Schneider on 12/11/15.
//  Copyright © 2015 none. All rights reserved.
//

import Foundation

public protocol NumericType: AbsoluteValuable, Comparable {
    static func +(lhs:Self, rhs:Self) -> Self
    static func -(lhs:Self, rhs:Self) -> Self
    static func *(lhs:Self, rhs:Self) -> Self
    static func /(lhs:Self, rhs:Self) -> Self
    static func %(lhs:Self, rhs:Self) -> Self
    static func **(lhs:Self, rhs:Self) -> Self
    
    static func max(_ lhs: Self, _ rhs: Self) -> Self
    static func min(_ lhs: Self, _ rhs: Self) -> Self
    
    static var infinity:Self { get }
    
    static func trunc(_ value: Self) -> Int
    
    init(_ v:Int)
}

public protocol FloatNumericType: NumericType, FloatingPoint {
    static func exp(_ value:Self) -> Self
    static func sqrt(_ value:Self) -> Self
    static func pow(_ value: Self, _ power: Self) -> Self
    static func log(_ value: Self) -> Self
    static func tanh(_ value: Self) -> Self
    
    static func trunc(_ value: Self) -> Int
    
    init(_ v:Float)
    init(_ v:Double)
}

public func **(lhs:Int, rhs:Int) -> Int {
    return Int(pow(Float(lhs), Float(rhs)))
}

public func **(lhs:UInt8, rhs:UInt8) -> UInt8 {
    return UInt8(pow(Float(lhs), Float(rhs)))
}

public func **(lhs:Float, rhs:Float) -> Float {
    return pow(lhs, rhs)
}

public func **(lhs:Double, rhs:Double) -> Double {
    return pow(lhs, rhs)
}

extension Int: AbsoluteValuable {
    public static func abs(_ x:Int) -> Int {
        return Int.abs(x)
    }
}

extension Int: NumericType {
    init<StorageType:Storage>
        (_ tensor:Tensor<StorageType>) where StorageType.ElementType == Int
    {
        precondition(tensor.shape.elements == 1, "Can only convert tensors with a single element to a Int")
        
        self.init(tensor.storage[0])

    }
    
    static public func max(_ lhs:Int, _ rhs:Int) -> Int {
        return lhs > rhs ? lhs : rhs
    }
    
    static public func min(_ lhs:Int, _ rhs:Int) -> Int {
        return lhs < rhs ? lhs : rhs
    }
    
    public static var infinity:Int{ return Int.infinity }
    
    // This is needed to allow any Numeric type to be used as an index. However,
    // this isn't a great way to do this. Need to come up with a better strategy.
    public static func trunc(_ value:Int) -> Int { return value }
}

extension UInt8: AbsoluteValuable {
    public static func abs(_ x:UInt8) -> UInt8 {
        return UInt8.abs(x)
    }
}

extension UInt8: NumericType {
    init<StorageType:Storage>
        (_ tensor:Tensor<StorageType>) where StorageType.ElementType == UInt8
    {
        precondition(tensor.shape.elements == 1, "Can only convert tensors with a single element to a Int")
        
        self.init(tensor.storage[0])
        
    }
    
    static public func max(_ lhs:UInt8, _ rhs:UInt8) -> UInt8 {
        return lhs > rhs ? lhs : rhs
    }
    
    static public func min(_ lhs:UInt8, _ rhs:UInt8) -> UInt8 {
        return lhs < rhs ? lhs : rhs
    }
    
    public static var infinity:UInt8{ return UInt8.infinity }
    
    public static func trunc(_ value:UInt8) -> Int { return Int(value) }
}

extension Float: FloatNumericType {
    init<StorageType:Storage>
        (_ tensor:Tensor<StorageType>) where StorageType.ElementType == Float
    {
        precondition(tensor.shape.elements == 1, "Can only convert tensors with a single element to a Float")
        
        self.init(tensor.storage[0])
    }
    
    static public func exp(_ value:Float) -> Float {
        return Foundation.exp(value)
    }
    
    static public func sqrt(_ value:Float) -> Float {
        return Foundation.sqrtf(value)
    }
    
    static public func pow(_ value:Float, _ power:Float) -> Float {
        return powf(value, power)
    }
    static public func log(_ value:Float) -> Float {
        return Foundation.log(value)
    }
    
    static public func max(_ lhs:Float, _ rhs:Float) -> Float {
        return lhs > rhs ? lhs : rhs
    }
    
    static public func min(_ lhs:Float, _ rhs:Float) -> Float {
        return lhs < rhs ? lhs : rhs
    }
    
    static public func tanh(_ value:Float) -> Float {
        return Darwin.tanhf(value)
    }
    
    public static func trunc(_ value:Float) -> Int {
        return Int(value)
    }
}

extension Double: FloatNumericType {
    init<StorageType:Storage>
        (_ tensor:Tensor<StorageType>) where StorageType.ElementType == Double
    {
        precondition(tensor.shape.elements == 1, "Can only convert tensors with a single element to a Double")
        
        self.init(tensor.storage[0])
    }
    
    static public func exp(_ value: Double) -> Double {
        return Foundation.exp(value)
    }
    
    static public func sqrt(_ value:Double) -> Double {
        return Foundation.sqrt(value)
    }
    
    static public func pow(_ value:Double, _ power:Double) -> Double {
        return Foundation.pow(value, power)
    }
    
    static public func log(_ value:Double) -> Double {
        return Foundation.log(value)
    }
    
    static public func max(_ lhs:Double, _ rhs:Double) -> Double {
        return lhs > rhs ? lhs : rhs
    }
    
    static public func min(_ lhs:Double, _ rhs:Double) -> Double {
        return lhs < rhs ? lhs : rhs
    }
    
    static public func tanh(_ value:Double) -> Double {
        return Darwin.tanh(value)
    }
    
    public static func trunc(_ value:Double) -> Int {
        return Int(value)
    }
}

