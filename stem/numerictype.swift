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
    
    public static var infinity:Float { return Float.infinity }
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
    
    public static var infinity:Double { return Double.infinity }
}

