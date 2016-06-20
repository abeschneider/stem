//
//  numerictype.swift
//  stem
//
//  Created by Abe Schneider on 12/11/15.
//  Copyright © 2015 none. All rights reserved.
//

import Foundation

public protocol NumericType: AbsoluteValuable, Comparable { //, FloatingPointType {
    func +(lhs:Self, rhs:Self) -> Self
    func -(lhs:Self, rhs:Self) -> Self
    func *(lhs:Self, rhs:Self) -> Self
    func /(lhs:Self, rhs:Self) -> Self
    func %(lhs:Self, rhs:Self) -> Self
    func **(lhs:Self, rhs:Self) -> Self
    
    static func max(lhs: Self, _ rhs: Self) -> Self
    static func min(lhs: Self, _ rhs: Self) -> Self
    
    init(_ v:Int)
}

public protocol FloatNumericType: NumericType, FloatingPointType {
    static func exp(value:Self) -> Self
    static func sqrt(value:Self) -> Self
    static func pow(value: Self, _ power: Self) -> Self
    static func log(value: Self) -> Self


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
    public static func abs(x:Int) -> Int {
        return Int.abs(x)
    }
}

extension Int: NumericType {
    init<StorageType:Storage where StorageType.ElementType == Int>
        (_ tensor:Tensor<StorageType>)
    {
        precondition(tensor.shape.elements == 1, "Can only convert tensors with a single element to a Int")
        
        self.init(tensor.storage[0])

    }
    
    init(value:Int) {
        self.init(value)
    }
    
//    init<T:FloatNumericType>(value:T) {
//        self.init(value: value)
//    }
    
    static public func max(lhs:Int, _ rhs:Int) -> Int {
        return lhs > rhs ? lhs : rhs
    }
    
    static public func min(lhs:Int, _ rhs:Int) -> Int {
        return lhs < rhs ? lhs : rhs
    }
}

extension Float: FloatNumericType {
    init<StorageType:Storage where StorageType.ElementType == Float>
        (_ tensor:Tensor<StorageType>)
    {
        precondition(tensor.shape.elements == 1, "Can only convert tensors with a single element to a Float")
        
        self.init(tensor.storage[0])
    }
    
    static public func exp(value:Float) -> Float {
        return Foundation.exp(value)
    }
    
    static public func sqrt(value:Float) -> Float {
        return Foundation.sqrtf(value)
    }
    
    static public func pow(value:Float, _ power:Float) -> Float {
        return powf(value, power)
    }
    static public func log(value:Float) -> Float {
        return Foundation.log(value)
    }
    
    static public func max(lhs:Float, _ rhs:Float) -> Float {
        return lhs > rhs ? lhs : rhs
    }
    
    static public func min(lhs:Float, _ rhs:Float) -> Float {
        return lhs < rhs ? lhs : rhs
    }
}

extension Double: FloatNumericType {
    init<StorageType:Storage where StorageType.ElementType == Double>
        (_ tensor:Tensor<StorageType>)
    {
        precondition(tensor.shape.elements == 1, "Can only convert tensors with a single element to a Double")
        
        self.init(tensor.storage[0])
    }
    
    static public func exp(value: Double) -> Double {
        return Foundation.exp(value)
    }
    
    static public func sqrt(value:Double) -> Double {
        return Foundation.sqrt(value)
    }
    
    static public func pow(value:Double, _ power:Double) -> Double {
        return Foundation.pow(value, power)
    }
    
    static public func log(value:Double) -> Double {
        return Foundation.log(value)
    }
    
    static public func max(lhs:Double, _ rhs:Double) -> Double {
        return lhs > rhs ? lhs : rhs
    }
    
    static public func min(lhs:Double, _ rhs:Double) -> Double {
        return lhs < rhs ? lhs : rhs
    }
}

