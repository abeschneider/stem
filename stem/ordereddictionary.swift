//
//  ordereddictionary.swift
//  stem
//
//  Created by Schneider, Abraham R. on 6/12/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation

public enum TensorParam<S:Storage> {
    case TensorValue(Tensor<S>)
    case TensorArray([Tensor<S>])
    
    public init(_ value:Tensor<S>) {
        self = .TensorValue(value)
    }

    public init(_ values:[Tensor<S>]) {
        self = .TensorArray(values)
    }
}

public struct OrderedDictionary<T>: CollectionType {
    public typealias Index = Array<T>.Index
    public typealias _Element = [T]
    
    public var keys:[String] = []
    public var values:[String:_Element] = [:]
    public var orderedValues:[_Element] = []
    
    public var startIndex:Index { return orderedValues.startIndex }
    public var endIndex:Int { return orderedValues.endIndex }
    
    public init() {}
    
    public init(_ values:[(String, T)]) {
        add(values)
    }
    
    public init(_ values:[(String, _Element)]) {
        add(values)
    }
    
    public mutating func add(values:[(String, T)]) {
        for (key, value) in values {
            self[key] = [value]
        }
    }
    
    public mutating func add(tensors:[(String, _Element)]) {
        for (key, value) in tensors {
            self[key] = value
        }
    }
    
    public mutating func add(key:String, values:[T]) {
        self[key] = values
    }
    
    mutating func setValue(key:String, _ value:T) {
        if values[key] == nil {
            // if key is new, insert it into our indices
            keys.append(key)
            orderedValues.append([value])
        } else {
            // otherwise, just update its current value
            let index = keys.indexOf(key)!
            orderedValues[index] = [value]
        }
        
        values[key] = [value]
    }
    
    mutating func setValue(key:String, _ value:_Element) {
        if values[key] == nil {
            // if key is new, insert it into our indices
            keys.append(key)
            orderedValues.append(value)
        } else {
            // otherwise, just update its current value
            let index = keys.indexOf(key)!
            orderedValues[index] = value
        }
        
        values[key] = value
    }

    public subscript(key:String) -> _Element? {
        get { return values[key] }
        set { setValue(key, newValue!) }
    }
    
    public subscript(key:String) -> T? {
        get {
            if let value = values[key] {
                return value[0]
            }
            return nil
        }
        set { setValue(key, newValue!) }
    }
    
    public subscript(index:Int) -> _Element {
        get {
            return orderedValues[index]
        }
        set {
            orderedValues[index] = newValue
        }
    }
    
    public subscript(index:Int) -> T {
        get {
            precondition(orderedValues[index].count == 1)
            return orderedValues[index][0]
        }
        set {
            orderedValues[index] = [newValue]
        }
    }
}
