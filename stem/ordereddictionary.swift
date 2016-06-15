//
//  ordereddictionary.swift
//  stem
//
//  Created by Schneider, Abraham R. on 6/12/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation

public enum InputType<S:Storage> {
    case OpInput(Op<S>)
    case ArrayInput([Op<S>])
    
    public init(_ op:Op<S>) {
        self = .OpInput(op)
    }

    public init(_ ops:[Op<S>]) {
        self = .ArrayInput(ops)
    }
}

public struct OrderedDictionary<S:Storage>: CollectionType {
    public typealias Index = Array<InputType<S>>.Index
    public typealias _Element = InputType<S>
    
    public var keys:[String] = []
    public var values:[String:InputType<S>] = [:]
    public var orderedValues:[InputType<S>] = []
    
    public var startIndex:Index { return orderedValues.startIndex }
    public var endIndex:Int { return orderedValues.endIndex }
    
    public init() {}
    
    public init(_ values:[(String, InputType<S>)]) {
        add(values)
    }
    
    public init(_ values:[(String, Op<S>)]) {
        add(values)
    }
    
    public init(_ values:[(String, [Op<S>])]) {
        add(values)
    }
    
    public mutating func add(ops:[(String, InputType<S>)]) {
        for (key, value) in ops {
            self[key] = value
        }
    }
    
    public mutating func add(ops:[(String, Op<S>)]) {
        for (key, value) in ops {
            self[key] = InputType<S>(value)
        }
    }
    
    public mutating func add(ops:[(String, [Op<S>])]) {
        for (key, value) in ops {
            self[key] = InputType<S>(value)
        }
    }
    
    public mutating func add(key:String, ops:[Op<S>]) {
        self[key] = InputType<S>(ops)
    }
    
    // unlabeled inputs
//    public mutating func add(key:String, ops:[Op<S>]) -> [Int] {
//        var indices = [Int]()
//        for op in ops {
//            indices.append(orderedValues.count)
//            keys.append(key)
//            orderedValues.append(op)
//        }
//        return indices
//    }
    
    mutating func setValue(key:String, _ value:Op<S>) {
        if values[key] == nil {
            // if key is new, insert it into our indices
            keys.append(key)
            orderedValues.append(InputType<S>(value))
        } else {
            // otherwise, just update its current value
            let index = keys.indexOf(key)!
            orderedValues[index] = InputType<S>(value)
        }
        
        values[key] = InputType<S>(value)
    }
    
    mutating func setValue(key:String, _ value:[Op<S>]) {
        if values[key] == nil {
            // if key is new, insert it into our indices
            keys.append(key)
            orderedValues.append(InputType<S>(value))
        } else {
            // otherwise, just update its current value
            let index = keys.indexOf(key)!
            orderedValues[index] = InputType<S>(value)
        }
        
        values[key] = InputType<S>(value)
    }

    public subscript(key:String) -> InputType<S>? {
        get { return values[key] }
        set {
            switch newValue! {
            case .OpInput(let op):
                setValue(key, op)
            case .ArrayInput(let ops):
                setValue(key, ops)
            }
        }
    }
    
    public subscript(key:String) -> Op<S>? {
        get {
            switch values[key]! {
            case .OpInput(let op):
                return op
            case .ArrayInput:
                return nil
            }
        }
        
        set {
            setValue(key, newValue!)
        }
    }
    
    public subscript(key:String) -> [Op<S>]? {
        get {
            switch values[key]! {
            case .OpInput:
                return nil
            case .ArrayInput(let ops):
                return ops
            }
        }
        
        set {
            setValue(key, newValue!)
        }
    }

    
    public subscript(index:Int) -> InputType<S> {
        get {
            return orderedValues[index]
        }
    }
    
    public subscript(index:Int) -> Op<S>? {
        get {
            switch orderedValues[index] {
            case .OpInput(let op):
                return op
            case .ArrayInput:
                return nil
            }
        }
    }
    
    public subscript(index:Int) -> [Op<S>]? {
        get {
            switch orderedValues[index] {
            case .OpInput:
                return nil
            case .ArrayInput(let ops):
                return ops
            }
        }
    }
}
