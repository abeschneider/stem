//
//  op.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/12/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation

// contains selected output from a given op that
// provides input to target op
public struct InputType<S:Storage> {
    public var op:Op<S>?
    
    // outputs of op
    public var outputs:[Tensor<S>]
    
    public var index:Int?
    
    public init() {
        outputs = []
        index = 0
    }
    
    public init(_ op:Op<S>, _ label:String) {
        self.op = op
        outputs = op.outputs[label]!
        index = 0
    }
    
    public init(_ op:Op<S>) {
        self.op = op
        outputs = op.outputs["output"]!
        index = 0
    }
    
    public func output() -> Tensor<S> {
        return outputs[index!]
    }
    
    public func output() -> [Tensor<S>] {
        return outputs
    }
}

// source for a connection
public struct Source<S:Storage> {
    public var op:Op<S>
    public var label:String
    public var index:Int?
    
    public init(op:Op<S>, label:String="output", index:Int?=0) {
        self.op = op
        self.label = label
        self.index = index
    }
}

// target for a connection
public struct Target<S:Storage> {
    public var op:Op<S>
    public var label:String
    public var index:Int?
    
    
    public init(op:Op<S>, label:String="input", index:Int?=0) {
        self.op = op
        self.label = label
        self.index = index
    }
}

// Op without associated type
public protocol OpType {
    func apply()
    func reset()
}

public protocol Copyable {
    associatedtype StorageType:Storage
    init(op:Op<StorageType>, shared:Bool)
}

public func copy<T:OpType & Copyable>(op:T, shared:Bool) -> T {
    return type(of: op).init(op: op as! Op<T.StorageType>, shared: shared)
}

open class Op<S:Storage>: OpType, Copyable, Hashable, CustomStringConvertible {
    public typealias InputAction = (String, [Op<S>]) -> ()
    
    open var id:Int = createUID()
    
    open var inputs:OrderedDictionary<InputType<S>>
    open var outputs:OrderedDictionary<Tensor<S>>
    
    // convenience variable for case when there is only a single output
    open var output:Tensor<S> {
        get { return outputs["output"]![0] }
        set { outputs["output"] = [newValue] }
    }
    
    open var inputActions:[String:InputAction] = [:]
    
    public init() {
        inputs = OrderedDictionary<InputType<S>>()
        outputs = OrderedDictionary<Tensor<S>>()
    }
    
    public init(inputs:[String], outputs:[String]) {
        self.inputs = OrderedDictionary<InputType<S>>(inputs.map { ($0, InputType<S>()) })
        self.outputs = OrderedDictionary<Tensor<S>>(outputs.map { ($0, Tensor<S>()) })
    }
    
    required public init(op:Op<S>, shared:Bool) {
        inputs = OrderedDictionary<InputType<S>>()
        outputs = OrderedDictionary<Tensor<S>>()
    }
    
    open func setInput(_ inputLabel:String, to:InputType<S>) {
        inputs[inputLabel] = [to]
        
        if let action = inputActions[inputLabel] {
            action(inputLabel, [to.op!])
        }
    }
    
    open func setInput(_ inputLabel:String, to:Op<S>, _ outputLabel:String="output") {
        inputs[inputLabel]![0] = InputType(to, outputLabel)
        
        if let action = inputActions[inputLabel] {
            action(inputLabel, [to])
        }
    }
    
    open func setInput(_ inputLabel:String, to:[Op<S>], _ outputLabel:String="output") {
        inputs[inputLabel] = to.map { InputType($0, outputLabel) }
        
        if let action = inputActions[inputLabel] {
            action(inputLabel, to)
        }
    }
    
    open func setAction(_ key:String, action:@escaping (String, [Op<S>]) -> ()) {
        inputActions[key] = action
    }
    
    open func getInput(_ label:String) -> Op<S> {
        return inputs[label]![0].op!
    }
    
    open func apply() {
        assertionFailure()
    }
    
    open func reset() {
        fill(output, value: 0)
    }
    
    open func params() -> [Tensor<S>] {
        return []
    }
    
    open var hashValue: Int { return id }
    
    func inputsToString() -> String {
        return inputs.keys.map {
            if let op = inputs[$0]![0].op {
                return "\($0):\(op.outputs[0][0].shape.dims)"
            } else {
                return "\($0):<empty>"
            }
            }.joined(separator: ", ")
    }
    
    func outputsToString() -> String {
        return outputs.map { String(describing: $0[0].shape.dims) }.joined(separator: ", ")
    }
    
    open var description: String {
        let className = String(describing: Mirror(reflecting: self).subjectType)
        let input_values:String = inputsToString()
        let output_values:String = outputsToString()
        
        return "<#\(id) \(className)> inputs: {\(input_values)} outputs: {\(output_values)}>"
    }
}

public func ==<S:Storage>(lhs:Op<S>, rhs:Op<S>) -> Bool {
    return lhs.id == rhs.id
}

public func connect<S:Storage>(from:Op<S>, _ outputLabel:String="output", to:Op<S>, _ inputLabel:String="input") {
    to.setInput(inputLabel, to: from, outputLabel)
}

public func connect<S:Storage>(from:[Op<S>], _ outputLabel:String="output", to:Op<S>, _ inputLabel:String="input") {
    to.setInput(inputLabel, to: from, outputLabel)
}

public func connect<S:Storage>(_ source:Source<S>, _ target:Target<S>) {
    target.op.setInput(target.label, to: source.op, source.label)
}
