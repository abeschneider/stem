//
//  op.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/12/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation
import Tensor

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

func createUID() -> Int {
    return uniform()
}

open class Op<S:Storage>: OpType, Copyable, Hashable, CustomStringConvertible {
    public typealias InputAction = (String, [Source<S>]) -> ()
    
    open var id:Int = createUID()
    open var inputs:OrderedDictionary<InputType<S>>
    open var outputs:OrderedDictionary<Tensor<S>>
    
    // convenience variable for case when there is only a single output
    open var output:Tensor<S> {
        get { return outputs["output"]![0] }
        set { outputs["output"] = [newValue] }
    }
    
    var inputActions:[String:InputAction] = [:]
    
    public init(inputs:[String], outputs:[String]) {
        self.inputs = OrderedDictionary<InputType<S>>(inputs.map { ($0, InputType<S>()) })
        self.outputs = OrderedDictionary<Tensor<S>>(outputs.map { ($0, Tensor<S>()) })
    }
    
    // required for Copyable
    required public init(op:Op<S>, shared:Bool) {
        inputs = OrderedDictionary<InputType<S>>()
        outputs = OrderedDictionary<Tensor<S>>()
        assertionFailure()
    }
    
    open func apply() { assertionFailure() }
    open func reset() { fill(output, value: 0) }
    
    open func setAction(_ key:String, action:@escaping (String, [Source<S>]) -> ()) { inputActions[key] = action }
    open func params() -> [Tensor<S>] { return [] }
    open var hashValue: Int { return id }
    
    open var description: String {
        let className = String(describing: Mirror(reflecting: self).subjectType)
        let input_values:String = inputsToString()
        let output_values:String = outputsToString()
        
        return "<#\(id) \(className)> inputs: {\(input_values)} outputs: {\(output_values)}>"
    }
    
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
    
    func setInput(_ inputLabel:String, to:[Source<S>]) {
        inputs[inputLabel]![0] = InputType(to[0].op, to[0].label)
        
        if let action = inputActions[inputLabel] {
            action(inputLabel, to)
        }
    }
    
    func setInput(_ inputLabel:String, to:Op<S>, _ outputLabel:String="output") {
        inputs[inputLabel]![0] = InputType(to, outputLabel)
        
        if let action = inputActions[inputLabel] {
            action(inputLabel, [Source(op: to, label: outputLabel)])
        }
    }
    
    func setInput(_ inputLabel:String, to:[Op<S>], _ outputLabel:String="output") {
        inputs[inputLabel] = to.map { InputType($0, outputLabel) }
        
        if let action = inputActions[inputLabel] {
            action(inputLabel, to.map { Source(op: $0, label: outputLabel) })
        }
    }
    
    func getInput(_ label:String) -> Op<S> { return inputs[label]![0].op! }
}

public func ==<S:Storage>(lhs:Op<S>, rhs:Op<S>) -> Bool {
    return lhs.id == rhs.id
}
