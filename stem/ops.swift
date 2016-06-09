//
//  ops.swift
//  stem
//
//  Created by Schneider, Abraham R. on 5/25/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation

func createUID() -> Int {
    return uniform()
}

func += <K, V> (inout left: [K:V], right: [K:V]) {
    for (k, v) in right {
        left[k] = v
    }
}

public protocol OpType {
    func apply()
    func setInput(label:String, to:OpType)
}

public protocol Copyable {
    associatedtype StorageType:Storage
    init(op:Op<StorageType>, shared:Bool)
}

public func copy<T:protocol<OpType, Copyable>>(op:T, shared:Bool) -> T {
    return op.dynamicType.init(op: op as! Op<T.StorageType>, shared: shared)
}

public class Op<S:Storage>: OpType, Copyable, Hashable, CustomStringConvertible {
    public typealias StorageType = S
    public typealias InputAction = (String, Op<S>) -> ()
    
    public var id:Int = createUID()
    public var inputs:[Op<StorageType>] = []
    public var output:Tensor<StorageType>
    public var inputLabels:[String:Int] = [:]
    public var inputActions:[String:InputAction] = [:]
    
    required public init(inputs:[Op<StorageType>], output:Tensor<StorageType>, labels:[String], actions:[String:InputAction]=[:]) {
        self.inputs = inputs
        self.output = output
        inputActions = actions
        
        for (i, label) in labels.enumerate() {
            inputLabels[label] = i
        }
    }
    
    required public init(op:Op<S>, shared:Bool) {
        inputs = []
        output = Tensor<S>(op.output.shape)
        inputLabels = op.inputLabels
    }
    
    public func setInput(label:String, to:OpType) {
        let index = inputLabels[label]!
        inputs[index] = to as! Op<S>
        
        if let action = inputActions[label] {
            action(label, to as! Op<S>)
        }
    }
    
    public func setAction(key:String, action:InputAction) {
        inputActions[key] = action
    }
    
    public func getInput(label:String) -> Op<StorageType> {
        let index = inputLabels[label]!
        return inputs[index]
    }
    
    public func setOutput(output:Tensor<StorageType>) {
        self.output = output
    }
    
    public func apply() {
        assertionFailure()
    }
    
    public func params() -> [Tensor<StorageType>] {
        return []
    }
    
    public var hashValue: Int { return id }
    
    public var description: String {
        let className = String(Mirror(reflecting: self).subjectType)
        let inputShapes = inputs.map { $0.output.shape.dims }
        
        return "<\(className): inputs=\(inputShapes), outputs=\(output.shape.dims)>"
    }
}

public func ==<S:Storage>(lhs:Op<S>, rhs:Op<S>) -> Bool {
    return lhs.id == rhs.id
}

public protocol GradientType: OpType {
    func reset()
//    func updateGradOutput()
//    func accumulateGradients()
}

public protocol Gradient: GradientType {
    associatedtype StorageType:Storage
    associatedtype OpType
    
    init(op:OpType)    
}

public protocol Differentiable {
    func gradient() -> GradientType
}

public protocol Loss: Differentiable, OpType {
    associatedtype StorageType:Storage
    
    var value:StorageType.ElementType { get }
}

public class NoOp<S:Storage>: Op<S> {
    public init() {
        super.init(inputs: [],
                   output: Tensor<S>(),
                   labels: [])
    }
}

public class Symbol<S:Storage>: Op<S> {
    public init(_ value:S.ElementType) {
        super.init(inputs: [],
                   output: Tensor<S>([value]),
                   labels: [])
    }
    
    public init(_ input:Tensor<S>) {
        super.init(inputs: [],
                   output: input,
                   labels: [])
    }
    
    public init(_ shape:Extent) {
        super.init(inputs: [],
                   output: Tensor<S>(shape),
                   labels: [])
    }
    
    // required for Copyable
    public required init(op:Op<S>, shared:Bool) {
        super.init(inputs: [],
                   output: Tensor<S>(op.output.shape),
                   labels: [])
    }
    
    public func set(input:Tensor<S>) {
        output = copy(input)
    }
    
    public override func apply() {}
}

public class Sigmoid<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    public init() {
        super.init(inputs: [NoOp<S>()],
                   output: Tensor<S>(),
                   labels: ["input"])
        
        setAction("input", action: self.inputSet)
    }
    
    public init(size:Int) {
        super.init(inputs: [NoOp<S>()],
                   output: Tensor<S>(Extent(size)),
                   labels: ["input"])
        
        setAction("input", action: self.inputSet)
    }
    
//    public init(input:Op<S>) {
//        super.init(inputs: [input], output: Tensor<S>(input.output.shape), labels: ["input"])
//    }

    // required for Copyable
    public required init(op:Op<S>, shared:Bool) {
        super.init(inputs: [NoOp<S>()], output: Tensor<S>(op.output.shape), labels: ["input"])
    }
    
    func inputSet(label:String, op:Op<S>) {
        output.resize(op.output.shape)
    }
    
    public override func apply() {
        sigmoid(inputs[0].output, output: output)
    }
}

public class SigmoidGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    public typealias StorageType = S
    public typealias OpType = Sigmoid<S>
    
    public var sigmoid:Sigmoid<S> { return inputs[0] as! Sigmoid<S> }
    public var input:Tensor<S> { return inputs[1].output }
    public var gradOutput:Tensor<S> { return inputs[2].output }
    
    public required init(op:Sigmoid<S>) {
        super.init(inputs: [op, op.inputs[0], NoOp<S>()],
                   output: Tensor<S>(Extent(op.output.shape)),
                   labels: ["op", "input", "gradOutput"])
    }

    public init(size:Int) {
        super.init(inputs: [NoOp<S>(), NoOp<S>(), NoOp<S>()],
                   output: Tensor<S>(Extent(size)),
                   labels: ["op", "input", "gradOutput"])
    }
    
    public init(op:Sigmoid<S>, input:Op<S>, gradInput:Op<S>) {
        super.init(inputs: [op, input, gradInput],
                   output: Tensor<S>(input.output.shape),
                   labels: ["op", "input", "gradOutput"])
    }
    
    public override func apply() {
        // sigmoid(x)*(1 - sigmoid(x))
        sub(S.ElementType(1.0), sigmoid.output, result: output)
        output *= sigmoid.output
        output *= gradOutput
    }
    
    public func reset() {
        fill(output, value: 0)
    }
}

extension Sigmoid:Differentiable {
    public func gradient() -> GradientType {
        return SigmoidGrad<S>(op: self)
    }
}


public class Linear<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    public typealias StorageType = S
    public typealias OpType = Linear<S>
    
    public var weight:Tensor<S>
    public var bias:Tensor<S>
    
    public var input:Tensor<S> { return inputs[0].output }
    
    
    public init(outputSize:Int) {
        weight = Tensor<S>()
        bias = zeros(Extent(outputSize))
        super.init(inputs: [NoOp<S>()],
                   output: zeros(Extent(outputSize)),
                   labels: ["input"])
        
        setAction("input", action: self.inputSet)
    }
    
    public init(inputSize:Int, outputSize:Int) {
        weight = uniform(Extent(outputSize, inputSize))
        bias = zeros(Extent(outputSize))
        super.init(inputs: [NoOp<S>()],
                   output: zeros(Extent(outputSize)),
                   labels: ["input"])
        
        setAction("input", action: self.inputSet)
    }
    
    public init(weight:Tensor<S>, bias:Tensor<S>) {
        self.weight = weight
        self.bias = bias
        super.init(inputs: [NoOp<S>()],
                   output: zeros(bias.shape),
                   labels: ["input"])
        
        setAction("input", action: self.inputSet)
    }
    
    public required init(op:Op<S>, shared:Bool) {
        let linear = op as! Linear<S>
        weight = shared ? linear.weight : copy(linear.weight)
        bias = shared ? linear.bias : copy(linear.bias)
        super.init(inputs: [NoOp<S>()],
                   output: zeros(bias.shape),
                   labels: ["input"])
        
        setAction("input", action: self.inputSet)
    }
    
    func inputSet(label:String, op:Op<S>) {
        weight.resize(Extent(output.shape[0], op.output.shape[0]))
        weight.uniform()
    }
    
    public override func apply() {
        // TODO: review, not sure this is the best way to do this
        if input.dims == 1 {
            output.resize(Extent(weight.shape[0]))
            dot(weight, inputs[0].output, result: output)
            add(output, bias, result: output)
        } else if input.dims == 2 {
            output.resize(Extent(weight.shape[0], input.shape[1]))
            dot(weight, inputs[0].output, result: output)
            add(output, bias.reshape(Extent(bias.shape[0], 1)), result: output)
        } else {
            assertionFailure()
        }
    }
    
    public override func params() -> [Tensor<S>] {
        return [weight, bias]
    }
}

public class LinearGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    public typealias StorageType = S

    public var weight:Tensor<S>
    public var bias:Tensor<S>
    
    public var linear:Linear<S> { return inputs[0] as! Linear<S> }
    public var input:Tensor<S> { return inputs[1].output }
    public var gradOutput:Tensor<S> { return inputs[2].output }
    
    public required init(op:Linear<S>) {
        weight = zeros(op.weight.shape)
        bias = zeros(op.bias.shape)
        super.init(inputs: [op, op.inputs[0], NoOp<S>()],
                   output: zeros(Extent(op.weight.shape[1])),
                   labels: ["op", "input", "gradOutput"])
    }

    public init(inputSize:Int, outputSize:Int) {
        weight = zeros(Extent(outputSize, inputSize))
        bias = zeros(Extent(outputSize))
        super.init(inputs: [NoOp<S>(), NoOp<S>(), NoOp<S>()],
                   output: zeros(Extent(outputSize)),
                   labels: ["op", "input", "gradOutput"])
    }
    
    public required init(op:Linear<S>, input:Op<S>, gradInput:Op<S>, weight:Tensor<S>?=nil, bias:Tensor<S>?=nil) {
        if let w = weight {
            self.weight = w
        } else {
            self.weight = zeros(op.weight.shape)
        }
        
        if let b = bias {
            self.bias = b
        } else {
            self.bias = zeros(Extent(op.weight.shape[0]))
        }
        
        super.init(inputs: [op, input, gradInput],
                   output: zeros(Extent(op.weight.shape[1])),
                   labels: ["op", "input", "gradOutput"])
    }
    
    // need to separate apply into accumulate and apply
    public override func apply() {
        if input.dims == 1 {
            outer(gradOutput, input, addTo: weight)
            bias += gradOutput
        } else if input.dims == 2 {
            output.resize(Extent(linear.weight.shape[1], gradOutput.shape[1]))
            if bias.dims == 1 {
                bias.shape = Extent(bias.shape[0], 1)
            }
            matmul(gradOutput, input.transpose(), addTo: weight)
            bias += sum(gradOutput, axis: 1)
        } else {
            assertionFailure()
        }
        
        dot(linear.weight.transpose(), gradOutput, addTo: output)
    }
    
    public override func params() -> [Tensor<S>] {
        return [weight, bias]
    }
    
    public func reset() {
        fill(weight, value: 0)
        fill(bias, value: 0)
        fill(output, value: 0)
    }
}

extension Linear:Differentiable {
    public func gradient() -> GradientType {
        return LinearGrad<S>(op: self)
    }
}

public class L2Loss<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Loss {
    public typealias StorageType = S
    
    public var value:S.ElementType = 0
    
    public var input:Tensor<S> { return inputs[0].output }
    public var target:Tensor<S> { return inputs[1].output }
    
    public init() {
        super.init(inputs: [NoOp<S>(), NoOp<S>()],
                   output: Tensor<S>(),
                   labels: ["input", "target"])
    }
    
    public init(size:Int) {
        super.init(inputs: [NoOp<S>(), NoOp<S>()],
                   output: zeros(Extent(size)),
                   labels: ["input", "target"])
    }
    
    public init(target t:Op<S>) {
        super.init(inputs: [NoOp<S>(), t],
                   output: zeros(t.output.shape),
                   labels: ["input", "target"])
    }
    
    public init(value:Op<S>, target t:Op<S>) {
        super.init(inputs: [value, t],
                   output: Tensor<S>(value.output.shape),
                   labels: ["input", "target"])
    }
    
    public required init(op:Op<S>, shared:Bool) {
        super.init(inputs: [NoOp<S>(), NoOp<S>()],
                   output: Tensor<S>(op.output.shape),
                   labels: ["input", "target"])
    }
    
    public override func apply() {
        if input.dims == 1 {
            sub(input, target, result: output)
        } else if input.dims == 2 {
            sub(input, target, result: output)
        }
        
        pow(output, 2, result: output)
        value = sum(output)
    }
}

public class L2LossGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    public typealias StorageType = S
    public typealias OpType = L2Loss<S>
    
    public var input:Tensor<S> { return inputs[1].output }
    public var target:Tensor<S> { return inputs[2].output }
    
    public required init(op:L2Loss<S>) {
        super.init(inputs: [op, op.inputs[0], op.inputs[1]],
                   output: Tensor<S>(Extent(op.target.shape)),
                   labels: ["op", "input", "target"])
    }

    public init(size:Int) {
        super.init(inputs: [NoOp<S>(), NoOp<S>(), NoOp<S>()],
                   output: Tensor<S>(Extent(size)),
                   labels: ["op", "input", "target"])
    }
    
//    public init(op:L2Loss<S>, input:Op<S>, target:Op<S>) {
//        super.init(inputs: [op, input, target],
//                   output: Tensor<S>(op.output.shape),
//                   labels: ["op", "input", "target"])
//    }
    
    public override func apply() {
        sub(input, target, result: output)
        output *= 2
    }
    
    public func reset() {
        fill(output, value: 0)
    }
}

extension L2Loss:Differentiable {
    public func gradient() -> GradientType {
        return L2LossGrad<S>(op: self)
    }
}

// need to extend to provide a multi-dimensional version
public class SumOp<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    public init() {
        super.init(inputs: [NoOp<S>(), NoOp<S>()],
                   output: Tensor<S>(Extent(1)),
                   labels: ["input", "axis"])
    }
    
    public override func apply() {
        let input:Tensor<S> = inputs[0].output
        let axis:S.ElementType = inputs[1].output[0]
        let iaxis = Int(value: axis)
        sum(input, axis: iaxis, result: output)
    }
}

public class Log<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    public init(size:Int) {
        super.init(inputs: [NoOp<S>()],
                   output: Tensor<S>(Extent(size)),
                   labels: ["input"])
        
        setAction("input", action: self.inputSet)
    }
    
    public init() {
        super.init(inputs: [NoOp<S>()],
                   output: Tensor<S>(),
                   labels: ["input"])
        
        setAction("input", action: self.inputSet)
    }
    
    func inputSet(label:String, op:Op<S>) {
        output.resize(op.output.shape)
    }
    
    public override func apply() {
        if output.shape != inputs[0].output.shape {
            output = Tensor<S>(Extent(inputs[0].output.shape))
        }
        
        log(inputs[0].output, result: output)
    }
}

extension Log:Differentiable {
    public func gradient() -> GradientType {
        return LogGrad<S>(op: self)
    }
}

public class LogGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    public required init(op:Log<S>) {
        super.init(inputs: [op, op.inputs[0], NoOp<S>()],
                   output: Tensor<S>(op.output.shape),
                   labels: ["op", "input", "gradOutput"])
    }
    
    public override func apply() {
        if output.shape != inputs[2].output.shape {
            output = Tensor<S>(Extent(inputs[2].output.shape))
        }

        fill(output, value: 1)
        output /= inputs[1].output
        output *= inputs[2].output
    }
    
    public func reset() {
        fill(output, value: 0)
    }
}

