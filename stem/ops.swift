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

public class Op<S:Storage>: OpType, Hashable {
    public typealias StorageType = S
    
    public var id:Int = createUID()
    public var inputs:[Op<StorageType>] = []
    public var output:Tensor<StorageType>?
    public var inputLabels:[String:Int] = [:]
    
    required public init(inputs:[Op<StorageType>], output:Tensor<StorageType>?, labels:[String]) {
        self.inputs = inputs
        self.output = output
        
        for (i, label) in labels.enumerate() {
            inputLabels[label] = i
        }
    }
    
    public func setInput(label:String, to:OpType) {
        let index = inputLabels[label]!
        inputs[index] = to as! Op<S>
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
}

public func ==<S:Storage>(lhs:Op<S>, rhs:Op<S>) -> Bool {
    return lhs.id == rhs.id
}

public protocol GradientType: OpType {
    func reset()
}

public protocol Gradient: GradientType {
    associatedtype StorageType:Storage
    associatedtype OpType
    
    init(op:OpType)    
}

public protocol Differentiable {
    func gradient() -> GradientType
}

public protocol Loss: Differentiable {
    associatedtype StorageType:Storage
    
    var value:StorageType.ElementType { get }
}

public class NoOp<S:Storage>: Op<S> {
    public init() {
        super.init(inputs: [],
                   output: nil,
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
    
    public func set(input:Tensor<S>) {
        output = copy(input)
    }
    
    public override func apply() {}
}

public class Sigmoid<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    public init(size:Int) {
        super.init(inputs: [NoOp<S>()], output: Tensor<S>(Extent(size)), labels: ["input"])
    }
    
    public init(input:Op<S>) {
        super.init(inputs: [input], output: Tensor<S>(input.output!.shape), labels: ["input"])
    }
    
    public override func apply() {
        sigmoid(inputs[0].output!, output: output!)
    }
}

public class SigmoidGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    public typealias StorageType = S
    public typealias OpType = Sigmoid<S>
    
    public var sigmoid:Sigmoid<S> { return inputs[0] as! Sigmoid<S> }
    public var input:Tensor<S> { return inputs[1].output! }
    public var gradOutput:Tensor<S> { return inputs[2].output! }
    
    public required init(op:Sigmoid<S>) {
        super.init(inputs: [op, op.inputs[0], NoOp<S>()],
                   output: Tensor<S>(Extent(op.output!.shape)),
                   labels: ["op", "input", "gradOutput"])
    }

    public init(size:Int) {
        super.init(inputs: [NoOp<S>(), NoOp<S>(), NoOp<S>()],
                   output: Tensor<S>(Extent(size)),
                   labels: ["op", "input", "gradOutput"])
    }
    
    public init(op:Sigmoid<S>, input:Op<S>, gradInput:Op<S>) {
        super.init(inputs: [op, input, gradInput],
                   output: Tensor<S>(input.output!.shape),
                   labels: ["op", "input", "gradOutput"])
    }
    
    public override func apply() {
        // sigmoid(x)*(1 - sigmoid(x))
        sub(S.ElementType(1.0), sigmoid.output!, result: output!)
        output! *= sigmoid.output!
        output! *= gradOutput
    }
    
    public func reset() {
        fill(output!, value: 0)
    }
}

extension Sigmoid:Differentiable {
    public func gradient() -> GradientType {
        return SigmoidGrad<S>(op: self)
    }
}


public class Linear<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    public typealias StorageType = S
    
    public var weight:Tensor<S>
    public var bias:Tensor<S>
    
    public init(inputSize:Int, outputSize:Int) {
        weight = uniform(Extent(outputSize, inputSize))
        bias = zeros(Extent(outputSize))
        super.init(inputs: [NoOp<S>()],
                   output: zeros(Extent(outputSize)),
                   labels: ["input"])
    }
    
    public init(input:Op<S>, weight:Tensor<S>, bias:Tensor<S>) {
        self.weight = weight
        self.bias = bias
        super.init(inputs: [input],
                   output: zeros(bias.shape),
                   labels: ["input"])
    }
    
    public init(weight:Tensor<S>, bias:Tensor<S>) {
        self.weight = weight
        self.bias = bias
        super.init(inputs: [NoOp<S>()],
                   output: zeros(bias.shape),
                   labels: ["input"])
    }
    
    public override func apply() {
        dot(weight, inputs[0].output!, result: output!)
        add(output!, bias, result: output!)
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
    public var input:Tensor<S> { return inputs[1].output! }
    public var gradOutput:Tensor<S> { return inputs[2].output! }
    
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
    
    public override func apply() {
        outer(gradOutput, input, result: weight)
        bias += gradOutput
        dot(linear.weight.transpose(), gradOutput, addTo: output!)
    }
    
    public override func params() -> [Tensor<S>] {
        return [weight, bias]
    }
    
    public func reset() {
        fill(weight, value: 0)
        fill(bias, value: 0)
        fill(output!, value: 0)
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
    
    public var inputValue:Tensor<S> { return inputs[0].output! }
    public var targetValue:Tensor<S> { return inputs[1].output! }
    
    public init() {
        super.init(inputs: [NoOp<S>(), NoOp<S>()],
                   output: nil,
                   labels: ["input", "target"])
    }
    
    public init(size:Int) {
        super.init(inputs: [NoOp<S>(), NoOp<S>()],
                   output: zeros(Extent(size)),
                   labels: ["input", "target"])
    }
    
    public init(target:Op<S>) {
        super.init(inputs: [NoOp<S>(), target],
                   output: zeros(target.output!.shape),
                   labels: ["input", "target"])
    }
    
    public init(value:Op<S>, target:Op<S>) {
        super.init(inputs: [value, target],
                   output: Tensor<S>(value.output!.shape),
                   labels: ["input", "target"])
    }
    
    public override func apply() {
        sub(inputValue, targetValue, result: output!)
        pow(output!, 2, result: output!)
        value = sum(output!)
    }
}

public class L2LossGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    public typealias StorageType = S
    public typealias OpType = L2Loss<S>
    
    public var inputValue:Tensor<S> { return inputs[1].output! }
    public var targetValue:Tensor<S> { return inputs[2].output! }
    
    public required init(op:L2Loss<S>) {
        super.init(inputs: [op, op.inputs[0], op.inputs[1]],
                   output: Tensor<S>(Extent(op.targetValue.shape)),
                   labels: ["op", "input", "target"])
    }

    public init(size:Int) {
        super.init(inputs: [NoOp<S>(), NoOp<S>(), NoOp<S>()],
                   output: Tensor<S>(Extent(size)),
                   labels: ["op", "input", "target"])
    }
    
    public init(op:L2Loss<S>, input:Op<S>, target:Op<S>) {
        super.init(inputs: [op, input, target],
                   output: Tensor<S>(op.output!.shape),
                   labels: ["op", "input", "target"])
    }
    
    public override func apply() {
        sub(inputValue, targetValue, result: output!)
        output! *= 2
    }
    
    public func reset() {
        fill(output!, value: 0)
    }
}

extension L2Loss:Differentiable {
    public func gradient() -> GradientType {
        return L2LossGrad<S>(op: self)
    }
}


