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
    
    // TODO: consider allowing the Op to be Optional and get rid of NoOp.
    public var inputs:OrderedDictionary<S>
    
//    public var output:Tensor<StorageType>
    public var outputs:[Tensor<S>]
    
    // convenience variable for case when there is only a single output
    public var output:Tensor<S> {
        get { return outputs[0] }
        set { outputs[0] = newValue }
    }
    
    public var inputActions:[String:InputAction] = [:]
    
    required public init(outputs:[Tensor<S>], actions:[String:InputAction]=[:]) {
        self.inputs = OrderedDictionary<S>()
        self.outputs = outputs
        inputActions = actions
    }
    
    required public init(inputs:[(String, Op<S>)], outputs:[Tensor<S>], actions:[String:InputAction]=[:]) {
        self.inputs = OrderedDictionary<S>(inputs)
        self.outputs = outputs
        inputActions = actions
    }
    
    required public init(inputs:[(String, [Op<S>])], outputs:[Tensor<S>], actions:[String:InputAction]=[:]) {
        self.inputs = OrderedDictionary<S>(inputs)
        self.outputs = outputs
        inputActions = actions
    }
    
    required public init(inputs:OrderedDictionary<S>, outputs:[Tensor<S>], actions:[String:InputAction]=[:]) {
        self.inputs = inputs
        self.outputs = outputs
        inputActions = actions
    }

    
    required public init(op:Op<S>, shared:Bool) {
        self.inputs = OrderedDictionary<S>()
//        output = Tensor<S>(op.output.shape)
        outputs = op.outputs.map { Tensor<S>($0.shape) }
    }
    
    public func setInput(label:String, to:OpType) {
        inputs[label] = to as? Op<S>
        
        if let action = inputActions[label] {
            action(label, to as! Op<S>)
        }
    }
    
    public func setInput(label:String, to:[Op<S>]) {
        inputs[label] = to //to as? [Op<S>]
        
//        if let action = inputActions[label] {
//            action(label, to as! Op<S>)
//        }
    }
    
    public func setAction(key:String, action:InputAction) {
        inputActions[key] = action
    }
    
    public func getInput(label:String) -> Op<S> {
        return inputs[label]!
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
//        let inputShapes:[String] = inputs.map {
//            switch $0 {
//            case .OpInput(let op):
//                return String(op.output.shape.dims)
//            case .ArrayInput(let ops):
//                return (ops.map { String($0.output.shape.dims) }).joinWithSeparator(", ")
//            }
//        }
//        
//        return "<\(className): inputs=\(inputShapes), outputs=\(output.shape.dims)>"
        return "<\(className)>"
    }
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

public protocol Loss: Differentiable, OpType {
    associatedtype StorageType:Storage
    
    var value:StorageType.ElementType { get }
}

public class NoOp<S:Storage>: Op<S> {
    public init() {
        super.init(outputs: [])
    }
}

public class Symbol<S:Storage>: Op<S> {
    public init(_ value:S.ElementType) {
        super.init(outputs: [Tensor<S>([value])])
    }
    
    public init(_ input:Tensor<S>) {
        super.init(outputs: [input])
    }
    
    public init(_ shape:Extent) {
        super.init(outputs: [Tensor<S>(shape)])
    }
    
    // required for Copyable
    public required init(op:Op<S>, shared:Bool) {
        super.init(outputs: [Tensor<S>(op.output.shape)])
    }
    
    public func set(input:Tensor<S>) {
        output = copy(input)
    }
    
    public override func apply() {}
}

public class Sigmoid<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    var input:Op<S> { return inputs[0]! }
    
    public init() {
        super.init(inputs: [("input", NoOp<S>())],
                   outputs: [Tensor<S>()])
        
        setAction("input", action: self.inputSet)
    }
    
    public init(size:Int) {
        super.init(inputs: [("input", NoOp<S>())],
                   outputs: [Tensor<S>(Extent(size))])
        
        setAction("input", action: self.inputSet)
    }
    
    // required for Copyable
    public required init(op:Op<S>, shared:Bool) {
        super.init(inputs: [("input", NoOp<S>())],
                   outputs: [Tensor<S>(op.output.shape)])
    }
    
    func inputSet(label:String, op:Op<S>) {
        output.resize(op.output.shape)
    }
    
    public override func apply() {
        sigmoid(input.output, output: output)
    }
}

public class SigmoidGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    public typealias StorageType = S
    public typealias OpType = Sigmoid<S>
    
    public var sigmoid:Sigmoid<S> {
        let op:Op<S> = inputs[0]!
        return op as! Sigmoid<S>
    }
    
    public var input:Tensor<S> { return inputs[1]!.output }
    public var gradOutput:Tensor<S> { return inputs[2]!.output }
    
    public required init(op:Sigmoid<S>) {
        let s:Op<S> = op.inputs[0]!
        super.init(inputs: [("op", op), ("input", s), ("gradOutput", NoOp<S>())],
                   outputs: [Tensor<S>(Extent(op.output.shape))])
    }

    public init(size:Int) {
        super.init(inputs: [("op", NoOp<S>()), ("input", NoOp<S>()), ("gradOutput", NoOp<S>())],
                   outputs: [Tensor<S>(Extent(size))])
    }
    
    public init(op:Sigmoid<S>, input:Op<S>, gradInput:Op<S>) {
        super.init(inputs: [("op", op), ("input", input), ("gradOutput", gradInput)],
                   outputs: [Tensor<S>(input.output.shape)])
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
    
    public var input:Op<S> { return inputs[0]! }
    
    public init(outputSize:Int) {
        weight = Tensor<S>()
        bias = zeros(Extent(outputSize))
        super.init(inputs: [("input", NoOp<S>())],
                   outputs: [zeros(Extent(outputSize))])
        
        setAction("input", action: self.inputSet)
    }
    
    public init(_ input:Op<S>, outputSize:Int) {
        let inputSize = input.output.shape[0]
        weight = uniform(Extent(outputSize, inputSize))
        bias = zeros(Extent(outputSize))
        super.init(inputs: [("input", input)],
                   outputs: [zeros(Extent(outputSize))])
        
        setAction("input", action: self.inputSet)
    }

    
    public init(inputSize:Int, outputSize:Int) {
        weight = uniform(Extent(outputSize, inputSize))
        bias = zeros(Extent(outputSize))
        super.init(inputs: [("input", NoOp<S>())],
                   outputs: [zeros(Extent(outputSize))])
        
        setAction("input", action: self.inputSet)
    }
    
    public init(weight:Tensor<S>, bias:Tensor<S>) {
        self.weight = weight
        self.bias = bias
        super.init(inputs: [("input", NoOp<S>())],
                   outputs: [zeros(bias.shape)])
        
        setAction("input", action: self.inputSet)
    }
    
    public required init(op:Op<S>, shared:Bool) {
        let linear = op as! Linear<S>
        weight = shared ? linear.weight : copy(linear.weight)
        bias = shared ? linear.bias : copy(linear.bias)
        super.init(inputs: [("input", NoOp<S>())],
                   outputs: [zeros(bias.shape)])
        
        setAction("input", action: self.inputSet)
    }
    
    func inputSet(label:String, op:Op<S>) {
        let newShape = Extent(output.shape[0], op.output.shape[0])
        if weight.shape != newShape {
            weight.resize(newShape)
            weight.uniform()
        }
    }
    
    public override func apply() {
        // TODO: review, not sure this is the best way to do this
        if input.output.dims == 1 {
            output.resize(Extent(weight.shape[0]))
            dot(weight, input.output, result: output)
            add(output, bias, result: output)
        } else if input.output.dims == 2 {
            output.resize(Extent(weight.shape[0], input.output.shape[1]))
            dot(weight, inputs[0]!.output, result: output)
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
    
    public var op:Op<S> { return inputs[0]! }
    public var linear:Linear<S> { return op as! Linear<S> }
    public var input:Tensor<S> { return inputs[1]!.output }
    public var gradOutput:Tensor<S> { return inputs[2]!.output }
    
    public required init(op:Linear<S>) {
        weight = zeros(op.weight.shape)
        bias = zeros(op.bias.shape)
        super.init(inputs: [("op", op),
                            ("input", op.inputs["input"]!),
                            ("gradOutput", NoOp<S>())],
                   outputs: [zeros(Extent(op.weight.shape[1]))])
    }

    public init(inputSize:Int, outputSize:Int) {
        weight = zeros(Extent(outputSize, inputSize))
        bias = zeros(Extent(outputSize))
        super.init(inputs: [("op", NoOp<S>()), ("input", NoOp<S>()), ("gradOutput", NoOp<S>())],
                   outputs: [zeros(Extent(outputSize))])
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
        
        super.init(inputs: [("op", op), ("input", input), ("gradOutput", gradInput)],
                   outputs: [zeros(Extent(op.weight.shape[1]))])
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
    
    public var input:Op<S> { return inputs[0]! }
    public var target:Op<S> { return inputs[1]! }
    
    public init() {
        super.init(inputs: [("input", NoOp<S>()), ("target", NoOp<S>())],
                   outputs: [Tensor<S>()])
    }
    
    public init(size:Int) {
        super.init(inputs: [("input", NoOp<S>()), ("target", NoOp<S>())],
                   outputs: [zeros(Extent(size))])
    }
    
    public init(target t:Op<S>) {
        super.init(inputs: [("input", NoOp<S>()), ("target", t)],
                   outputs: [zeros(t.output.shape)])
    }
    
    public init(value:Op<S>, target t:Op<S>) {
        super.init(inputs: [("input", value), ("target", t)],
                   outputs: [Tensor<S>(value.output.shape)])
    }
    
    public required init(op:Op<S>, shared:Bool) {
        super.init(inputs: [("input", NoOp<S>()), ("target", NoOp<S>())],
                   outputs: [Tensor<S>(op.output.shape)])
    }
    
    public override func apply() {
        if input.output.dims == 1 {
            sub(input.output, target.output, result: output)
        } else if input.output.dims == 2 {
            sub(input.output, target.output, result: output)
        }
        
        pow(output, 2, result: output)
        value = sum(output)
    }
}

public class L2LossGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    public typealias StorageType = S
    public typealias OpType = L2Loss<S>
    
    public var linear:Op<S> { return inputs[0]! }
    public var input:Op<S> { return inputs[1]! }
    public var target:Op<S> { return inputs[2]! }
    
    public required init(op:L2Loss<S>) {
        let l:Op<S> = op.inputs[0]!
        let t:Op<S> = op.inputs[1]!
        super.init(inputs: [("op", op), ("input", l), ("target", t)],
                   outputs: [Tensor<S>(Extent(op.target.output.shape))])
    }

    public init(size:Int) {
        super.init(inputs: [("op", NoOp<S>()), ("input", NoOp<S>()), ("target", NoOp<S>())],
                   outputs: [Tensor<S>(Extent(size))])
    }
    
    public override func apply() {
        sub(input.output, target.output, result: output)
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
        super.init(inputs: [("input", NoOp<S>()), ("axis", NoOp<S>())],
                   outputs: [Tensor<S>(Extent(1))])
    }
    
    public override func apply() {
        let input:Tensor<S> = inputs[0]!.output
        let axis:S.ElementType = inputs[1]!.output[0]
        let iaxis = Int(value: axis)
        sum(input, axis: iaxis, result: output)
    }
}

public class Log<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    public init(size:Int) {
        super.init(inputs: [("input", NoOp<S>())],
                   outputs: [Tensor<S>(Extent(size))])
        
        setAction("input", action: self.inputSet)
    }
    
    public init() {
        super.init(inputs: [("input", NoOp<S>())],
                   outputs: [Tensor<S>()])
        
        setAction("input", action: self.inputSet)
    }
    
    func inputSet(label:String, op:Op<S>) {
        output.resize(op.output.shape)
    }
    
    public override func apply() {
        if output.shape != inputs[0]!.output.shape {
            output = Tensor<S>(Extent(inputs[0]!.output.shape))
        }
        
        log(inputs[0]!.output, result: output)
    }
}

extension Log:Differentiable {
    public func gradient() -> GradientType {
        return LogGrad<S>(op: self)
    }
}

public class LogGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    public var input:Op<S> { return inputs[1]! }
    
    public required init(op:Log<S>) {
        let opInput:Op<S> = op.inputs[0]!
        super.init(inputs: [("op", op), ("input", opInput), ("gradOutput", NoOp<S>())],
                   outputs: [Tensor<S>(op.output.shape)])
    }
    
    public override func apply() {
        if output.shape != inputs[2]!.output.shape {
            output = Tensor<S>(Extent(inputs[2]!.output.shape))
        }

        fill(output, value: 1)
        output /= inputs[1]!.output
        output *= inputs[2]!.output
    }
    
    public func reset() {
        fill(output, value: 0)
    }
}

public class Concat<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    var input:[Op<S>] { return inputs[0]! }
    
    public init(_ ops:[Op<S>]) {
        super.init(inputs: [("input", ops)],
                   outputs: [Tensor<S>()])
        
        setAction("input", action: self.inputSet)
    }
    
    public convenience init(_ ops:Op<S>...) {
        self.init(ops)
    }
    
    func inputSet(label:String, op:Op<S>) {}
    
    public override func apply() {
        output = concat(input.map { $0.output })
    }
}

public class ConcatGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    var gradOutput:Op<S> { return inputs[2]! }
    
    public required init(op:Concat<S>) {
        let opInputs:[Op<S>] = op.inputs[0]!
        let outputs:[Tensor<S>] = opInputs.map { Tensor<S>($0.output.shape) }

        super.init(outputs: outputs)
        self.setInput("op", to: op)
        self.setInput("input", to: opInputs)
        self.setInput("gradOutput", to: NoOp<S>())
        
        // currently init doesn't support mixed single Ops and array ops
//        super.init(inputs: [("op", op), ("input", opInput), ("gradOutput", NoOp<S>())],
//                   outputs: outputs)
    }
    
    public override func apply() {
        var index:Int = 0
        for output in outputs {
            let first = index
            let last = output.shape[0]+index
            copy(from: gradOutput.output[first..<last], to: output)
            index = last
        }
    }
    
    public func reset() {
        fill(output, value: 0)
    }
}

extension Concat:Differentiable {
    public func gradient() -> GradientType {
        return ConcatGrad<S>(op: self)
    }
}

