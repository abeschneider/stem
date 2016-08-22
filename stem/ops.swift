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
    func reset()
}

public protocol Copyable {
    associatedtype StorageType:Storage
    init(op:Op<StorageType>, shared:Bool)
}

public func copy<T:protocol<OpType, Copyable>>(op:T, shared:Bool) -> T {
    return op.dynamicType.init(op: op as! Op<T.StorageType>, shared: shared)
}


public struct InputType<S:Storage> {
    public var op:Op<S>?
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

public class Op<S:Storage>: OpType, Copyable, Hashable, CustomStringConvertible {
    public typealias StorageType = S
    public typealias InputAction = (String, [Op<S>]) -> ()
    
    public var id:Int = createUID()
    
    public var inputs:OrderedDictionary<InputType<S>>
    public var outputs:OrderedDictionary<Tensor<S>>
    
    // convenience variable for case when there is only a single output
    public var output:Tensor<S> {
        get { return outputs["output"]! }
        set { outputs["output"] = newValue }
    }
    
    public var inputActions:[String:InputAction] = [:]
    
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
    
    public func setInput(inputLabel:String, to:InputType<S>) {
        inputs[inputLabel] = to
        
        if let action = inputActions[inputLabel] {
            action(inputLabel, [to.op!])
        }
    }
    
    public func setInput(inputLabel:String, to:Op<S>, _ outputLabel:String="output") {
        inputs[inputLabel] = InputType(to, outputLabel)
        
        if let action = inputActions[inputLabel] {
            action(inputLabel, [to])
        }
    }

    public func setInput(inputLabel:String, to:[Op<S>], _ outputLabel:String="output") {
        inputs[inputLabel] = to.map { InputType($0, outputLabel) }
        
        if let action = inputActions[inputLabel] {
            action(inputLabel, to)
        }
    }
    
    public func setAction(key:String, action:(String, [Op<S>]) -> ()) {
        inputActions[key] = action
    }
    
    public func getInput(label:String) -> Op<S> {
        return inputs[label]!.op!
    }
    
    public func apply() {
        assertionFailure()
    }
    
    public func reset() {
        fill(output, value: 0)
    }
    
    public func params() -> [Tensor<StorageType>] {
        return []
    }
    
    public var hashValue: Int { return id }
    
    func inputsToString() -> String {
        return inputs.keys.map {
            if let op = inputs[$0]![0].op {
                return "\($0):\(op.outputs[0].shape.dims)"
            } else {
                return "\($0):<empty>"
            }
        }.joinWithSeparator(", ")
    }
    
    func outputsToString() -> String {
        return outputs.map { String($0[0].shape.dims) }.joinWithSeparator(", ")
    }
    
    public var description: String {
        let className = String(Mirror(reflecting: self).subjectType)
        let input_values:String = inputsToString()
        let output_values:String = outputsToString()
        
        return "<#\(id) \(className)> inputs: {\(input_values)} outputs: {\(output_values)}>"
    }
}

public func ==<S:Storage>(lhs:Op<S>, rhs:Op<S>) -> Bool {
    return lhs.id == rhs.id
}

public func connect<S:Storage>(from from:Op<S>, _ outputLabel:String="output", to:Op<S>, _ inputLabel:String="input") {
    to.setInput(inputLabel, to: from, outputLabel)
}

public func connect<S:Storage>(from from:[Op<S>], _ outputLabel:String="output", to:Op<S>, _ inputLabel:String="input") {
    to.setInput(inputLabel, to: from, outputLabel)
}

public func connect<S:Storage>(source:Source<S>, _ target:Target<S>) {
    target.op.setInput(target.label, to: source.op, source.label)
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

public func gradient(op:Differentiable) -> GradientType {
    return op.gradient()
}

public protocol Loss: Differentiable, OpType {
    associatedtype StorageType:Storage
    
    var value:StorageType.ElementType { get }
}

public class DataSequence<S:Storage>: GeneratorType {
    var data:[Tensor<S>] = []
    var index:Int = 0
    
    public func next() -> Tensor<S>? {
        if index < data.count {
            return data[index]
        }
        
        return nil
    }
}

public class Constant<S:Storage>: Op<S> {
    public init(_ value:S.ElementType) {
        super.init(inputs: [], outputs: ["output"])
        outputs["output"] = Tensor<S>([value])
    }
    
    public init(_ value:Tensor<S>) {
        super.init(inputs: [], outputs: ["output"])
        outputs["output"] = value
    }
    
    public init(_ shape:Extent) {
        super.init(inputs: [], outputs: ["output"])
        outputs["output"] = Tensor<S>(shape)
    }
    
    public init(data:DataSequence<S>) {
        super.init(inputs: [], outputs: ["output"])
        outputs["output"] = data.next()!
    }
    
    // required for Copyable
    public required init(op:Op<S>, shared:Bool) {
        super.init(inputs: [], outputs: ["output"])
        outputs["output"] = Tensor<S>(op.output.shape)
    }
    
    public func set(input:Tensor<S>, copy makeCopy:Bool=true) {
        if makeCopy {
            copy(from: input, to: output)
        } else {
            output = input
        }
    }
    
    public override func apply() {}
}

public class Variable<S:Storage>: Op<S> {
    public var _input:Tensor<S> { return inputs[0].output() }

    public override init() {
        super.init(inputs: ["input"], outputs: ["output"])
        setAction("input", action: inputSet)
    }
    
    func inputSet(label:String, ops:[Op<S>]) {
        outputs["output"] = Tensor<S>(ops[0].output.shape)
    }
    
    public override func apply() {
        copy(from: _input, to: output)
    }
}

public class VariableGradient<S:Storage>: Op<S>, Gradient {
    public var _variable:Tensor<S> { return inputs[0].output() }
    public var _input:Tensor<S> { return inputs[1].output() }
    public var _gradOutput:Tensor<S> { return inputs[2].output() }

    
    public required init(op:Variable<S>) {
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
    }

    public override func apply() {
        copy(from: _input, to: output)
    }
    
    public override func reset() {
        fill(output, value: 0)
    }
}

extension Variable: Differentiable {
    public func gradient() -> GradientType {
        return VariableGradient<S>(op: self)
    }
}

// TODO: remove?
//public class VariableSequence<S:Storage>: Op<S> {
//    var i:Int = 0
//    var vars:[Tensor<S>]
//    public var count:Int { return vars.count }
//    
//    public init(_ vars:[Tensor<S>]) {
//        self.vars = vars
//        super.init(inputs: [], outputs: ["output"])
//        outputs["output"] = self.vars[0]
//    }
//    
//    public required init(op:Op<S>) {
//        let vl = op as! VariableSequence<S>
//        self.vars = vl.vars
//        super.init(inputs: [], outputs: ["output"])
//        outputs["output"] = vl.vars[0]
//    }
//    
//    public override func apply() {
//        defer {
//            i = (i + 1) % vars.count
//        }
//
//        output = vars[i]
//    }
//    
//    public subscript(index:Int) -> Tensor<S> {
//        return vars[index]
//    }
//    
//    public override func reset() {
//        i = 0
//    }
//}

public class SigmoidOp<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    var _input:Tensor<S> { return inputs[0].output() }
    
    public override init() {
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = Tensor<S>()
        setAction("input", action: self.inputSet)
    }
    
    public init(size:Int) {
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = Tensor<S>(Extent(size))
        
        setAction("input", action: self.inputSet)
    }
    
    // required for Copyable
    public required init(op:Op<S>, shared:Bool) {
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = Tensor<S>(op.output.shape)
    }
    
    func inputSet(label:String, input:[Op<S>]) {
        output.resize(_input.shape)
    }
    
    public override func apply() {
        sigmoid(_input, output: output)
    }
}

public class SigmoidGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    public typealias StorageType = S
    public typealias OpType = SigmoidOp<S>
    
    public var _sigmoid:Tensor<S> { return inputs[0].output() }
    public var _input:Tensor<S> { return inputs[1].output() }
    public var _gradOutput:Tensor<S> { return inputs[2].output() }
    
    public required init(op:SigmoidOp<S>) {
        let s:InputType<S> = op.inputs[0]
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        connect(from: op, "output", to: self, "op")
        connect(from: s.op!, "output", to: self, "input")
        outputs["output"] = Tensor<S>(op.output.shape)
    }

    public init(size:Int) {
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        outputs["output"] = Tensor<S>(Extent(size))

    }
    
    public init(op:SigmoidOp<S>, input:Op<S>, gradInput:Op<S>) {
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        connect(from: op, "output", to: self, "op")
        connect(from: input, "output", to: self, "input")
        connect(from: gradInput, "output", to: self, "gradOutput")
        outputs["output"] = Tensor<S>(input.output.shape)
    }
    
    public override func apply() {
        // sigmoid(x)*(1 - sigmoid(x))
        let result = S.ElementType(1.0) - _sigmoid
        result *= _sigmoid
        result *= _gradOutput
        output += result
    }
    
    public override func reset() {
        fill(output, value: 0)
    }
}

extension SigmoidOp: Differentiable {
    public func gradient() -> GradientType {
        return SigmoidGrad<S>(op: self)
    }
}

public class TanhOp<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    var _input:Tensor<S> { return inputs[0].output() }
    
    public override init() {
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = Tensor<S>()
        
        setAction("input", action: self.inputSet)
    }
    
    public init(size:Int) {
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = Tensor<S>(Extent(size))
        
        setAction("input", action: self.inputSet)
    }
    
    // required for Copyable
    public required init(op:Op<S>, shared:Bool) {
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = Tensor<S>(op.output.shape)
    }
    
    func inputSet(label:String, input:[Op<S>]) {
        output.resize(_input.shape)
    }
    
    public override func apply() {
        tanh(_input, output: output)
    }
}

public class TanhGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    public typealias StorageType = S
    public typealias OpType = TanhOp<S>
    
    public var _tanh:Tensor<S> { return inputs[0].output() }
    
    public var _input:Tensor<S> { return inputs[1].output() }
    public var _gradOutput:Tensor<S> { return inputs[2].output() }
    
    public required init(op:TanhOp<S>) {
        let s:InputType<S> = op.inputs[0]
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        connect(from: op, "output", to: self, "op")
        connect(from: s.op!, "output", to: self, "input")
        outputs["output"] = Tensor<S>(op.output.shape)
    }
    
    public init(size:Int) {
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        outputs["output"] = Tensor<S>(Extent(size))

    }
    
    public init(op:SigmoidOp<S>, input:Op<S>, gradInput:Op<S>) {
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        connect(from: op, "output", to: self, "op")
        connect(from: input, "output", to: self, "input")
        connect(from: gradInput, to: self, "gradOutput")
        outputs["output"] = Tensor<S>(op.output.shape)
    }
    
    /*
     dtanh(x)/dx = (1 - tanh^2 x)*dx
    */
    public override func apply() {
        let result = _tanh * _tanh
        result *= -1
        add(S.ElementType(1.0), result, result: result)
        result *= _gradOutput
        output += result
    }
    
    public override func reset() {
        fill(output, value: 0)
    }
}

extension TanhOp: Differentiable {
    public func gradient() -> GradientType {
        return TanhGrad<S>(op: self)
    }
}

public class LinearOp<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    public typealias StorageType = S
    public typealias OpType = LinearOp<S>
    
    public var weight:Tensor<S>
    public var bias:Tensor<S>
    
    public var _input:Tensor<S> { return inputs[0].output() }
    
    public override init() {
        weight = Tensor<S>()
        bias = Tensor<S>()
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = Tensor<S>()
        
        setAction("input", action: self.inputSet)
    }
    
    public init(outputSize:Int) {
        weight = Tensor<S>()
        bias = zeros(Extent(outputSize))
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = zeros(Extent(outputSize))
        
        setAction("input", action: self.inputSet)
    }
    
    public init(_ input:Op<S>, outputSize:Int) {
        let inputSize = input.output.shape[0]
        weight = uniform(Extent(outputSize, inputSize))
        bias = zeros(Extent(outputSize))
        super.init(inputs: ["input"], outputs: ["output"])
        connect(from: input, "output", to: self, "input")
        outputs["output"] = zeros(Extent(outputSize))
        
        setAction("input", action: self.inputSet)
    }
    
    public init(inputSize:Int, outputSize:Int) {
        weight = uniform(Extent(outputSize, inputSize))
        bias = zeros(Extent(outputSize))
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = zeros(Extent(outputSize))
        
        setAction("input", action: self.inputSet)
    }
    
    public init(weight:Tensor<S>, bias:Tensor<S>) {
        self.weight = weight
        self.bias = bias
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = zeros(bias.shape)
        
        setAction("input", action: self.inputSet)
    }
    
    public required init(op:Op<S>, shared:Bool) {
        let linear = op as! LinearOp<S>
        weight = shared ? linear.weight : copy(linear.weight)
        bias = shared ? linear.bias : copy(linear.bias)
        super.init(inputs: ["inputs"], outputs: ["output"])
        outputs["output"] = zeros(bias.shape)
        
        setAction("input", action: self.inputSet)
    }
    
    func inputSet(label:String, input:[Op<S>]) {
        let newShape = Extent(output.shape[0], input[0].output.shape[0])
        if weight.shape != newShape {
            weight.resize(newShape)
            weight.uniform()
        }
    }
    
    public override func apply() {
        // TODO: review, not sure this is the best way to do this
        if _input.dims == 1 {
            output.resize(Extent(weight.shape[0]))
            dot(weight, _input, result: output)
            add(output, bias, result: output)
        } else if _input.dims == 2 {
            output.resize(Extent(weight.shape[0], _input.shape[1]))
            dot(weight, _input, result: output)
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
    
    public var linear:LinearOp<S> {
        let input:InputType<S> = inputs[0]
        return input.op as! LinearOp<S>
    }

    public var _input:Tensor<S> { return inputs[1].output() }
    public var _gradOutput:Tensor<S> { return inputs[2].output() }
    
    public required init(op:LinearOp<S>) {
        weight = zeros(op.weight.shape)
        bias = zeros(op.bias.shape)

        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        connect(from: op, "output", to: self, "op")
        connect(from: op.inputs[0].op!, "output", to: self, "input")
        outputs["output"] = zeros(Extent(op.weight.shape[1]))
    }
    
    public required init(op:LinearOp<S>, input:Op<S>, gradInput:Op<S>, weight:Tensor<S>?=nil, bias:Tensor<S>?=nil) {
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
        
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        connect(from: op, "output", to: self, "op")
        connect(from: op.inputs[0].op!, "output", to: self, "input")
        connect(from: gradInput, "output", to: self, "gradOutput")
        outputs["output"] = zeros(Extent(op.weight.shape[1]))
    }
    
    // need to separate apply into accumulate and apply
    public override func apply() {
        if _input.dims == 1 {
            outer(_gradOutput, _input, addTo: weight)
            bias += _gradOutput
        } else if _input.dims == 2 {
            output.resize(Extent(linear.weight.shape[1], _gradOutput.shape[1]))
            if bias.dims == 1 {
                bias.shape = Extent(bias.shape[0], 1)
            }
            matmul(_gradOutput, _input.transpose(), addTo: weight)
            bias += sum(_gradOutput, axis: 1)
        } else {
            assertionFailure()
        }
        
        dot(linear.weight.transpose(), _gradOutput, addTo: output)
    }
    
    public override func params() -> [Tensor<S>] {
        return [weight, bias]
    }
    
    public override func reset() {
        fill(weight, value: 0)
        fill(bias, value: 0)
        fill(output, value: 0)
    }
}

extension LinearOp: Differentiable {
    public func gradient() -> GradientType {
        return LinearGrad<S>(op: self)
    }
}

public class L2Loss<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Loss {
    public typealias StorageType = S
    
    public var value:S.ElementType = 0
    
    public var _input:Tensor<S> { return inputs[0].output() }
    public var _target:Tensor<S> { return inputs[1].output() }
    
    public override init() {
        super.init(inputs: ["input", "target"], outputs: ["output"])
        outputs["output"] = Tensor<S>()
    }
    
    public init(size:Int) {
        super.init(inputs: ["input", "target"], outputs: ["output"])
        outputs["output"] = zeros(Extent(size))
    }
    
    public init(target t:Op<S>) {
        super.init(inputs: ["input", "target"], outputs: ["output"])
        connect(from: t, "output", to: self, "target")
        outputs["output"] = zeros(t.output.shape)
    }
    
    public init(value:Op<S>, target t:Op<S>) {
        super.init(inputs: ["input", "target"], outputs: ["output"])
        connect(from: value, "output", to: self, "input")
        connect(from: t, "output", to: self, "target")
        outputs["output"] = Tensor<S>(value.output.shape)
    }
    
    public required init(op:Op<S>, shared:Bool) {
        super.init(inputs: ["input", "target"], outputs: ["output"])
        outputs["output"] = Tensor<S>(op.output.shape)
    }
    
    public override func apply() {
        // TODO: change to setAction
        if output.shape != _input.shape {
            output.resize(_input.shape)
        }
        
        if _input.dims == 1 {
            sub(_input, _target, result: output)
        } else if _input.dims == 2 {
            sub(_input, _target, result: output)
        }
        
        pow(output, 2, result: output)
        value = sum(output)
    }
}

public class L2LossGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    public typealias StorageType = S
    public typealias OpType = L2Loss<S>
    
    public var linear:LinearOp<S> {
        let input:InputType<S> = inputs[0]
        return input.op as! LinearOp<S>
    }
    public var _input:Tensor<S> { return inputs[1].output() }
    public var _target:Tensor<S> { return inputs[2].output() }
    
    public required init(op:L2Loss<S>) {
        let l:InputType<S> = op.inputs[0]
        let t:InputType<S> = op.inputs[1]

        super.init(inputs: ["op", "input", "target"], outputs: ["output"])
        connect(from: op, "output", to: self, "op")
        connect(Source(op: l.op!), Target(op: self, label: "input"))
        connect(Source(op: t.op!), Target(op: self, label: "target"))
        outputs["output"] = Tensor<S>(Extent(op._target.shape))
    }

    public init(size:Int) {
        super.init(inputs: ["op", "input", "target"], outputs: ["output"])
        outputs["output"] = Tensor<S>(Extent(size))
    }
    
    public override func apply() {
        sub(_input, _target, result: output)
        output *= 2
    }
    
    public override func reset() {
        fill(output, value: 0)
    }
}

extension L2Loss: Differentiable {
    public func gradient() -> GradientType {
        return L2LossGrad<S>(op: self)
    }
}

// adds together inputs
public class AddOp<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    public var inOps:[Op<S>] {
        let inputs:[InputType<S>] = self.inputs[0]
        return inputs.map { $0.op! }
    }
    
    public init(_ ops:Op<S>...) {
        super.init(inputs: ["input"], outputs: ["output"])
        connect(from: ops, "output", to: self, "input")
        outputs["output"] = [Tensor<S>](count: ops.count, repeatedValue: zeros(ops[0].output.shape))
    }
    
    public override func apply() {
        if output.shape != inOps[0].output.shape {
            output.resize(inOps[0].output.shape)
        }
        
        copy(from: inOps[0].output, to: output)
        for op in inOps[1..<inOps.count] {
            iadd(output, op.output)
        }
    }
}

public class AddOpGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    public var _add:Tensor<S> { return inputs[0].output() }
    public var _input:[Tensor<S>] {
        let _in:[InputType<S>] = inputs[1]
        return _in.map { $0.output() }
    }
    public var _gradOutput:Tensor<S> { return inputs[2].output() }
    
    public required init(op:AddOp<S>) {
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        
        let opInputs:[InputType<S>] = op.inputs[0]
        connect(from: op, "output", to: self, "op")
        connect(from: opInputs.map { $0.op! }, "output", to: self, "input")
        outputs["output"] = [Tensor<S>](count: _input.count, repeatedValue: Tensor<S>(_input[0].shape))
    }
    
    public override func apply() {
        if outputs["output"]!.count != _input.count {
            outputs["output"] = [Tensor<S>](count: _input.count, repeatedValue: Tensor<S>(_input[0].shape))
        }
        
        for out in outputs["output"]! {
            copy(from: _gradOutput, to: out)
        }
    }
    
    public override func reset() {
        for out in outputs["output"]! {
            fill(out, value: 0)
        }
    }
}

extension AddOp: Differentiable {
    public func gradient() -> GradientType {
        return AddOpGrad<S>(op: self)
    }
}

// multiplies together inputs
public class MulOp<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    public var _input:[Tensor<S>] {
        let _in:[InputType<S>] = inputs[0]
        return _in.map { $0.output() }
    }
    
    public init(_ ops:Op<S>...) {
        super.init(inputs: ["input"], outputs: ["output"])
        connect(from: ops, "output", to: self, "input")
        outputs["output"] = [Tensor<S>](count: _input.count, repeatedValue: zeros(_input[0].shape))
    }
    
    public override func apply() {
        if output.shape != _input[0].shape {
            output.resize(_input[0].shape)
        }
        
        copy(from: _input[0], to: output)
        for o in _input[1..<_input.count] {
            imul(output, o)
        }
    }
}

public class MulOpGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    public var _op:Tensor<S> { return inputs[0].output() }
    public var _input:[Tensor<S>] {
        let _in:[InputType<S>] = inputs[1]
        return _in.map { $0.output() }
    }
    public var _gradOutput:Tensor<S> { return inputs[2].output() }
    
    public required init(op:MulOp<S>) {
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        
        let opInputs:[InputType<S>] = op.inputs[0]
        connect(from: op, "output", to: self, "op")
        connect(from: opInputs.map { $0.op! }, "output", to: self, "input")
        outputs["output"] = [Tensor<S>](count: _input.count, repeatedValue: Tensor<S>(_input[0].shape))
    }
    
    public override func apply() {
        if outputs["output"]!.count != _input.count {
            outputs["output"] = [Tensor<S>](count: _input.count, repeatedValue: Tensor<S>(_input[0].shape))
        }
        
        // TODO: there is a more efficient way to implement this
        for (i, out) in outputs[0].enumerate() {
            copy(from: _gradOutput, to: out)
            
            for (j, output) in _input.enumerate() {
                if i != j {
                    imul(out, output)
                }
            }
        }
    }
    
    public override func reset() {
        for out in outputs["output"]! {
            fill(out, value: 0)
        }
    }
}

extension MulOp: Differentiable {
    public func gradient() -> GradientType {
        return MulOpGrad<S>(op: self)
    }
}

public class LogOp<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    var _input:Tensor<S> { return inputs[0].output() }
    
    public init(size:Int) {
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = Tensor<S>(Extent(size))
        
        setAction("input", action: self.inputSet)
    }
    
    public override init() {
        super.init(inputs: ["input"], outputs: ["output"])
        setAction("input", action: self.inputSet)
    }
    
    func inputSet(label:String, input:[Op<S>]) {
        output.resize(_input[0].shape)
    }
    
    public override func apply() {
        if output.shape != _input.shape {
            output.resize(_input.shape)
        }
        
        log(_input, result: output)
    }
}

extension LogOp: Differentiable {
    public func gradient() -> GradientType {
        return LogGrad<S>(op: self)
    }
}

public class LogGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    public var _input:Tensor<S> { return inputs[1].output() }
    public var _gradOutput:Tensor<S> { return inputs[2].output() }
    
    public required init(op:LogOp<S>) {
        let opInput:InputType<S> = op.inputs[0]
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        connect(from: op, "output", to: self, "op")
        connect(from: opInput.op!, "output", to: self, "input")
        outputs["output"] = Tensor<S>(op.output.shape)
    }
    
    public override func apply() {
        if output.shape != _gradOutput.shape {
            output.resize(_gradOutput.shape)
        }

        fill(output, value: 1)
        output /= _input
        output *= _gradOutput
    }
    
    public override func reset() {
        fill(output, value: 0)
    }
}

public class ConcatOp<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    public override init() {
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = Tensor<S>()
        setAction("input", action: self.inputSet)
    }
    
    public init(_ ops:[Op<S>]) {
        super.init(inputs: ["input"], outputs: ["output"])
        connect(from: ops, "output", to: self, "input")
        outputs["output"] = concat(ops.map { $0.output })
        
        setAction("input", action: self.inputSet)
    }
    
    public convenience init(_ ops:Op<S>...) {
        self.init(ops)
    }
    
    func inputSet(label:String, inputs:[Op<S>]) {
        // TODO: to make more efficient, calculate size of concat first
        // so no temporary is required
        let result = concat(inputs.map { $0.output })
        output.resize(result.shape)
        copy(from: result, to: output)
    }
    
    public override func apply() {
        let result = concat(inputs[0].map { $0.output() })
        output.resize(result.shape)
        copy(from: result, to: output)
    }
}

public class ConcatGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    var _op:Tensor<S> { return inputs[0].output() }
    var _input:[Tensor<S>] { return inputs[1].output() }
    public var _gradOutput:Tensor<S> { return inputs[2].output() }
    
    public required init(op:ConcatOp<S>) {
        let opInputs:[InputType<S>] = op.inputs["input"]!
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        connect(from: op, "output", to: self, "op")
        connect(from: opInputs.map { $0.op! }, "output", to: self, "input")
        
        // TODO: check this makes sense
        outputs["output"] = opInputs.map { Tensor<S>($0.op!.output.shape) }
    }
    
    public override func apply() {
        var index:Int = 0
        for output in outputs["output"]! {
            let first = index
            let last = output.shape[0]+index
            copy(from: _gradOutput[first..<last], to: output)
            
            index = last
        }
    }
    
    public override func reset() {
        fill(output, value: 0)
    }
}

extension ConcatOp: Differentiable {
    public func gradient() -> GradientType {
        return ConcatGrad<S>(op: self)
    }
}

public class ViewOp<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    var ranges:[TensorIndex]
    
    public init(input:Op<S>, ranges:[TensorIndex]) {
        self.ranges = ranges
        super.init(inputs: ["input"], outputs: ["output"])
        connect(from: input, "output", to: self, "input")
        outputs["output"] = input.outputs["output"]![ranges]
        
        setAction("input", action: self.inputSet)
    }
    
    public init(ranges:TensorIndex...) {
        self.ranges = ranges
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = Tensor<S>()
        
        setAction("input", action: self.inputSet)
    }
    
    func inputSet(label:String, inputs:[Op<S>]) {
        outputs[0] = inputs[0].outputs["output"]![ranges]
    }
    
    public override func apply() {}
}

public class ViewGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    var _op:Tensor<S> { return inputs[0].output() }
    var _input:Tensor<S> { return inputs[1].output() }
    var _gradOutput:Tensor<S> { return inputs[2].output() }
    
    public required init(op:ViewOp<S>) {
        let input:InputType<S> = op.inputs["input"]!
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        connect(from: op, "output", to: self, "op")
        connect(from: input.op!, "output", to: self, "input")
        
        let opOutput = _input[op.ranges]
        outputs["output"] = Tensor<S>(opOutput.shape)
    }
    
    public override func apply() {
        copy(from: _gradOutput, to: output)
    }
    
    public override func reset() {
        fill(output, value: 0)
    }
}

extension ViewOp: Differentiable {
    public func gradient() -> GradientType {
        return ViewGrad<S>(op: self)
    }
}
