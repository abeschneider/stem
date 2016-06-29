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
    public typealias InputAction = (String, [Op<S>]) -> ()
    
    public var id:Int = createUID()
    
    // TODO: consider allowing the Op to be Optional and get rid of NoOp.
    // Q: make `inputs` tensors and store Ops separately? Or store Ops at all?
    // Can easily make `outputs` an OrderedDictionary if both are same type.
    // If that is done, can't ever allow an Op to change where `output` points to
    // (which is okay).
    public var inputs:OrderedDictionary<Op<S>>
    
//    public var output:Tensor<StorageType>
    public var outputs:OrderedDictionary<Tensor<S>>
    
    // convenience variable for case when there is only a single output
    public var output:Tensor<S> {
        get { return outputs["output"]! }
        set { outputs["output"] = newValue }
//        get { return outputs["output"]![0] }
//        set { outputs["output"]![0] = newValue }
    }
    
    public var inputActions:[String:InputAction] = [:]
    
    required public init(actions:[String:InputAction]=[:]) {
        self.inputs = OrderedDictionary<Op<S>>()
        self.outputs = OrderedDictionary<Tensor<S>>()
        inputActions = actions
    }
    
    required public init(inputs:[(String, [Op<S>])], outputs:[(String, [Tensor<S>])], actions:[String:InputAction]=[:]) {
        self.inputs = OrderedDictionary<Op<S>>(inputs)
        self.outputs = OrderedDictionary<Tensor<S>>(outputs)
        inputActions = actions
    }
    
    required public init(outputs:[Tensor<S>], actions:[String:InputAction]=[:]) {
        self.inputs = OrderedDictionary<Op<S>>()
        
        self.outputs = OrderedDictionary<Tensor<S>>()
        self.outputs["output"] = outputs
        inputActions = actions
    }
    
    required public init(inputs:[(String, Op<S>)], outputs:[(String, Tensor<S>)], actions:[String:InputAction]=[:]) {
        self.inputs = OrderedDictionary<Op<S>>(inputs)
        self.outputs = OrderedDictionary<Tensor<S>>(outputs)
        inputActions = actions
    }
    
    required public init(inputs:[(String, Op<S>)], outputs:OrderedDictionary<Tensor<S>>, actions:[String:InputAction]=[:]) {
        self.inputs = OrderedDictionary<Op<S>>(inputs)
        self.outputs = outputs
        inputActions = actions
    }
    
//    required public init(inputs:[(String, [Tensor<S>])], outputs:[Tensor<S>], actions:[String:InputAction]=[:]) {
//        self.inputs = OrderedDictionary<S>(inputs)
//        self.outputs = OrderedDictionary<S>()
//        self.outputs["output"] = outputs
//        inputActions = actions
//    }
    
    required public init(inputs:OrderedDictionary<Op<S>>, outputs:OrderedDictionary<Tensor<S>>, actions:[String:InputAction]=[:]) {
        self.inputs = inputs
        self.outputs = outputs
        inputActions = actions
    }

    
    required public init(op:Op<S>, shared:Bool) {
        self.inputs = OrderedDictionary<Op<S>>()
        self.outputs = OrderedDictionary<Tensor<S>>()
    }
    
    public func setInput(label:String, to:Op<S>) {
        inputs[label] = to
        
        if let action = inputActions[label] {
            action(label, [to])
        }
    }

    public func setInput(label:String, to:[Op<S>]) {
        inputs[label] = to
        
        if let action = inputActions[label] {
            action(label, to)
        }
    }
    
    public func setOutput(label:String, to:Tensor<S>) {
        outputs[label] = to
    }
    
    public func setOutput(label:String, to:[Tensor<S>]) {
        outputs[label] = to
    }
    
    public func setAction(key:String, action:(String, [Op<S>]) -> ()) {
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
//        output = copy(input)
        copy(from: input, to: output)
    }
    
    public override func apply() {}
}

public class Sigmoid<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    var input:Tensor<S> { return inputs[0].output }
    
    public init() {
        super.init(inputs: [("input", NoOp<S>())],
                   outputs: [("output", Tensor<S>())])
        
        setAction("input", action: self.inputSet)
    }
    
    public init(size:Int) {
        super.init(inputs: [("input", NoOp<S>())],
                   outputs: [("output", Tensor<S>(Extent(size)))])
        
        setAction("input", action: self.inputSet)
    }
    
    // required for Copyable
    public required init(op:Op<S>, shared:Bool) {
        super.init(inputs: [("input", NoOp<S>())],
                   outputs: [("output", Tensor<S>(op.output.shape))])
    }
    
    func inputSet(label:String, input:[Op<S>]) {
        output.resize(input[0].output.shape)
    }
    
    public override func apply() {
        sigmoid(input, output: output)
    }
}

public class SigmoidGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    public typealias StorageType = S
    public typealias OpType = Sigmoid<S>
    
    public var sigmoid:Tensor<S> {
//        let op:Op<S> = inputs[0]!
//        return op as! Sigmoid<S>
        return inputs[0].output
    }
    
    public var input:Tensor<S> { return inputs[1].output }
    public var gradOutput:Tensor<S> { return inputs[2].output }
    
    public required init(op:Sigmoid<S>) {
        let s:Op<S> = op.inputs[0]
        super.init(inputs: [("op", op), ("input", s), ("gradOutput", NoOp<S>())],
                   outputs: [("output", Tensor<S>(Extent(op.output.shape)))])
    }

    public init(size:Int) {
        super.init(inputs: [("op", NoOp<S>()), ("input", NoOp<S>()), ("gradOutput", NoOp<S>())],
                   outputs: [("output", Tensor<S>(Extent(size)))])
    }
    
    public init(op:Sigmoid<S>, input:Op<S>, gradInput:Op<S>) {
        super.init(inputs: [("op", op), ("input", input), ("gradOutput", gradInput)],
                   outputs: [("output", Tensor<S>(input.output.shape))])
    }
    
    public override func apply() {
        // sigmoid(x)*(1 - sigmoid(x))
        let result = S.ElementType(1.0) - sigmoid
        result *= sigmoid
        result *= gradOutput
        output += result
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

public class Tanh<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    var input:Tensor<S> { return inputs[0].output }
    
    public init() {
        super.init(inputs: [("input", NoOp<S>())],
                   outputs: [("output", Tensor<S>())])
        
        setAction("input", action: self.inputSet)
    }
    
    public init(size:Int) {
        super.init(inputs: [("input", NoOp<S>())],
                   outputs: [("output", Tensor<S>(Extent(size)))])
        
        setAction("input", action: self.inputSet)
    }
    
    // required for Copyable
    public required init(op:Op<S>, shared:Bool) {
        super.init(inputs: [("input", NoOp<S>())],
                   outputs: [("output", Tensor<S>(op.output.shape))])
    }
    
    func inputSet(label:String, input:[Op<S>]) {
        output.resize(input[0].output.shape)
    }
    
    public override func apply() {
        tanh(input, output: output)
    }
}

public class TanhGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    public typealias StorageType = S
    public typealias OpType = Tanh<S>
    
    public var tanh:Tensor<S> { return inputs[0].output }
    
    public var input:Tensor<S> { return inputs[1].output }
    public var gradOutput:Tensor<S> { return inputs[2].output }
    
    public required init(op:Tanh<S>) {
        let s:Op<S> = op.inputs[0]
        super.init(inputs: [("op", op), ("input", s), ("gradOutput", NoOp<S>())],
                   outputs: [("output", Tensor<S>(Extent(op.output.shape)))])
    }
    
    public init(size:Int) {
        super.init(inputs: [("op", NoOp<S>()), ("input", NoOp<S>()), ("gradOutput", NoOp<S>())],
                   outputs: [("output", Tensor<S>(Extent(size)))])
    }
    
    public init(op:Sigmoid<S>, input:Op<S>, gradInput:Op<S>) {
        super.init(inputs: [("op", op), ("input", input), ("gradOutput", gradInput)],
                   outputs: [("output", Tensor<S>(input.output.shape))])
    }
    
    /*
     dtanh(x)/dx = (1 - tanh^2 x)*dx
    */
    public override func apply() {
        let result = tanh * tanh
        result *= -1
        add(S.ElementType(1.0), result, result: result)
        result *= gradOutput
        output += result
    }
    
    public func reset() {
        fill(output, value: 0)
    }
}

extension Tanh:Differentiable {
    public func gradient() -> GradientType {
        return TanhGrad<S>(op: self)
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
        super.init(inputs: [("input", NoOp<S>())],
                   outputs: [("output", zeros(Extent(outputSize)))])
        
        setAction("input", action: self.inputSet)
    }
    
    public init(_ input:Op<S>, outputSize:Int) {
        let inputSize = input.output.shape[0]
        weight = uniform(Extent(outputSize, inputSize))
        bias = zeros(Extent(outputSize))
        super.init(inputs: [("input", input)],
                   outputs: [("output", zeros(Extent(outputSize)))])
        
        setAction("input", action: self.inputSet)
    }

    
    public init(inputSize:Int, outputSize:Int) {
        weight = uniform(Extent(outputSize, inputSize))
        bias = zeros(Extent(outputSize))
        super.init(inputs: [("input", NoOp<S>())],
                   outputs: [("output", zeros(Extent(outputSize)))])
        
        setAction("input", action: self.inputSet)
    }
    
    public init(weight:Tensor<S>, bias:Tensor<S>) {
        self.weight = weight
        self.bias = bias
        super.init(inputs: [("input", NoOp<S>())],
                   outputs: [("output", zeros(bias.shape))])
        
        setAction("input", action: self.inputSet)
    }
    
    public required init(op:Op<S>, shared:Bool) {
        let linear = op as! Linear<S>
        weight = shared ? linear.weight : copy(linear.weight)
        bias = shared ? linear.bias : copy(linear.bias)
        super.init(inputs: [("input", NoOp<S>())],
                   outputs: [("output", zeros(bias.shape))])
        
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
        if input.dims == 1 {
            output.resize(Extent(weight.shape[0]))
            dot(weight, input, result: output)
            add(output, bias, result: output)
        } else if input.dims == 2 {
            output.resize(Extent(weight.shape[0], input.shape[1]))
            dot(weight, input, result: output)
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

//    public var linear:Linear<S>
    public var weight:Tensor<S>
    public var bias:Tensor<S>
    
    public var linear:Linear<S> {
        let op:Op<S> = inputs[0]
        return op as! Linear<S>
    }
//    public var op:Tensor<S> { return inputs[0].output }
    public var input:Tensor<S> { return inputs[1].output }
    public var gradOutput:Tensor<S> { return inputs[2].output }
    
    public required init(op:Linear<S>) {
        weight = zeros(op.weight.shape)
        bias = zeros(op.bias.shape)
        super.init(inputs: [("op", op),
                            ("input", op.inputs[0]),
                            ("gradOutput", NoOp<S>())],
                   outputs: [("output", zeros(Extent(op.weight.shape[1])))])
    }

//    public init(inputSize:Int, outputSize:Int) {
//        weight = zeros(Extent(outputSize, inputSize))
//        bias = zeros(Extent(outputSize))
//        super.init(inputs: [("op", Tensor<S>()), ("input", Tensor<S>()), ("gradOutput", Tensor<S>())],
//                   outputs: [zeros(Extent(outputSize))])
//    }
    
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
        
//        self.linear = op
        super.init(inputs: [("op", op), ("input", input), ("gradOutput", gradInput)],
                   outputs: [("output", zeros(Extent(op.weight.shape[1])))])
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
        super.init(inputs: [("input", NoOp<S>()), ("target", NoOp<S>())],
                   outputs: [("output", Tensor<S>())])
    }
    
    public init(size:Int) {
        super.init(inputs: [("input", NoOp<S>()), ("target", NoOp<S>())],
                   outputs: [("output", zeros(Extent(size)))])
    }
    
    public init(target t:Op<S>) {
        super.init(inputs: [("input", NoOp<S>()), ("target", t)],
                   outputs: [("output", zeros(t.output.shape))])
    }
    
    public init(value:Op<S>, target t:Op<S>) {
        super.init(inputs: [("input", value), ("target", t)],
                   outputs: [("output", Tensor<S>(value.output.shape))])
    }
    
    public required init(op:Op<S>, shared:Bool) {
        super.init(inputs: [("input", NoOp<S>()), ("target", NoOp<S>())],
                   outputs: [("output", Tensor<S>(op.output.shape))])
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
    
    public var linear:Linear<S> {
        let op:Op<S> = inputs[0]
        return op as! Linear<S>
    }
    public var input:Tensor<S> { return inputs[1].output }
    public var target:Tensor<S> { return inputs[2].output }
    
    public required init(op:L2Loss<S>) {
        let l:Op<S> = op.inputs[0]
        let t:Op<S> = op.inputs[1]
        super.init(inputs: [("op", op), ("input", l), ("target", t)],
                   outputs: [("output", Tensor<S>(Extent(op.target.shape)))])
    }

    public init(size:Int) {
        super.init(inputs: [("op", NoOp<S>()), ("input", NoOp<S>()), ("target", NoOp<S>())],
                   outputs: [("output", Tensor<S>(Extent(size)))])
    }
    
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

// adds together inputs
public class AddOp<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    public var inOps:[Op<S>] { return inputs[0] }
    
    public init(_ ops:Op<S>...) {
        let inputs = OrderedDictionary<Op<S>>([("input", ops)])
        let outputs = OrderedDictionary<Tensor<S>>([("output", [Tensor<S>](count: ops.count, repeatedValue: zeros(ops[0].output.shape)))])
        super.init(inputs: inputs, outputs: outputs)
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
    public var inOps:[Op<S>] { return inputs[1] }
    public var gradOutput:Tensor<S> { return inputs[2].output }
    
    public required init(op:AddOp<S>) {
        super.init()
        
        let opInputs:[Op<S>] = op.inputs[0]
        setInput("op", to: op)
        setInput("input", to: opInputs)
        setInput("gradOutput", to: NoOp<S>())
        
        let out:[Tensor<S>] = [Tensor<S>](count: opInputs.count, repeatedValue: Tensor<S>(opInputs[0].output.shape))
        setOutput("output", to: out)
    }
    
    public override func apply() {
        if outputs["output"]!.count != inOps.count {
            outputs["output"] = [Tensor<S>](count: inOps.count, repeatedValue: Tensor<S>(inOps[0].output.shape))
        }
        
        for out in outputs["output"]! {
            copy(from: gradOutput, to: out)
        }
    }
    
    public func reset() {
        for out in outputs["output"]! {
            fill(out, value: 0)
        }
    }
}

extension AddOp:Differentiable {
    public func gradient() -> GradientType {
        return AddOpGrad<S>(op: self)
    }
}

public class Log<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    var input:Tensor<S> { return inputs[0].output }
    
    public init(size:Int) {
        super.init(inputs: [("input", NoOp<S>())],
                   outputs: [("output", Tensor<S>(Extent(size)))])
        
        setAction("input", action: self.inputSet)
    }
    
    public init() {
        super.init(inputs: [("input", NoOp<S>())],
                   outputs: [("output", Tensor<S>())])
        
        setAction("input", action: self.inputSet)
    }
    
    func inputSet(label:String, input:[Op<S>]) {
        output.resize(input[0].output.shape)
    }
    
    public override func apply() {
        if output.shape != input.shape {
            output.resize(input.shape)
        }
        
        log(input, result: output)
    }
}

extension Log:Differentiable {
    public func gradient() -> GradientType {
        return LogGrad<S>(op: self)
    }
}

public class LogGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    public var input:Tensor<S> { return inputs[1].output }
    public var gradOutput:Tensor<S> { return inputs[2].output }
    
    public required init(op:Log<S>) {
        let opInput:Op<S> = op.inputs[0]
        super.init(inputs: [("op", op), ("input", opInput), ("gradOutput", NoOp<S>())],
                   outputs: [("output", Tensor<S>(op.output.shape))])
    }
    
    public override func apply() {
        if output.shape != gradOutput.shape {
            output.resize(gradOutput.shape)
        }

        fill(output, value: 1)
        output /= input
        output *= gradOutput
    }
    
    public func reset() {
        fill(output, value: 0)
    }
}

public class Concat<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
//    var inOps:[Op<S>] { return inputs[0] }
    
    public init() {
        super.init(outputs: [Tensor<S>()])
        setAction("input", action: self.inputSet)
    }
    
    public init(_ ops:[Op<S>]) {
        let inputs = OrderedDictionary<Op<S>>([("input", ops)])
        let outputs = OrderedDictionary<Tensor<S>>([("output", concat(ops.map { $0.output }))])
        super.init(inputs: inputs, outputs: outputs)
        
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
        let result = concat(inputs[0].map { $0.output })
        output.resize(result.shape)
        copy(from: result, to: output)
    }
}

public class ConcatGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    var gradOutput:Tensor<S> { return inputs[2].output }
    
    public required init(op:Concat<S>) {
        let opInputs:[Op<S>] = op.inputs[0]
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
        for output in outputs["output"]! {
            let first = index
//            let last = output.shape[0]+index
            let last = output.shape[0]+index
            copy(from: gradOutput[first..<last], to: output)
            
//            switch output {
//            case .TensorValue(let value):
//                last = value.shape[0]+index
//                copy(from: gradOutput[first..<last], to: value)
//            case .TensorArray:
//                assertionFailure()
//            }
            
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

