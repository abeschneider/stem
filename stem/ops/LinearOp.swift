//
//  linear.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/12/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation

open class LinearOp<S:Storage>: Op<S> where S.ElementType:FloatNumericType {
    //    public typealias StorageType = S
    public typealias OpType = LinearOp<S>
    
    open var weight:Tensor<S>
    open var bias:Tensor<S>
    
    open var _input:Tensor<S> { return inputs[0].output() }
    
    public init() {
        weight = Tensor<S>()
        bias = Tensor<S>()
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = [Tensor<S>()]
        
        setAction("input", action: self.inputSet)
    }
    
    public init(outputSize:Int) {
        weight = Tensor<S>()
        bias = zeros(Extent(outputSize))
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = [zeros(Extent(outputSize))]
        
        setAction("input", action: self.inputSet)
    }
    
    public init(_ input:Op<S>, outputSize:Int) {
        let inputSize = input.output.shape[0]
        weight = uniform(Extent(outputSize, inputSize))
        bias = zeros(Extent(outputSize))
        super.init(inputs: ["input"], outputs: ["output"])
        connect(from: input, "output", to: self, "input")
        outputs["output"] = [zeros(Extent(outputSize))]
        
        setAction("input", action: self.inputSet)
    }
    
    public init(inputSize:Int, outputSize:Int) {
        weight = uniform(Extent(outputSize, inputSize))
        bias = zeros(Extent(outputSize))
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = [zeros(Extent(outputSize))]
        
        setAction("input", action: self.inputSet)
    }
    
    public init(weight:Tensor<S>, bias:Tensor<S>) {
        self.weight = weight
        self.bias = bias
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = [zeros(bias.shape)]
        
        setAction("input", action: self.inputSet)
    }
    
    // required for Copyable
    public required init(op:Op<S>, shared:Bool) {
        let linear = op as! LinearOp<S>
        weight = shared ? linear.weight : copy(linear.weight)
        bias = shared ? linear.bias : copy(linear.bias)
        super.init(inputs: ["inputs"], outputs: ["output"])
        outputs["output"] = [zeros(bias.shape)]
        
        setAction("input", action: self.inputSet)
    }
    
    func inputSet(_ label:String, input:[Op<S>]) {
        let newShape = Extent(output.shape[0], input[0].output.shape[0])
        if weight.shape != newShape {
            weight.resize(newShape)
            weight.uniform()
        }
    }
    
    open override func apply() {
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
    
    open override func params() -> [Tensor<S>] {
        return [weight, bias]
    }
}

open class LinearGrad<S:Storage>: Op<S>, Gradient where S.ElementType:FloatNumericType {
    //    public typealias StorageType = S
    
    open var weight:Tensor<S>
    open var bias:Tensor<S>
    
    open var linear:LinearOp<S> {
        let input:InputType<S> = inputs[0]
        return input.op as! LinearOp<S>
    }
    
    open var _input:Tensor<S> { return inputs[1].output() }
    open var _gradOutput:Tensor<S> { return inputs[2].output() }
    
    public required init(op:LinearOp<S>) {
        weight = zeros(op.weight.shape)
        bias = zeros(op.bias.shape)
        
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        connect(from: op, "output", to: self, "op")
        connect(from: op.inputs[0].op!, "output", to: self, "input")
        outputs["output"] = [zeros(Extent(op.weight.shape[1]))]
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
        outputs["output"] = [zeros(Extent(op.weight.shape[1]))]
    }
    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    // need to separate apply into accumulate and apply
    open override func apply() {
        print("linear: \(_gradOutput)")
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
    
    open override func params() -> [Tensor<S>] {
        return [weight, bias]
    }
    
    open override func reset() {
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
