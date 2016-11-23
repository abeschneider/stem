//
//  sigmoid.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/12/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation
import Tensor

open class SigmoidOp<S:Storage>: Op<S> where S.ElementType:FloatNumericType {
    var _input:Tensor<S> { return inputs[0].output() }
    
    public init() {
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = [Tensor<S>()]
        setAction("input", action: self.inputSet)
    }
    
    public init(size:Int) {
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = [Tensor<S>(Extent(size))]
        
        setAction("input", action: self.inputSet)
    }
    
    // required for Copyable
    public required init(op:Op<S>, shared:Bool) {
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = [Tensor<S>(op.output.shape)]
    }
    
    func inputSet(_ label:String, input:[Op<S>]) {
        output.resize(_input.shape)
    }
    
    open override func apply() {
        sigmoid(_input, output: output)
    }
}

open class SigmoidGrad<S:Storage>: Op<S>, Gradient where S.ElementType:FloatNumericType {
    public typealias OpType = SigmoidOp<S>
    
    open var _sigmoid:Tensor<S> { return inputs[0].output() }
    open var _input:Tensor<S> { return inputs[1].output() }
    open var _gradOutput:Tensor<S> { return inputs[2].output() }
    
    public required init(op:SigmoidOp<S>) {
        let s:InputType<S> = op.inputs[0]
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        connect(from: op, "output", to: self, "op")
        connect(from: s.op!, "output", to: self, "input")
        outputs["output"] = [Tensor<S>(op.output.shape)]
    }
    
    public init(size:Int) {
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        outputs["output"] = [Tensor<S>(Extent(size))]
        
    }
    
    public init(op:SigmoidOp<S>, input:Op<S>, gradInput:Op<S>) {
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        connect(from: op, "output", to: self, "op")
        connect(from: input, "output", to: self, "input")
        connect(from: gradInput, "output", to: self, "gradOutput")
        outputs["output"] = [Tensor<S>(input.output.shape)]
    }
    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    open override func apply() {
        // sigmoid(x)*(1 - sigmoid(x))
        let result = S.ElementType(1.0) - _sigmoid
        result *= _sigmoid
        result *= _gradOutput
        output += result
    }
    
    open override func reset() {
        fill(output, value: 0)
    }
}

extension SigmoidOp: Differentiable {
    public func gradient() -> GradientType {
        return SigmoidGrad<S>(op: self)
    }
}
