//
//  variable.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/12/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation
import Tensor

open class VariableOp<S:Storage>: Op<S> {
    open var _input:Tensor<S> { return inputs[0].output }
    
    public init() {
        super.init(inputs: ["input"], outputs: ["output"])
        setAction("input", action: inputSet)
    }
    
    public init(_ size:Extent) {
        super.init(inputs: ["input"], outputs: ["output"])
        setAction("input", action: inputSet)
        
//        let constant = ConstantOp<S>(size)
//        connect(from: constant, to: self, "input")
        output.resize(size)
    }
    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    func inputSet(_ label:String, input:[Source<S>]) {
        setInput(to: input[0])
        output = Tensor<S>(input[0].output.shape)
    }
    
    open override func apply() {
        copy(from: _input, to: output)
    }
}

open class VariableGrad<S:Storage>: Op<S>, Gradient {
    open var _variable:Tensor<S> { return inputs[0].output }
    open var _input:Tensor<S> { return inputs[1].output }
    open var _gradOutput:Tensor<S> { return inputs[2].output }
    
    
    public required init(op:VariableOp<S>) {
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        
        let v:Source<S> = op.inputs[0]
        connect(from: op, "output", to: self, "op")
        connect(from: v.op!, "output", to: self, "input")
        outputs["output"] = [Tensor<S>(op.output.shape)]
    }
    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    open override func apply() {
        copy(from: _gradOutput, to: output)
    }
    
    open override func reset() {
        fill(output, value: 0)
    }
}

extension VariableOp: Differentiable {
    public func gradient() -> GradientType {
        return VariableGrad<S>(op: self)
    }
}
