//
//  add.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/12/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation
import Tensor

open class AddOp<S:Storage>: Op<S> where S.ElementType:FloatNumericType {
    open var inOps:[Op<S>] {
        let inputs:[Source<S>] = self.inputs[0]
        return inputs.map { $0.op! }
    }
    
    public init(_ ops:Op<S>...) {
        super.init(inputs: ["input"], outputs: ["output"])
        connect(from: ops, "output", to: self, "input")
        outputs["output"] = [Tensor<S>](repeating: zeros(ops[0].output.shape), count: ops.count)
    }
    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    open override func apply() {
        if output.shape != inOps[0].output.shape {
            output.resize(inOps[0].output.shape)
        }
        
        copy(from: inOps[0].output, to: output)
        for op in inOps[1..<inOps.count] {
            iadd(output, op.output)
        }
    }
}

open class AddOpGrad<S:Storage>: Op<S>, Gradient where S.ElementType:FloatNumericType {
    open var _add:Tensor<S> { return inputs[0].output }
    open var _input:[Tensor<S>] {
        let _in:[Source<S>] = inputs[1]
        return _in.map { $0.output }
    }
    open var _gradOutput:Tensor<S> { return inputs[2].output }
    
    public required init(op:AddOp<S>) {
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        
        let opInputs:[Source<S>] = op.inputs[0]
        connect(from: op, "output", to: self, "op")
        connect(from: opInputs.map { $0.op! }, "output", to: self, "input")
        outputs["output"] = [Tensor<S>](repeating: Tensor<S>(_input[0].shape), count: _input.count)
    }
    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    open override func apply() {
        if outputs["output"]!.count != _input.count {
            outputs["output"] = [Tensor<S>](repeating: Tensor<S>(_input[0].shape), count: _input.count)
        }
        
        for out in outputs["output"]! {
            copy(from: _gradOutput, to: out)
        }
    }
    
    open override func reset() {
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
