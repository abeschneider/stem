//
//  mul.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/12/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation
import Tensor

open class MulOp<S:Storage>: Op<S> where S.ElementType:FloatNumericType {
    open var _input:[Tensor<S>] {
        let _in:[Source<S>] = inputs[0]
        return _in.map { $0.output }
    }
    
    public init(_ ops:Op<S>...) {
        super.init(inputs: ["input"], outputs: ["output"])
        connect(from: ops, "output", to: self, "input")
        outputs["output"] = [Tensor<S>](repeating: zeros(_input[0].shape), count: _input.count)
    }
    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    open override func apply() {
        if output.shape != _input[0].shape {
            output.resize(_input[0].shape)
        }
        
        copy(from: _input[0], to: output)
        for o in _input[1..<_input.count] {
            imul(output, o)
        }
    }
}

open class MulOpGrad<S:Storage>: Op<S>, Gradient where S.ElementType:FloatNumericType {
    open var _op:Tensor<S> { return inputs[0].output }
    open var _input:[Tensor<S>] {
        let _in:[Source<S>] = inputs[1]
        return _in.map { $0.output }
    }
    open var _gradOutput:Tensor<S> { return inputs[2].output }
    
    public required init(op:MulOp<S>) {
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
        
        // TODO: there is a more efficient way to implement this
        for (i, out) in outputs[0].enumerated() {
            copy(from: _gradOutput, to: out)
            
            for (j, output) in _input.enumerated() {
                if i != j {
                    imul(out, output)
                }
            }
        }
    }
    
    open override func reset() {
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
