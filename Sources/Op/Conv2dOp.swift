//
//  Conv2dOp.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/13/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation
import Tensor


// TODO:
// 1. add more parameters (e.g. stride, padding, etc.)
// 2. allow more than one channel to be specified
open class Conv2dOp<S:Storage>: Op<S> where S.ElementType:FloatNumericType {
    open var _input:Tensor<S> { return inputs[0].output() }
    open var filter:Tensor<S>
    
    public init(input: Op<S>, filterSize:Extent) {
        filter = uniform(filterSize)

        super.init(inputs: ["input"], outputs: ["output"])
        connect(from: input, "output", to: self, "input")
        
        outputs["output"] = Tensor<S>()
    }
    
    public init(filterSize:Extent) {
        filter = uniform(filterSize)
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = Tensor<S>()
    }

    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    open override func apply() {
        if output.shape != _input.shape {
            output.resize(_input.shape)
        }
        
        // TODO: conv2d should take a Tensor to store results
        let result = conv2d(_input, kernel: filter, padding: [1, 1])
        copy(from: result, to: output)
    }
}

open class Conv2dGrad<S:Storage>: Op<S>, Gradient where S.ElementType:FloatNumericType {
    open var conv:Conv2dOp<S> {
        let input:InputType<S> = inputs[0]
        return input.op as! Conv2dOp<S>
    }
    
    open var filter:Tensor<S>
    
    open var _input:Tensor<S> { return inputs[1].output() }
    open var _gradOutput:Tensor<S> { return inputs[2].output() }
    
    public required init(op:Conv2dOp<S>) {
        filter = Tensor<S>(op.filter.shape)
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        
        let opInputs:[InputType<S>] = op.inputs[0]
        connect(from: op, "output", to: self, "op")
        connect(from: opInputs.map { $0.op! }, "output", to: self, "input")
        outputs["output"] = [Tensor<S>()]
    }
    
    // required for copying
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    open override func apply() {
        if output.shape != _input.shape {
            output.resize(_input.shape)
        }
        
        // TODO: conv2d should take a Tensor to store results
        let result = conv2d(_gradOutput, kernel: conv.filter, padding: [1, 1], flip: false)
        iadd(filter, result)
        iadd(output, result)
    }
    
    open override func reset() {
        for out in outputs["output"]! {
            fill(out, value: 0)
        }
        
        fill(filter, value: 0)
    }
}

extension Conv2dOp: Differentiable {
    public func gradient() -> GradientType {
        return Conv2dGrad<S>(op: self)
    }
}
