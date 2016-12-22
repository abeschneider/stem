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
open class Conv2dOp<S:Storage>: Op<S> where S.ElementType:FloatNumericType {
    open var _input:Tensor<S> { return inputs[0].output() }
    open var kernels:Tensor<S>
    open var padding:[Int]
    open var stride:[Int]
    
    public init(input: Op<S>, numFilters:Int, filterSize:Extent, stride:[Int] = [1, 1], padding:[Int]=[1, 1]) {
        let size = Extent(numFilters, filterSize[0], filterSize[1])
        kernels = uniform(size)

        self.stride = stride
        self.padding = padding
        super.init(inputs: ["input"], outputs: ["output"])
        connect(from: input, "output", to: self, "input")
        
        outputs["output"] = Tensor<S>()
    }
    
    public init(numFilters:Int, filterSize:Extent, stride:[Int]=[1, 1], padding:[Int]=[1, 1]) {
        let size = Extent(numFilters, filterSize[0], filterSize[1])
        kernels = uniform(size)
        
        self.stride = stride
        self.padding = padding
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = Tensor<S>()
    }

    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    open override func apply() {
//        if output.shape != _input.shape {
//            output.resize(_input.shape)
//        }
        
        let kernel = kernels[0, all, all]
        let outputShape = calculateConv2DSize(input: _input, kernel: kernel, stride: stride, padding: padding)
//        let result = Tensor<S>(outputShape)
        output.resize(outputShape)
        
        // TODO: conv2d should take a Tensor to store results
//        let result = conv2d(_input, kernel: filter, padding: [1, 1])
        for i in 0..<kernels.shape[0] {
            let kernel = kernels[i, all, all]
            conv2d(_input, kernel: kernel, padding: [1, 1], addTo: output)
        }
//        copy(from: result, to: output)
    }
}

open class Conv2dGrad<S:Storage>: Op<S>, Gradient where S.ElementType:FloatNumericType {
    open var conv:Conv2dOp<S> {
        let input:InputType<S> = inputs[0]
        return input.op as! Conv2dOp<S>
    }
    
    open var kernels:Tensor<S>
    
    open var _input:Tensor<S> { return inputs[1].output() }
    open var _gradOutput:Tensor<S> { return inputs[2].output() }
    
    public required init(op:Conv2dOp<S>) {
        kernels = Tensor<S>(op.kernels.shape)
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
        
        for i in 0..<kernels.shape[0] {
            let result = conv2d(_gradOutput, kernel: conv.kernels[i, all, all], padding: [1, 1], flip: false)
            iadd(kernels[i, all, all], result)
            iadd(output, result)
        }
    }
    
    open override func reset() {
        for out in outputs["output"]! {
            fill(out, value: 0)
        }
        
        fill(kernels, value: 0)
    }
}

extension Conv2dOp: Differentiable {
    public func gradient() -> GradientType {
        return Conv2dGrad<S>(op: self)
    }
}
