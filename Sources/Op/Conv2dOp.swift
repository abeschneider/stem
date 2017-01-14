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
    open var _input:Tensor<S> { return inputs[0].output }
    open var kernels:Tensor<S>
    open var padding:[Int]
    open var stride:[Int]
    
    open var _numOutputs:Int { return kernels.shape[0] }
    open var _numInputs:Int { return kernels.shape[1] }
    
    public init(input: Op<S>, numInputs:Int, numOutputs:Int, kernelSize:Extent, stride:[Int] = [1, 1], padding:[Int]=[1, 1]) {
        let size = Extent(numOutputs, numInputs, kernelSize[0], kernelSize[1])
        
        // pretty sure kernels should be stored in unfolded state
        kernels = uniform(size)

        self.stride = stride
        self.padding = padding
        super.init(inputs: ["input"], outputs: ["output"])
        
        outputs["output"] = Tensor<S>()
        setAction("input", action: self.inputSet)
        
        connect(from: input, "output", to: self, "input")
    }
    
    public init(numInputs:Int, numOutputs:Int, kernelSize:Extent, stride:[Int]=[1, 1], padding:[Int]=[1, 1]) {
        let size = Extent(numOutputs, numInputs, kernelSize[0], kernelSize[1])
        kernels = uniform(size)
        
        self.stride = stride
        self.padding = padding
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = Tensor<S>()
        setAction("input", action: self.inputSet)
    }
    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    func inputSet(_ label:String, input:[Source<S>]) {
        // Currently follows Torch convention
        var shape:Extent
        
        setInput(to: input[0])
        let convShape = calculateConv2DSize(input: input[0].output[0, all, all],
                                            kernel: kernels[0, 0, all, all],
                                            stride: stride,
                                            padding: padding)
        
        output.resize(Extent(_numOutputs, convShape[0], convShape[1]))
    }
    
    /*open override func apply() {
        fill(output, value: 0)
        
        if _input.shape.count == 2 {
            for i in 0..<kernels.shape[0] {
                let kernel = kernels[i, all, all]
                conv2d(_input, kernel: kernel, padding: [1, 1], addTo: output)
            }
        } else {
            for b in 0..<_input.shape[0] {
                for i in 0..<kernels.shape[0] {
                    let kernel = kernels[i, all, all]
                    conv2d(_input[b, all, all], kernel: kernel, padding: [1, 1], addTo: output[b, all, all])
                }
            }
        }
    }*/
    
    
    open override func apply() {
        let unfoldedKernel = unroll(kernels: kernels)
        let unfoldedInput = unroll(tensor: _input, kernelShape: Extent(kernels.shape[2], kernels.shape[3]))
        let result = Tensor<S>(Extent(unfoldedInput.shape[0], unfoldedKernel.shape[1]))
        dot(unfoldedInput, unfoldedKernel, result: result)
        
        for i in 0..<_numOutputs {
            output[i, all, all] = result[all, i].reshape(Extent(output.shape[1], output.shape[2]))
        }
    }
    
    open override func params() -> [Tensor<S>] { return [ravel(kernels)] }
}

open class Conv2dGrad<S:Storage>: Op<S>, Gradient where S.ElementType:FloatNumericType {
    open var conv:Conv2dOp<S> {
        let input:Source<S> = inputs[0]
        return input.op as! Conv2dOp<S>
    }
    
    open var kernels:Tensor<S>
    
    open var _input:Tensor<S> { return inputs[1].output }
    open var _gradOutput:Tensor<S> { return inputs[2].output }
    
    public required init(op:Conv2dOp<S>) {
        kernels = Tensor<S>(op.kernels.shape)
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        
        let opInputs:[Source<S>] = op.inputs[0]
        connect(from: op, "output", to: self, "op")
        connect(from: opInputs.map { $0.op! }, "output", to: self, "input")
        output = Tensor<S>(_input.shape)
    }
    
    // required for copying
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    /*open override func apply() {
        fill(output, value: 0)
        
        if _input.shape.count == 2 {
            for i in 0..<kernels.shape[0] {
                let kernel = kernels[i, all, all]
                conv2d(_input, kernel: kernel, padding: [1, 1], addTo: output)
            }
        } else {
            for batch in 0..<_input.shape[0] {
                for i in 0..<kernels.shape[0] {
                    let kernel = kernels[i, all, all]
                    conv2d(_input[batch, all, all], kernel: kernel, padding: [1, 1], addTo: output)
                }
            }
        }
    }*/

    open override func apply() {
        if _input.shape.count == 2 {
            for i in 0..<kernels.shape[0] {
                let grad_wrt_kernel = conv2d(_gradOutput, kernel: _input, padding: [1, 1], flip: false)
                iadd(kernels[i, all, all], grad_wrt_kernel)
                
                let grad_wrt_output = conv2d(_gradOutput, kernel: conv.kernels[i, all, all], padding: [1, 1], flip: false)
                iadd(output, grad_wrt_output)
            }
        } else {
            for b in 0..<_input.shape[0] {
                for i in 0..<kernels.shape[0] {
                    let grad_wrt_kernel = conv2d(_gradOutput[b, all, all], kernel: _input[b, all, all], padding: [1, 1], flip: false)
                    iadd(kernels[i, all, all], grad_wrt_kernel)
                    
                    let grad_wrt_output = conv2d(_gradOutput[b, all, all], kernel: conv.kernels[i, all, all], padding: [1, 1], flip: false)
                    iadd(output[b, all, all], grad_wrt_output)
                }
            }
        }
    }
    
    open override func reset() {
        fill(kernels, value: 0)
        fill(output, value: 0)
    }
    
    open override func params() -> [Tensor<S>] {
        return [ravel(kernels)]
    }
}

extension Conv2dOp: Differentiable {
    public func gradient() -> GradientType {
        return Conv2dGrad<S>(op: self)
    }
}
