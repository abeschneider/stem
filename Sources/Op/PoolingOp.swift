//
//  PoolingOp.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/14/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation
import Tensor

public func max<S:Storage>(input:Tensor<S>) -> ([Int], S.ElementType) where S.ElementType:NumericType {
    var best = -S.ElementType.infinity
    var bestIndex:[Int] = [0, 0]
    for index in input.indices() {
        let value = input[index]
        if value > best {
            best = value
            bestIndex = index
        }
    }
    
    return (bestIndex, best)
}

public func min<S:Storage>(input:Tensor<S>) -> ([Int], S.ElementType) where S.ElementType:NumericType {
    var best = S.ElementType.infinity
    var bestIndex:[Int] = [0, 0]
    for index in input.indices() {
        let value = input[index]
        if value < best {
            best = value
            bestIndex = index
        }
    }
    
    return (bestIndex, best)
}

open class PoolingOp<S:Storage>: Op<S> where S.ElementType:FloatNumericType {
    open var _input:Tensor<S> { return inputs[0].output() }
    open var _output:Tensor<S> { return outputs[0] }
    
    var poolingSize:Extent
    var stride:Extent
    var evalFn:(_ input:Tensor<S>) -> ([Int], S.ElementType)
    var indices:Tensor<NativeStorage<Int>>

    public init(poolingSize:Extent,
                stride:Extent,
                evalFn:@escaping (_ input:Tensor<S>) -> ([Int], S.ElementType))
    {
        self.poolingSize = poolingSize
        self.stride = stride
        self.evalFn = evalFn
        
        indices = Tensor<NativeStorage<Int>>()
        super.init(inputs: ["input"], outputs: ["output"])
        
        outputs["output"] = Tensor<S>()
    }
    
    // required for copying
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }

    open override func apply() {
        let width = _input.shape[0] / stride[0]
        let height = _input.shape[0] / stride[1]
        
        if indices.shape != Extent(width, height) {
            indices.resize(Extent(2, width, height))
            _output.resize(Extent(width, height))
        }
        
        for i in 0..<width {
            for j in 0..<height {
                let row_start = i*stride[0]
                let row_end = row_start + stride[0]
                let col_start = j*stride[1]
                let col_end = col_start + stride[1]
                let view = _input[row_start..<row_end, col_start..<col_end]
                let (index, value) = evalFn(view)
                indices[0, i, j] = index[0]
                indices[1, i, j] = index[1]
                _output[i, j] = value
            }
        }
    }
}

open class PoolingGrad<S:Storage>: Op<S>, Gradient where S.ElementType:FloatNumericType {
    open var pooling:PoolingOp<S> {
        let input:InputType<S> = inputs[0]
        return input.op as! PoolingOp<S>
    }

    open var _input:Tensor<S> { return inputs[1].output() }
    open var _gradOutput:Tensor<S> { return inputs[2].output() }

    public required init(op:PoolingOp<S>) {
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        
        let opInputs:[InputType<S>] = op.inputs[0]
        connect(from: op, "output", to: self, "op")
        connect(from: opInputs.map { $0.op! }, "output", to: self, "input")
        outputs["output"] = [Tensor<S>()]
    }
    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    open override func apply() {
        if output.shape != _input.shape {
            output.resize(_input.shape)
        }
        
        let width = _input.shape[0] / pooling.stride[0]
        let height = _input.shape[0] / pooling.stride[1]
        
        fill(output, value: 0)
        for i in 0..<width {
            for j in 0..<height {
                output[i*pooling.stride[0] + pooling.indices[0, i, j], j*pooling.stride[1] + pooling.indices[1, i, j]] = _gradOutput[i, j]
            }
        }
    }
}

extension PoolingOp: Differentiable {
    public func gradient() -> GradientType {
        return PoolingGrad<S>(op: self)
    }
}
