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
    open var _input:Tensor<S> { return inputs[0].output }
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
        setAction("input", action: self.inputSet)
    }
    
    func inputSet(_ label:String, input:[Source<S>]) {
        let shape = input[0].output.shape
        
        var depth:Int
        var width:Int
        var height:Int
        
        setInput(to: input[0])
        
        // TODO: currently following Torch conventions
        if shape.count == 2 {
            depth = 1
            width = (shape[0] - poolingSize[0])/stride[0] + 1
            height = (shape[1] - poolingSize[1])/stride[1] + 1
        } else {
            depth = shape[0]
            width = (shape[1] - poolingSize[0])/stride[0] + 1
            height = (shape[2] - poolingSize[1])/stride[1] + 1
        }
        
        let reducedShape = Extent(depth, width, height)
                                  
        _output.resize(reducedShape)
        indices.resize(Extent(2, reducedShape[0], reducedShape[1], reducedShape[2]))
    }
    
    // required for copying
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }

    // TODO: figure out a more compact way to write this that allows
    // the same for-loop to be used for differently dimensioned inputs
    open override func apply() {
        if _input.shape.count == 2 {
            let d = 0
            let width = _output.shape[1]
            let height = _output.shape[2]

            for i in 0..<width {
                for j in 0..<height {
                    let row_start = i*stride[0]
                    let row_end = row_start + stride[0]
                    let col_start = j*stride[1]
                    let col_end = col_start + stride[1]
                    let view = _input[row_start..<row_end, col_start..<col_end]
                    let (index, value) = evalFn(view)
                    indices[0, d, i, j] = index[0]
                    indices[1, d, i, j] = index[1]
                    _output[d, i, j] = value
                }
            }
        } else {
            let depth = _output.shape[0]
            let width = _output.shape[1]
            let height = _output.shape[2]
            
            for d in 0..<depth {
                for i in 0..<width {
                    for j in 0..<height {
                        let row_start = i*stride[0]
                        let row_end = row_start + stride[0]
                        let col_start = j*stride[1]
                        let col_end = col_start + stride[1]
                        let view = _input[d, row_start..<row_end, col_start..<col_end]
                        let (index, value) = evalFn(view)
                        indices[0, d, i, j] = index[0]
                        indices[1, d, i, j] = index[1]
                        _output[d, i, j] = value
                    }
                }
            }
        }
    }
}

open class PoolingGrad<S:Storage>: Op<S>, Gradient where S.ElementType:FloatNumericType {
    open var pooling:PoolingOp<S> {
        let input:Source<S> = inputs[0]
        return input.op as! PoolingOp<S>
    }

    open var _input:Tensor<S> { return inputs[1].output }
    open var _gradOutput:Tensor<S> { return inputs[2].output }

    public required init(op:PoolingOp<S>) {
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        
        let opInputs:[Source<S>] = op.inputs[0]
        connect(from: op, to: self, "op")
        connect(from: opInputs.map { $0.op! }, to: self, "input")
        output = Tensor<S>(_input.shape)
        
//        setAction("input", action: self.inputSet)
    }
    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    func inputSet(_ label:String, input:[Source<S>]) {
        setInput(inputLabel: "gradOutput", to: input[0])
    }
    
    open override func apply() {
        var depth:Int
        var width:Int
        var height:Int
        
        if _input.shape.count == 2 {
            depth = 1
            width = _input.shape[0] / pooling.stride[0]
            height = _input.shape[1] / pooling.stride[1]
        } else {
            depth = _input.shape[0]
            width = _input.shape[1] / pooling.stride[0]
            height = _input.shape[2] / pooling.stride[1]
        }
        
        fill(output, value: 0)
        for d in 0..<depth {
            for i in 0..<width {
                for j in 0..<height {
                    let ii = i*pooling.stride[0] + pooling.indices[0, d, i, j]
                    let jj = j*pooling.stride[1] + pooling.indices[1, d, i, j]
                    output[ii, jj] = _gradOutput[i, j]
                }
            }
        }
    }
}

extension PoolingOp: Differentiable {
    public func gradient() -> GradientType {
        return PoolingGrad<S>(op: self)
    }
}
