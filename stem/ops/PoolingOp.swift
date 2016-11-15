//
//  PoolingOp.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/14/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation

func max<S:Storage>(input:Tensor<S>) -> Int {
    let flattenedInput = ravel(input)
    var best = S.ElementType(10000)
    var bestIndex:Int = 0
    for index in flattenedInput.indices() {
        let value = input[index]
        if value > best {
            best = value
            bestIndex = index[0]
        }
    }
    
    return bestIndex
}

func min<S:Storage>(input:Tensor<S>) -> Int {
    let flattenedInput = ravel(input)
    var best = S.ElementType(-10000)
    var bestIndex:Int = 0
    for index in flattenedInput.indices() {
        let value = input[index]
        if value < best {
            best = value
            bestIndex = index[0]
        }
    }
    
    return bestIndex
}

open class PoolingOp<S:Storage>: Op<S> where S.ElementType:FloatNumericType {
    open var _input:Tensor<S> { return inputs[0].output() }
    
    var poolingSize:Extent
    var stride:Extent
    var evalFn:(_ input:Tensor<S>) -> Int
    var indices:Tensor<NativeStorage<Int>>

    public init(input: Op<S>,
                poolingSize:Extent,
                stride:Extent,
                evalFn:@escaping (_ input:Tensor<S>) -> Int)
    {
        self.poolingSize = poolingSize
        self.stride = stride
        self.evalFn = evalFn
        
        indices = Tensor<NativeStorage<Int>>()
        super.init(inputs: ["input"], outputs: ["output"])
        connect(from: input, "output", to: self, "input")
        
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
            indices.resize(Extent(width, height))
        }
        
        for i in 0..<width {
            for j in 0..<height {
                let row_start = i*stride[0]
                let row_end = row_start + stride[0]
                let col_start = j*stride[1]
                let col_end = col_start + stride[1]
                let view = _input[row_start..<row_end, col_start..<col_end]
                indices[i, j] = evalFn(view)
            }
        }
    }
}
