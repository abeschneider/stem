//
//  PoolingOp.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/14/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation

open class PoolingOp<S:Storage>: Op<S> where S.ElementType:FloatNumericType {
    open var _input:Tensor<S> { return inputs[0].output() }
    
    var poolingSize:Extent
    var stride:Extent
    var initial:S.ElementType
    var evalFn:(_ a:S.ElementType, _ b:S.ElementType) -> Bool

    public init(input: Op<S>,
                poolingSize:Extent,
                stride:Extent,
                initial:S.ElementType,
                evalFn:@escaping (_ a:S.ElementType, _ b:S.ElementType) -> Bool)
    {
        self.poolingSize = poolingSize
        self.stride = stride
        self.initial = initial
        self.evalFn = evalFn
        
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
        var best = initial
        
        for i in 0..<width {
            for j in 0..<height {
                for k in 0..<poolingSize[0] {
                    for l in 0..<poolingSize[1] {
                        let v = _input[i+k, j+l]
                        if evalFn(v, best) {
                            best = v
                        }
                    }
                }
            }
        }
    }
}
