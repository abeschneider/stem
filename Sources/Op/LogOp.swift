//
//  log.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/12/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation
import Tensor

open class LogOp<S:Storage>: Op<S> where S.ElementType:FloatNumericType {
    var _input:Tensor<S> { return inputs[0].output() }
    
    public init(size:Int) {
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = [Tensor<S>(Extent(size))]
        
        setAction("input", action: self.inputSet)
    }
    
    public init() {
        super.init(inputs: ["input"], outputs: ["output"])
        setAction("input", action: self.inputSet)
    }
    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    func inputSet(_ label:String, input:[Op<S>]) {
        output.resize(_input[0].shape)
    }
    
    open override func apply() {
        if output.shape != _input.shape {
            output.resize(_input.shape)
        }
        
        log(_input, result: output)
    }
}

extension LogOp: Differentiable {
    public func gradient() -> GradientType {
        return LogGrad<S>(op: self)
    }
}

open class LogGrad<S:Storage>: Op<S>, Gradient where S.ElementType:FloatNumericType {
    open var _input:Tensor<S> { return inputs[1].output() }
    open var _gradOutput:Tensor<S> { return inputs[2].output() }
    
    public required init(op:LogOp<S>) {
        let opInput:InputType<S> = op.inputs[0]
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        connect(from: op, "output", to: self, "op")
        connect(from: opInput.op!, "output", to: self, "input")
        outputs["output"] = [Tensor<S>(op.output.shape)]
    }
    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    open override func apply() {
        if output.shape != _gradOutput.shape {
            output.resize(_gradOutput.shape)
        }
        
        fill(output, value: 1)
        output /= _input
        output *= _gradOutput
    }
    
    open override func reset() {
        fill(output, value: 0)
    }
}
