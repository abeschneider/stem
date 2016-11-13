//
//  view.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/12/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation

open class ViewOp<S:Storage>: Op<S> where S.ElementType:FloatNumericType {
    var ranges:[TensorIndex]
    
    public init(input:Op<S>, ranges:[TensorIndex]) {
        self.ranges = ranges
        super.init(inputs: ["input"], outputs: ["output"])
        connect(from: input, "output", to: self, "input")
        outputs["output"] = [input.outputs["output"]![0][ranges]]
        
        setAction("input", action: self.inputSet)
    }
    
    public init(ranges:TensorIndex...) {
        self.ranges = ranges
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = [Tensor<S>()]
        
        setAction("input", action: self.inputSet)
    }
    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    func inputSet(_ label:String, inputs:[Op<S>]) {
        outputs[0] = [inputs[0].outputs["output"]![0][ranges]]
    }
    
    open override func apply() {}
}

open class ViewGrad<S:Storage>: Op<S>, Gradient where S.ElementType:FloatNumericType {
    //    public typealias StorageType = S
    
    var _op:Tensor<S> { return inputs[0].output() }
    var _input:Tensor<S> { return inputs[1].output() }
    var _gradOutput:Tensor<S> { return inputs[2].output() }
    
    public required init(op:ViewOp<S>) {
        let input:InputType<S> = op.inputs["input"]![0]
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        connect(from: op, "output", to: self, "op")
        connect(from: input.op!, "output", to: self, "input")
        
        let opOutput = _input[op.ranges]
        outputs["output"] = Tensor<S>(opOutput.shape)
    }
    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    open override func apply() {
        copy(from: _gradOutput, to: output)
    }
    
    open override func reset() {
        fill(output, value: 0)
    }
}

extension ViewOp: Differentiable {
    public func gradient() -> GradientType {
        return ViewGrad<S>(op: self)
    }
}
