//
//  l2loss.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/12/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation
import Tensor

open class L2Loss<S:Storage>: Op<S>, Loss where S.ElementType:FloatNumericType {
    public typealias StorageType = S
    
    open var value:S.ElementType = 0
    
    open var _input:Tensor<S> { return inputs[0].output }
    open var _target:Tensor<S> { return inputs[1].output }
    
    public init() {
        super.init(inputs: ["input", "target"], outputs: ["output"])
        outputs["output"] = [Tensor<S>()]
        setAction("input", action: inputSet)
    }
    
    public init(size:Int) {
        super.init(inputs: ["input", "target"], outputs: ["output"])
        outputs["output"] = [zeros(Extent(size))]
        setAction("input", action: inputSet)
    }
    
    public init(target t:Op<S>) {
        super.init(inputs: ["input", "target"], outputs: ["output"])
        connect(from: t, "output", to: self, "target")
        outputs["output"] = [zeros(t.output.shape)]
        setAction("input", action: inputSet)
    }
    
    public init(value:Op<S>, target t:Op<S>) {
        super.init(inputs: ["input", "target"], outputs: ["output"])
        connect(from: value, "output", to: self, "input")
        connect(from: t, "output", to: self, "target")
        outputs["output"] = [Tensor<S>(value.output.shape)]
        setAction("input", action: inputSet)
    }
    
    public required init(op:Op<S>, shared:Bool) {
        super.init(inputs: ["input", "target"], outputs: ["output"])
        outputs["output"] = [Tensor<S>(op.output.shape)]
        setAction("input", action: inputSet)
    }
    
    func inputSet(_ label:String, input:[Source<S>]) {
        setInput(to: input[0])
        output.resize(input[0].op!.output.shape)
    }
    
    open override func apply() {
        sub(_input, _target, result: output)
        pow(output, 2, result: output)
        value = sum(output)
    }
    
    open override func reset() {
        fill(output, value: 0)
        value = 0
    }
}

open class L2LossGrad<S:Storage>: Op<S>, Gradient where S.ElementType:FloatNumericType {
    public typealias OpType = L2Loss<S>
    
    open var linear:LinearOp<S> {
        let input:Source<S> = inputs[0]
        return input.op as! LinearOp<S>
    }
    open var _input:Tensor<S> { return inputs[1].output }
    open var _target:Tensor<S> { return inputs[2].output }
    
    public required init(op:L2Loss<S>) {
        let loss:Source<S> = op.inputs[0]
        let target:Source<S> = op.inputs[1]
        
        super.init(inputs: ["op", "input", "target"], outputs: ["output"])
        
        connect(from: op, "output", to: self, "op")
        connect(from: Source(op: loss.op!), to: Target(op: self, label: "input"))
        connect(from: Source(op: target.op!, label: target.label),
                to: Target(op: self, label: "target"))
        
        output = Tensor<S>(Extent(op._input.shape))
    }
    
    public init(size:Int) {
        super.init(inputs: ["op", "input", "target"], outputs: ["output"])
        outputs["output"] = [Tensor<S>(Extent(size))]
    }
    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    open override func apply() {
        sub(_input, _target, result: output)
        output *= 2
    }
    
    open override func reset() {
        fill(output, value: 0)
    }
}

extension L2Loss: Differentiable {
    public func gradient() -> GradientType {
        return L2LossGrad<S>(op: self)
    }
}
