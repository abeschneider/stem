//
//  ClassNLLLoss.swift
//  stem
//
//  Created by Abraham Schneider on 1/4/17.
//
//

import Foundation
import Tensor

open class ClassNLLLoss<S:Storage>: Op<S>, Loss where S.ElementType:FloatNumericType {
    public typealias StorageType = S
    
    open var value:S.ElementType = 0
    
    open var softmax:LogSoftMaxOp<S> = LogSoftMaxOp<S>()
    
    open var _input:Tensor<S> { return inputs[0].output }
    open var _target:Tensor<S> { return inputs[1].output }
    
    public init() {
        super.init(inputs: ["input", "target"], outputs: ["output"])
        output = zeros(Extent(1))
//        setAction("input", action: self.inputSet)
    }
    
    public required init(op:Op<S>, shared:Bool) {
        super.init(inputs: ["input", "target"], outputs: ["output"])
        output = zeros(Extent(1))
//        setAction("input", action: self.inputSet)
    }

    open override func apply() {
        // NB: Technically this should be sum(_input*_target). However, under
        // the condition is 1 for a single class and 0 for all the others, this
        // is an equivalent value.
        let t = S.ElementType.trunc(_target[0])
        output[0] = -_input[t]
        value = -_input[t]
    }
}

open class ClassNLLLossGrad<S:Storage>: Op<S>, Gradient where S.ElementType:FloatNumericType {
    public typealias OpType = ClassNLLLoss<S>
    
    open var _input:Tensor<S> { return inputs[1].output }
    open var _target:Tensor<S> { return inputs[2].output }
    
    public required init(op:ClassNLLLoss<S>) {
        let loss:Source<S> = op.inputs[0]
        let target:Source<S> = op.inputs[1]
        
        super.init(inputs: ["op", "input", "target"], outputs: ["output"])
        connect(from: op, "output", to: self, "op")
        connect(from: Source(op: loss.op!), to: Target(op: self, label: "input"))
        
        connect(from: Source(op: target.op!, label: target.label), to: Target(op: self, label: "target"))
        output = Tensor<S>(Extent(op._input.shape))
    }
    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    open override func apply() {
        let t = S.ElementType.trunc(_target[0])
        output[t] = -1
    }
    
    open override func reset() {
        fill(output, value: 0)
    }
}

extension ClassNLLLoss: Differentiable {
    public func gradient() -> GradientType {
        return ClassNLLLossGrad<S>(op: self)
    }
}

