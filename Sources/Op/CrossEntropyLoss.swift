//
//  CrossEntropyLoss.swift
//  stem
//
//  Created by Abraham Schneider on 1/1/17.
//
//

import Foundation
import Tensor

open class CrossEntropyLoss<S:Storage>: Op<S>, Loss where S.ElementType:FloatNumericType {
    public typealias StorageType = S
    
    open var value:S.ElementType = 0
    
    open var softmax:LogSoftMaxOp<S> = LogSoftMaxOp<S>()
    
    open var _input:Tensor<S> { return inputs[0].output }
    open var _target:Tensor<S> { return inputs[1].output }
    
    public init() {
        super.init(inputs: ["input", "target"], outputs: ["output"])
        output = Tensor<S>()
        setAction("input", action: self.inputSet)
    }
    
    public init(size:Int) {
        super.init(inputs: ["input", "target"], outputs: ["output"])
        outputs["output"] = [zeros(Extent(size))]
        setAction("input", action: self.inputSet)
    }
    
    public init(target t:Op<S>) {
        super.init(inputs: ["input", "target"], outputs: ["output"])
        connect(from: t, "output", to: self, "target")
        output = zeros(t.output.shape)
        setAction("input", action: self.inputSet)
    }
    
    public init(value:Op<S>, target t:Op<S>) {
        super.init(inputs: ["input", "target"], outputs: ["output"])
        connect(from: value, "output", to: self, "input")
        connect(from: t, "output", to: self, "target")
        output = Tensor<S>(value.output.shape)
        setAction("input", action: self.inputSet)
    }
    
    public required init(op:Op<S>, shared:Bool) {
        super.init(inputs: ["input", "target"], outputs: ["output"])
        output = Tensor<S>(op.output.shape)
        setAction("input", action: self.inputSet)
    }
    
    func inputSet(_ label:String, input:[Source<S>]) {
        // connect input to softmax
        // (use softmax output as input)
    }
    
    open override func apply() {
        // TODO: change to setAction
        if output.shape != _input.shape {
            output.resize(_input.shape)
        }
        
//        sub(_input, _target, result: output)
//        pow(output, 2, result: output)
//        value = sum(output)
    }
}

open class CrossEntropyLossGrad<S:Storage>: Op<S>, Gradient where S.ElementType:FloatNumericType {
    public typealias OpType = CrossEntropyLoss<S>
    
    open var _input:Tensor<S> { return inputs[1].output }
    open var _target:Tensor<S> { return inputs[2].output }
    
    public required init(op:CrossEntropyLoss<S>) {
        let loss:Source<S> = op.inputs[0]
        let target:Source<S> = op.inputs[1]
        
        super.init(inputs: ["op", "input", "target"], outputs: ["output"])
        connect(from: op, "output", to: self, "op")
        connect(Source(op: loss.op!), Target(op: self, label: "input"))
        
        connect(Source(op: target.op!, label: target.label), Target(op: self, label: "target"))
        outputs["output"] = [Tensor<S>(Extent(op._input.shape))]
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

extension CrossEntropyLoss: Differentiable {
    public func gradient() -> GradientType {
        return CrossEntropyLossGrad<S>(op: self)
    }
}
