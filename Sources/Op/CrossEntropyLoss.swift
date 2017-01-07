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
    open var nll:ClassNLLLoss<S> = ClassNLLLoss<S>()
    
    open var _input:Tensor<S> { return inputs[0].output }
    open var _target:Tensor<S> { return inputs[1].output }
    
    public init() {
        super.init(inputs: ["input", "target"], outputs: ["output"])
        output = zeros(Extent(1))
        setAction("input", action: self.inputSet)
        setAction("target", action: self.targetSet)
    }
    
    public required init(op:Op<S>, shared:Bool) {
        fatalError("init(op:shared:) has not been implemented")
//        super.init(inputs: ["input", "target"], outputs: ["output"])
//        output = zeros(Extent(1))
//        setAction("input", action: self.inputSet)
    }
    
    func inputSet(_ label:String, input:[Source<S>]) {
        // input -> softmax
        connect(from: input[0], to: Target<S>(op: softmax))
        
        // softmax -> nll
        connect(from: softmax, to: nll)
        
        setInput(inputLabel: "input", to: input[0])
    }
    
    func targetSet(_ label:String, target:[Source<S>]) {
        setInput(inputLabel: "target", to: target[0])
        connect(from: target[0], to: Target(op: nll, label: "target"))
    }
    
    open override func apply() {
        softmax.apply()
        nll.apply()
        value = nll.value
    }
}

open class CrossEntropyLossGrad<S:Storage>: Op<S>, Gradient where S.ElementType:FloatNumericType {
    public typealias OpType = CrossEntropyLoss<S>
    
    open var _input:Tensor<S> { return inputs[1].output }
    open var _target:Tensor<S> { return inputs[2].output }
    
    open var softmaxGrad:LogSoftMaxGrad<S>
    open var nllGrad:ClassNLLLossGrad<S>
    
    public required init(op:CrossEntropyLoss<S>) {
        let loss:Source<S> = op.inputs[0]
        let target:Source<S> = op.inputs[1]
        
        softmaxGrad = op.softmax.gradient() as! LogSoftMaxGrad<S>
        nllGrad = op.nll.gradient() as! ClassNLLLossGrad<S>
        
        super.init(inputs: ["op", "input", "target"], outputs: ["output"])
        
        connect(from: op, "output", to: self, "op")
        connect(from: Source(op: loss.op!), to: Target(op: self, label: "input"))
        connect(from: Source(op: target.op!, label: target.label),
                to: Target(op: self, label: "target"))
        
        output = Tensor<S>(Extent(op._input.shape))
        
        // TODO: double check why this is necessary .. shouldn't .gradient() already do this?
        connect(from: nllGrad, to: softmaxGrad, "gradOutput")
    }
    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    open override func apply() {
        nllGrad.apply()
        softmaxGrad.apply()
        output = softmaxGrad.output
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
