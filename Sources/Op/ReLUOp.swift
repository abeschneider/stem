//
//  ReLUOp.swift
//  stem
//
//  Created by Abraham Schneider on 12/2/16.
//
//

import Foundation
import Tensor

open class ReLUOp<S:Storage>: Op<S> where S.ElementType:FloatNumericType {
    var _input:Tensor<S> { return inputs[0].output() }
    
    public init() {
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = [Tensor<S>()]
        setAction("input", action: self.inputSet)
    }
    
    public init(size:Int) {
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = [Tensor<S>(Extent(size))]
        
        setAction("input", action: self.inputSet)
    }
    
    // required for Copyable
    public required init(op:Op<S>, shared:Bool) {
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = [Tensor<S>(op.output.shape)]
    }
    
    func inputSet(_ label:String, input:[Source<S>]) {
        output.resize(input[0].output.shape)
    }
    
    open override func apply() {
        for (i, j) in zip(_input.indices(), output.indices()) {
            output[j] = max(0, _input[i])
        }
    }
}

open class ReLUGrad<S:Storage>: Op<S>, Gradient where S.ElementType:FloatNumericType {
    public typealias OpType = ReLUOp<S>
    
    open var _relu:Tensor<S> { return inputs[0].output() }
    open var _input:Tensor<S> { return inputs[1].output() }
    open var _gradOutput:Tensor<S> { return inputs[2].output() }
    
    public required init(op:ReLUOp<S>) {
        let s:InputType<S> = op.inputs[0]
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        connect(from: op, "output", to: self, "op")
        connect(from: s.op!, "output", to: self, "input")
        outputs["output"] = [Tensor<S>(op.output.shape)]
    }
    
    public init(size:Int) {
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        outputs["output"] = [Tensor<S>(Extent(size))]
        
    }
    
    public init(op:ReLUOp<S>, input:Op<S>, gradInput:Op<S>) {
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        connect(from: op, "output", to: self, "op")
        connect(from: input, "output", to: self, "input")
        connect(from: gradInput, "output", to: self, "gradOutput")
        outputs["output"] = [Tensor<S>(input.output.shape)]
    }
    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    open override func apply() {
        for (i, j) in zip(_input.indices(), output.indices()) {
            if _input[i] >= 0 {
                output[j] += S.ElementType(0)
            }
        }
    }
    
    open override func reset() {
        fill(output, value: 0)
    }
}

extension ReLUOp: Differentiable {
    public func gradient() -> GradientType {
        return ReLUGrad<S>(op: self)
    }
}
