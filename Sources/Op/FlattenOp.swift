//
//  FlattenOp.swift
//  stem
//
//  Created by Abraham Schneider on 12/24/16.
//
//

import Foundation
import Tensor

open class FlattenOp<S:Storage>: Op<S> {
    open var _input:Tensor<S> { return inputs[0].output }

    public init(_ op:Op<S>) {
        super.init(inputs: ["input"], outputs: ["output"])
        connect(from: [op], to: self, "input")
        outputs["output"] = Tensor<S>(Extent(op.output.shape.elements))
        setAction("input", action: self.inputSet)
    }
    
    public init() {
        super.init(inputs: ["input"], outputs: ["output"])
//        connect(from: [op], to: self, "input")
        outputs["output"] = Tensor<S>()
        setAction("input", action: self.inputSet)
    }
    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    func inputSet(_ label:String, input:[Source<S>]) {
        setInput(to: input[0])
        let inputShape = input[0].output.shape
        output.resize(Extent(inputShape.elements))
    }
    
    open override func apply() {
//        copy(from: _input, to: output)
        var k = 0
        for i in _input.indices() {
            output[k] = _input[i]
            k += 1
        }
    }
}

open class FlattenOpGrad<S:Storage>: Op<S>, Gradient {
    open var _flatten:Tensor<S> { return inputs[0].output }
    open var _input:[Tensor<S>] {
        let _in:[Source<S>] = inputs[1]
        return _in.map { $0.output }
    }
    open var _gradOutput:Tensor<S> { return inputs[2].output }
    
    public required init(op:FlattenOp<S>) {
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        
        let opInputs:[Source<S>] = op.inputs[0]
        connect(from: op, "output", to: self, "op")
        connect(from: opInputs.map { $0.op! }, "output", to: self, "input")
        output = Tensor<S>(_input[0].shape)
    }
    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    open override func apply() {
//        copy(from: _gradOutput, to: output)
        var k = 0
        for i in output.indices() {
            output[i] = _gradOutput[k]
            k += 1
        }
    }
    
    open override func reset() {
        fill(output, value: 0)
    }
}

extension FlattenOp:Differentiable {
    public func gradient() -> GradientType {
        return FlattenOpGrad<S>(op: self)
    }
}
