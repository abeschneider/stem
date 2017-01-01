//
//  concat.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/12/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation
import Tensor

open class ConcatOp<S:Storage>: Op<S> where S.ElementType:FloatNumericType {
    public init() {
        super.init(inputs: ["input"], outputs: ["output"])
        outputs["output"] = [Tensor<S>()]
        setAction("input", action: self.inputSet)
    }
    
    public init(_ ops:[Op<S>]) {
        super.init(inputs: ["input"], outputs: ["output"])
        connect(from: ops, "output", to: self, "input")
        outputs["output"] = [concat(ops.map { $0.output })]
        
        setAction("input", action: self.inputSet)
    }
    
    public convenience init(_ ops:Op<S>...) {
        self.init(ops)
    }
    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    func inputSet(_ label:String, input:[Source<S>]) {
        // TODO: to make more efficient, calculate size of concat first
        // so no temporary is required
//        let result = concat(input[0].outputs.map { $0 })
        let result = concat(input.map { $0.output })
        output.resize(result.shape)
        copy(from: result, to: output)
    }
    
    open override func apply() {
        let result = concat(inputs[0].map { $0.output() })
        output.resize(result.shape)
        copy(from: result, to: output)
    }
}

open class ConcatGrad<S:Storage>: Op<S>, Gradient where S.ElementType:FloatNumericType {
    var _op:Tensor<S> { return inputs[0].output() }
    var _input:[Tensor<S>] { return inputs[1].outputs() }
    open var _gradOutput:Tensor<S> { return inputs[2].output() }
    
    public required init(op:ConcatOp<S>) {
        let opInputs:[InputType<S>] = op.inputs["input"]!
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        connect(from: op, "output", to: self, "op")
        connect(from: opInputs.map { $0.op! }, "output", to: self, "input")
        
        // TODO: check this makes sense
        outputs["output"] = opInputs.map { Tensor<S>($0.op!.output.shape) }
    }
    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    open override func apply() {
        var index:Int = 0
        for output in outputs["output"]! {
            let first = index
            let last = output.shape[0]+index
            copy(from: _gradOutput[first..<last], to: output)            
            index = last
        }
    }
    
    open override func reset() {
        fill(output, value: 0)
    }
}

extension ConcatOp: Differentiable {
    public func gradient() -> GradientType {
        return ConcatGrad<S>(op: self)
    }
}
