//
//  ops.swift
//  stem
//
//  Created by Schneider, Abraham R. on 5/25/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation
import Tensor

/*
open class DataSequence<S:Storage>: IteratorProtocol {
    var data:[Tensor<S>] = []
    var index:Int = 0
    
    open func next() -> Tensor<S>? {
        if index < data.count {
            return data[index]
        }
        
        return nil
    }
}

// TODO: remove?
//public class VariableSequence<S:Storage>: Op<S> {
//    var i:Int = 0
//    var vars:[Tensor<S>]
//    public var count:Int { return vars.count }
//    
//    public init(_ vars:[Tensor<S>]) {
//        self.vars = vars
//        super.init(inputs: [], outputs: ["output"])
//        outputs["output"] = self.vars[0]
//    }
//    
//    public required init(op:Op<S>) {
//        let vl = op as! VariableSequence<S>
//        self.vars = vl.vars
//        super.init(inputs: [], outputs: ["output"])
//        outputs["output"] = vl.vars[0]
//    }
//    
//    public override func apply() {
//        defer {
//            i = (i + 1) % vars.count
//        }
//
//        output = vars[i]
//    }
//    
//    public subscript(index:Int) -> Tensor<S> {
//        return vars[index]
//    }
//    
//    public override func reset() {
//        i = 0
//    }
//}

open class IdentityOp<S:Storage>: Op<S> {
    var _input:Tensor<S> { return inputs[0].output() }

    public init() {
        super.init(inputs: ["input"], outputs: ["output"])
    }
    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    open override func apply() {
        // is it necessary to do the multiplication, or can we just forward the input?
        copy(from: _input, to: outputs["output"]![0])
    }
}

extension IdentityOp: Differentiable {
    public func gradient() -> GradientType {
        return IdentityGrad<S>(op: self)
    }
}

open class IdentityGrad<S:Storage>: Op<S>, Gradient {
//    public typealias StorageType = S

    open var _gradOutput:Tensor<S> { return inputs[2].output() }

    public required init(op:IdentityOp<S>) {
        let s:InputType<S> = op.inputs[0]
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["output"])
        connect(from: op, "output", to: self, "op")
        connect(from: s.op!, "output", to: self, "input")
        outputs["output"] = [Tensor<S>(op.output.shape)]
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
*/
