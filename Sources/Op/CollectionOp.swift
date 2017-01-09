//
//  CollectionOp.swift
//  stem
//
//  Created by Abraham Schneider on 12/26/16.
//
//

import Foundation
import Tensor

open class CollectionOp<S:Storage>: Op<S>, Sequence {
    open var ops:[Op<S>]
    open var ordering:AnyOrdering<S>
    
    open var count:Int { return ops.count }
    
    public init<T:Ordering>(
        ops:[Op<S>],
        inputs:[Op<S>],
        outputs:[Op<S>],
        ordering:T) where T.StorageType==S
    {
        self.ordering = AnyOrdering<S>(ordering)
        self.ops = ops
        super.init(inputs: ["input"], outputs: ["output"])
        
        if inputs.count > 0 {
            connect(from: inputs, "output", to: self, "input")
        }
        
        self.outputs["output"] = outputs.map { $0.output }
    }
    
    // required for Copyable
    public required init(op:Op<S>, shared:Bool) {
        let cop = op as! CollectionOp<S>
        ordering = AnyOrdering<S>(cop.ordering)
        ops = cop.ops.map { copy(op: $0, shared: shared) }
        
        super.init(inputs: ["input"], outputs: ["outputs"])
        outputs["output"] = [Tensor<S>(op.output.shape)]
    }

    open override func apply() {
        for op in ordering.traversal(ops) {
            op.apply()
        }
    }
    
    open func makeIterator() -> AnyIterator<Op<S>> {
        return ordering.traversal(ops).makeIterator()
    }
    
    open override func reset() {
        fill(output, value: 0)
    }
    
    open override var description: String {
        let className = String(describing: Mirror(reflecting: self).subjectType)
        var inputDescription:String
        
        if let inputOps:[Source<S>] = inputs["input"] {
            inputDescription = inputOps.map {
                if let op = $0.op {
                    let output:Tensor<S> = $0.output
                    return String(describing: output.shape.dims)
                } else {
                    return "[]"
                }
            }.joined(separator: ", ")
        } else {
            inputDescription = "<empty>"
        }
        
        let value = ops.map { String(describing: $0) }.joined(separator: "\n\t")
        return "<\(className): inputs:\(inputDescription), outputs:\(output.shape.dims)> {\n\t\(value)\n}"
    }
    
    open override func params() -> [Tensor<S>] {
        return ops.reduce([]) {
            $0 + $1.params()
        }
    }
}

open class CollectionGradient<S:Storage>: Op<S>, Gradient {
    public typealias StorageType = S
    open var ops:[Op<S>] = []
    var ordering:AnyOrdering<S>
    
    open var _input:Tensor<S> { return inputs[1].output }
    open var _gradOutput:Tensor<S> { return inputs[2].output }
    
    public required init(op:CollectionOp<S>) {
        ordering = op.ordering
        super.init(inputs: ["op", "inputs", "gradOutput"], outputs: ["output"])
        
        outputs["output"] = [Tensor<S>()]
        
        // start with outputs, and work backwards
        createBackwardsGraph(op.ops)
        //        output = ops.last!.output
        
        setAction("gradOutput", action: self.gradOutputSet)
    }
    
    public init<T:Ordering>(
        ops:[Op<S>],
        inputs:[Op<S>],
        outputs:[Op<S>],
        ordering:T) where T.StorageType==S
    {
        self.ordering = AnyOrdering<S>(ordering)
        self.ops = ops
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["outputs"])
        //        connect(from: outputs, "output", to: self, "gradOutput")
        
        self.outputs["output"] = outputs.map { $0.output }
        setAction("gradOutput", action: self.gradOutputSet)
    }
    
    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    func createBackwardsGraph(_ ops:[Op<S>]) {
        // convert all ops -> opGradient
        var gradOps:[Int:Op<S>] = [:]
        
        for op in ops.reversed() {
            if let diffOp = op as? Differentiable {
                let gradOp = (diffOp.gradient() as! Op<S>)
                gradOps[op.id] = gradOp
                self.ops.append(gradOp)
            }
        }
        
        // connect all (op1 -> op2) to (op2Gradient -> op1Gradient)
        for op in ops {
            if let opInputs:[Source<S>] = op.inputs["input"] {
                for input in opInputs {
                    if  let fromOp = gradOps[op.id], let toOp = gradOps[input.op!.id] {
                        connect(from: fromOp, to: toOp, "gradOutput")
                    }
                }
            }
        }
    }
    
    func gradOutputSet(_ key:String, value:[Source<S>]) {
//        connect(from: value, "output", to: ops.first!, "gradOutput")
        setInput(inputLabel: "gradOutput", to: value)
        let sourceOps:[Op<S>] = value.map { $0.op! }
        connect(from: sourceOps, "output", to: ops.first!, "gradOutput")
//        let target = Target<S>(op: op, label: "gradOutput")
//        connect(from: value, to)
//        connect(from: value, to: Target<S>(op: [op], label: "gradOutput"))
    }
    
    open subscript(index:Int) -> Op<S> {
        return ops[index]
    }
    
    open override func apply() {
        var lastOp:Op<S>?
        for op in ordering.traversal(ops) {
            op.apply()
            lastOp = op
        }
        
        if let op:Op<S> = lastOp {
            // TODO: make this a function (part of copy?)
            // Also: is it necessary to copy, or should we
            // just point?
            if output.shape != op.output.shape {
                output.resize(op.output.shape)
            }
            
            copy(from: op.output, to: output)
        }
    }
    
    open override func reset() {
        ops.forEach { $0.reset() }
        fill(output, value: 0)
    }
    
    open override var description: String {
        let className = String(describing: Mirror(reflecting: self).subjectType)
        
        let value = ops.map { String(describing: $0) }.joined(separator: "\n\t")
        return "<\(className): inputs=?, outputs=\(output.shape.dims)> {\n\t\(value)\n}"
    }
    
    open override func params() -> [Tensor<S>] {
        return ops.reversed().reduce([]) {
            $0 + $1.params()
        }
    }
}

extension CollectionOp:Differentiable {
    public func gradient() -> GradientType {
        return CollectionGradient<S>(op: self)
    }
}
