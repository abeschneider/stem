//
//  graph3.swift
//  stem
//
//  Created by Schneider, Abraham R. on 4/10/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation


@inline(never)
internal func _abstract(_ file:StaticString = #file, line:UInt=#line) -> Never  {
    fatalError("Method must be overridden", file: file, line: line)
}

public protocol Ordering {
    associatedtype StorageType:Storage
    
//    func generate() -> AnyGenerator<Op<StorageType>>
    func traversal(_ ops:[Op<StorageType>]) -> AnySequence<Op<StorageType>>
}

open class AnyOrdering<S:Storage>: Ordering {
    public typealias StorageType = S
    
    public required init<M:Ordering>(_ base:M) where M.StorageType==S
    {
        self._box = _AnyOrderingBox(base)
    }
    
    open func traversal(_ ops:[Op<S>]) -> AnySequence<Op<S>> {
        return _box.traversal(ops)
    }
    
    internal let _box: _AnyOrderingBoxBase<S>
}

class _AnyOrderingBase {}
class _AnyOrderingBoxBase<S:Storage>: _AnyOrderingBase, Ordering {
    internal func traversal(_ ops:[Op<S>]) -> AnySequence<Op<S>> { _abstract() }
}

class _AnyOrderingBox<Base:Ordering>: _AnyOrderingBoxBase<Base.StorageType> {
    internal init(_ base: Base) { self._base = base }
    
    override func traversal(_ ops:[Op<Base.StorageType>]) -> AnySequence<Op<Base.StorageType>> {
        return _base.traversal(ops)
    }
    
    internal var _base: Base
}

open class SequentialOrdering<S:Storage>: Ordering {
    public typealias StorageType = S
    public typealias Generator = AnyIterator<Op<S>>
    
    public init() {}

    open func traversal(_ ops:[Op<S>]) -> AnySequence<Op<S>> {
        return AnySequence<Op<S>>(ops)
    }
}

public struct RepeatGenerator<S:Storage>: IteratorProtocol {
    var ops:[Op<S>]
    var count:Int
    var index:Int
    
    init(ops:[Op<S>], count:Int) {
        self.ops = ops
        self.count = count*ops.count
        self.index = 0
    }
    
    mutating public func next() -> Op<S>? {
        defer { index += 1}
        if index < count {
            let i = (index % ops.count)
            return ops[i]
        }
        return nil
    }
}

public struct RepeatSequence<S:Storage>: Sequence {
    typealias GeneratorType = RepeatGenerator<S>
    
    var ops:[Op<S>]
    var count:Int
    
    init(ops:[Op<S>], count:Int) {
        self.ops = ops
        self.count = count
    }
    
    public func makeIterator() -> RepeatGenerator<S> {
        return RepeatGenerator<S>(ops: ops, count: count)
    }

}

open class RepeatedSequentialOrdering<S:Storage>: Ordering {
    public typealias StorageType = S
    typealias Generator = RepeatSequence<S>
    
    var count:Int
    
    public init(count:Int) {
        self.count = count
    }
    
    open func traversal(_ ops:[Op<S>]) -> AnySequence<Op<S>> {
        let seq = RepeatSequence<S>(ops: ops, count: count)
        return AnySequence<Op<S>>(seq)
    }

}

protocol Collection: OpType {
    associatedtype StorageType:Storage
    var ops:[Op<StorageType>] { get }
    
    func add(_ op:Op<StorageType>)
}

open class CollectionOp<S:Storage>: Op<S>, Sequence {
    open var ops:[Op<S>]
    open var ordering:AnyOrdering<S>
    
    public init<T:Ordering>(
        ops:[Op<S>],
        inputs:[Op<S>],
        outputs:[Op<S>],
        ordering:T) where T.StorageType==S
    {
        self.ordering = AnyOrdering<S>(ordering)
        self.ops = ops
        super.init(inputs: ["input"], outputs: ["output"])
        connect(from: inputs, "output", to: self, "input")
        self.outputs["output"] = outputs.map { $0.output }
        
        setAction("input", action: self.inputSet)
    }
    
    // required for Copyable
    public required init(op:Op<S>, shared:Bool) {
        let cop = op as! CollectionOp<S>
        ordering = AnyOrdering<S>(cop.ordering)
        ops = cop.ops.map { copy(op: $0, shared: shared) }
        
        super.init(inputs: ["input"], outputs: ["outputs"])
        outputs["output"] = [Tensor<S>(op.output.shape)]
    }
    
    func inputSet(_ label:String, value:[Op<S>]) {
        connect(from: value[0], to: ops[0])
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
        let inputShapes = inputs["input"]!.map {
            let output:Tensor<S> = $0.output()
            return String(describing: output.shape.dims)
        }.joined(separator: ", ")
        
        let value = ops.map { String(describing: $0) }.joined(separator: "\n\t")
        return "<\(className): inputs:\(inputShapes), outputs:\(output.shape.dims)> {\n\t\(value)\n}"
    }
}

open class CollectionGradient<S:Storage>: Op<S>, Gradient {
    public typealias StorageType = S
    open var ops:[Op<StorageType>] = []
    var ordering:AnyOrdering<S>
    
    open var _input:Tensor<S> { return inputs[1].output() }
    open var _gradOutput:Tensor<S> { return inputs[2].output() }
    
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
            if let opInputs:[InputType<S>] = op.inputs["input"] {
                for input in opInputs {
                    if  let fromOp = gradOps[op.id], let toOp = gradOps[input.op!.id] {
                        connect(from: fromOp, to: toOp, "gradOutput")
                    }
                }
            }
        }
    }
    
    func gradOutputSet(_ key:String, value:[Op<S>]) {
        connect(from: value, "output", to: ops.first!, "gradOutput")
    }
    
    open override func apply() {
        print("CollectionGrad: \(_gradOutput)")
        
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
            print("graph: \(output)")
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
}

extension CollectionOp:Differentiable {
    public func gradient() -> GradientType {
        return CollectionGradient<S>(op: self)
    }
}

open class SequentialOp<S:Storage>: CollectionOp<S> {
    public init(_ ops:[Op<S>], modify:Bool=true) {
        if modify {
            for i in 0..<ops.count-1 {
                connect(from: ops[i], to: ops[i+1])
            }
        }
        
        super.init(ops: ops,
                   inputs: [ops.first!],
                   outputs: [ops.last!],
                   ordering: SequentialOrdering())
    }
    
    // FIXME: had to get rid of `modify` to avoid compiler segfaulting
    public convenience init(_ ops:Op<S>...) {
        self.init(ops)
    }
    
    // required for Copyable
    public required convenience init(op:Op<S>, shared:Bool) {
        let cop = op as! SequentialOp<S>
        let ops = cop.ops.map { copy(op: $0, shared: shared) }
        
        self.init(ops, modify: true)
    }
}

// TODO: previous versions of Swift didn't allow override of an extension.
// it seems to compile now. Should finish this, as it allows a more efficient
// method to calculate the gradient
//public class SequentialGradient<S:Storage>: CollectionGradient<S> {
//    public required init(op:CollectionOp<S>) {
//        super.init(op: op)
//        
//        // TODO: can likely make init more efficient by traversing
//        // list in reverse order instead of using processOutputs
//    }
//    
//    required public init(op: Op<S>, shared: Bool) {
//        fatalError("init(op:shared:) has not been implemented")
//    }
//}

open class RecurrentCollectionOp<S:Storage>: CollectionOp<S> {
    var recurrentVars:[Variable<S>]
    
    public init<T:Ordering>(
        ops:[Op<S>],
        inputs:[Op<S>],
        outputs:[Op<S>],
        recurrentVars:[Variable<S>], // inputs get copied to these
        ordering:T) where T.StorageType==S
    {
        self.recurrentVars = recurrentVars
        super.init(ops: ops, inputs: inputs, outputs: outputs, ordering: ordering)
    }
    
    public required init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }

    public override func inputSet(_ label: String, value: [Op<S>]) {
        // do nothing?
    }
    
    open override func apply() {
        // copy over input array to recurrentVars
        if let inputs:[InputType<S>] = self.inputs["input"] {
            for (input, recurrentVar) in zip(inputs, recurrentVars) {
                copy(from: input.output(), to: recurrentVar.output)
            }
        }
        
        super.apply()
    }
}

open class RecurrentCollectionGrad<S:Storage>: CollectionGradient<S> {
    public typealias StorageType = S

    var recurrentVars:[VariableGrad<S>]

    public init<T:Ordering>(
        ops:[Op<S>],
        inputs:[Op<S>],
        outputs:[Op<S>],
        recurrentVars:[VariableGrad<S>], // inputs get copied to these
        ordering:T) where T.StorageType==S
    {
        self.recurrentVars = recurrentVars
        super.init(ops: ops, inputs: inputs, outputs: outputs, ordering: ordering)
    }
    
    public required init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    public required init(op: CollectionOp<S>) {
        fatalError("init(op:) has not been implemented")
    }
    
    open override func apply() {
        // copy over input array to recurrentVars
        if let inputs:[InputType<S>] = self.inputs["gradOutput"] {
            for (input, recurrentVar) in zip(inputs, recurrentVars) {
                copy(from: input.output(), to: recurrentVar.output)
            }
        }
        
        super.apply()
    }
    
    open override func reset() {
        ops.forEach { $0.reset() }
        
        fill(output, value: 0)
    }
    
    override func gradOutputSet(_ key:String, value:[Op<S>]) {
        // do nothing
    }
}
