//
//  graph3.swift
//  stem
//
//  Created by Schneider, Abraham R. on 4/10/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation


@noreturn @inline(never)
internal func _abstract(file:StaticString = #file, line:UInt=#line) {
    fatalError("Method must be overridden", file: file, line: line)
}

public protocol Ordering {
    associatedtype StorageType:Storage
    
//    func generate() -> AnyGenerator<Op<StorageType>>
    func traversal(ops:[Op<StorageType>]) -> AnySequence<Op<StorageType>>
}

public class AnyOrdering<S:Storage>: Ordering {
    public typealias StorageType = S
    
    public required init<M:Ordering where M.StorageType==S>(_ base:M)
    {
        self._box = _AnyOrderingBox(base)
    }
    
    public func traversal(ops:[Op<S>]) -> AnySequence<Op<S>> {
        return _box.traversal(ops)
    }
    
    internal let _box: _AnyOrderingBoxBase<S>
}

class _AnyOrderingBase {}
class _AnyOrderingBoxBase<S:Storage>: _AnyOrderingBase, Ordering {
    internal func traversal(ops:[Op<S>]) -> AnySequence<Op<S>> { _abstract() }
}

class _AnyOrderingBox<Base:Ordering>: _AnyOrderingBoxBase<Base.StorageType> {
    internal init(_ base: Base) { self._base = base }
    
    override func traversal(ops:[Op<Base.StorageType>]) -> AnySequence<Op<Base.StorageType>> {
        return _base.traversal(ops)
    }
    
    internal var _base: Base
}

public class SequentialOrdering<S:Storage>: Ordering {
    public typealias StorageType = S
    public typealias Generator = AnyGenerator<Op<S>>
    
    public init() {}

    public func traversal(ops:[Op<S>]) -> AnySequence<Op<S>> {
        return AnySequence<Op<S>>(ops)
    }
}

public struct RepeatGenerator<S:Storage>: GeneratorType {
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

public struct RepeatSequence<S:Storage>: SequenceType {
    typealias GeneratorType = RepeatGenerator<S>
    
    var ops:[Op<S>]
    var count:Int
    
    init(ops:[Op<S>], count:Int) {
        self.ops = ops
        self.count = count
    }
    
    public func generate() -> RepeatGenerator<S> {
        return RepeatGenerator<S>(ops: ops, count: count)
    }

}

public class RepeatedSequentialOrdering<S:Storage>: Ordering {
    public typealias StorageType = S
    typealias Generator = RepeatSequence<S>
    
    var count:Int
    
    public init(count:Int) {
        self.count = count
    }
    
    public func traversal(ops:[Op<S>]) -> AnySequence<Op<S>> {
        let seq = RepeatSequence<S>(ops: ops, count: count)
        return AnySequence<Op<S>>(seq)
    }

}

protocol Collection: OpType {
    associatedtype StorageType:Storage
    var ops:[Op<StorageType>] { get }
    
    func add(op:Op<StorageType>)
}

public class CollectionOp<S:Storage>: Op<S>, SequenceType {
    public var ops:[Op<S>]
    public var ordering:AnyOrdering<S>
    
    public init<T:Ordering where T.StorageType==S>(
        ops:[Op<S>],
        inputs:[Op<S>],
        outputs:[Op<S>],
        ordering:T)
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
        ops = cop.ops.map { copy($0, shared: shared) }
        
        super.init(inputs: ["input"], outputs: ["outputs"])
        outputs["output"] = Tensor<S>(op.output.shape)
    }
    
    func inputSet(label:String, value:[Op<S>]) {
        connect(from: value[0], to: ops[0])
    }
    
    public override func apply() {
        for op in ordering.traversal(ops) {
            op.apply()
        }
    }
    
    public func generate() -> AnyGenerator<Op<S>> {
        return ordering.traversal(ops).generate()
    }
    
    public override func reset() {
        fill(output, value: 0)
    }

    public override var description: String {
        let className = String(Mirror(reflecting: self).subjectType)
        let inputShapes = inputs["input"]!.map {
            let output:Tensor<S> = $0.output()
            return String(output.shape.dims)
        }.joinWithSeparator(", ")
        
        let value = ops.map { String($0) }.joinWithSeparator("\n\t")
        return "<\(className): inputs:\(inputShapes), outputs:\(output.shape.dims)> {\n\t\(value)\n}"
    }
}

public class CollectionGradient<S:Storage>: Op<S>, Gradient {
    public typealias StorageType = S
    public var ops:[Op<StorageType>] = []
    var ordering:AnyOrdering<S>
    
    public required init(op:CollectionOp<S>) {
        ordering = op.ordering
        super.init(inputs: ["op", "inputs", "gradOutput"], outputs: ["output"])
        
        outputs["output"] = Tensor<S>()

        // start with outputs, and work backwards
        createBackwardsGraph(op.ops)
        output = ops.last!.output
        
        setAction("gradOutput", action: self.gradOutputSet)
    }
    
    public init<T:Ordering where T.StorageType==S>(
        ops:[Op<S>],
        inputs:[Op<S>],
        outputs:[Op<S>],
        ordering:T)
    {
        self.ordering = AnyOrdering<S>(ordering)
        self.ops = ops
        super.init(inputs: ["op", "input", "gradOutput"], outputs: ["outputs"])
        connect(from: outputs, "output", to: self, "gradOutput")
        
        self.outputs["output"] = outputs.map { $0.output }
        setAction("gradOutput", action: self.gradOutputSet)
    }
    
    func createBackwardsGraph(ops:[Op<S>]) {
        // convert all ops -> opGradient
        var gradOps:[Int:Op<S>] = [:]
        
        for op in ops.reverse() {
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
                    if  let fromOp = gradOps[op.id], toOp = gradOps[input.op!.id] {
                        connect(from: fromOp, to: toOp, "gradOutput")
                    }
                }
            }
        }
    }
    
    func gradOutputSet(key:String, value:[Op<S>]) {
        connect(from: value, "output", to: ops.first!, "gradOutput")
    }
    
    public override func apply() {
        for op in ordering.traversal(ops) {
            op.apply()
        }
    }
    
    public override func reset() {
        for op in ops {
            (op as! GradientType).reset()
        }
        
        fill(output, value: 0)
    }
    
    public override var description: String {
        let className = String(Mirror(reflecting: self).subjectType)
        
        let value = ops.map { String($0) }.joinWithSeparator("\n\t")
        return "<\(className): inputs=?, outputs=\(output.shape.dims)> {\n\t\(value)\n}"
    }
}

extension CollectionOp:Differentiable {
    public func gradient() -> GradientType {
        return CollectionGradient<S>(op: self)
    }
}

public class SequentialOp<S:Storage>: CollectionOp<S> {
    public init(_ ops:[Op<S>], modify:Bool=true)
    {
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
        let ops = cop.ops.map { copy($0, shared: shared) }
        
        self.init(ops, modify: true)
    }
}

// NB: Currently cannot overide an extension. For now just use
// gradient as defined by CollectionGradient.
//
//public class SequentialGradient<S:Storage>: CollectionGradient<S> {
//    public required init(op:CollectionOp<S>) {
//        super.init(op: op)
//        
//        // TODO: can likely make init more efficient by traversing
//        // list in reverse order instead of using processOutputs
//    }
//}
