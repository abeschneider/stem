//
//  graph3.swift
//  stem
//
//  Created by Schneider, Abraham R. on 4/10/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation
import Tensor


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

open class RecurrentCollectionOp<S:Storage>: CollectionOp<S> {
    var recurrentVars:[VariableOp<S>]
    
    public init<T:Ordering>(
        ops:[Op<S>],
        inputs:[Op<S>],
        outputs:[Op<S>],
        recurrentVars:[VariableOp<S>], // inputs get copied to these
        ordering:T) where T.StorageType==S
    {
        self.recurrentVars = recurrentVars
        super.init(ops: ops, inputs: inputs, outputs: outputs, ordering: ordering)
    }
    
    public required init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }

//    public override func inputSet(_ label: String, value: [Op<S>]) {
//        // do nothing?
//    }
    
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
    
    override func gradOutputSet(_ key:String, value:[Source<S>]) {
        // do nothing
    }
}
