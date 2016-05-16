//
//  graph3.swift
//  stem
//
//  Created by Schneider, Abraham R. on 4/10/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation

public protocol Operation {
    associatedtype StorageType:Storage
    
    var inputs:[AnyOp<StorageType>] { get }
    var output:Tensor<StorageType> { get }
    var id:Int { get }

    func apply()
}

@noreturn @inline(never)
internal func _abstract(file: StaticString = #file, line: UInt = #line) {
    fatalError("Method must be overridden", file: file, line: line)
}

public class AnyOp<S:Storage>: Operation, Hashable {
    public typealias StorageType = S
    
    public var inputs:[AnyOp<StorageType>] { return _box.inputs }
    public var output:Tensor<StorageType> { return _box.output }
    public var id:Int { return _box.id }
    
    public init<M:Operation where M.StorageType==S>(_ base:M) {
        self._box = _AnyGraphOpBox(base)
    }
    
    public func apply() { return _box.apply() }
    
    public var hashValue:Int { return id }
    
    internal let _box: _AnyGraphModuleBoxBase<S>
}

public func ==<S:Storage>(lhs:AnyOp<S>, rhs:AnyOp<S>) -> Bool {
    return lhs.id == rhs.id
}


class _AnyGraphOpBase {}
class _AnyGraphModuleBoxBase<S:Storage>:
    _AnyGraphOpBase, Operation
{
    internal var inputs:[AnyOp<S>] { _abstract() }
    internal var output:Tensor<S> { _abstract() }
    internal var id:Int { _abstract() }
    internal func apply() { _abstract() }
}

class _AnyGraphOpBox<Base:Operation>:
    _AnyGraphModuleBoxBase<Base.StorageType>
{
    internal init(_ base: Base) { self._base = base }
    
    override var inputs:[AnyOp<Base.StorageType>] { return _base.inputs }
    override var output:Tensor<Base.StorageType> { return _base.output }
    override var id:Int { return _base.id }
    
    override func apply() { _base.apply() }
    
    internal var _base: Base
}

public func anyOp<Op:Operation>(op:Op) -> AnyOp<Op.StorageType>
{
    return AnyOp<Op.StorageType>(op)
}

// TODO: replace with random numbers
var idCounter:Int = 0

internal func nextId() -> Int {
    let value = idCounter
    idCounter += 1
    return value
}

public class Symbol<S:Storage>: Operation {
    public var inputs:[AnyOp<S>] = []
    public var output:Tensor<S>
    public var id:Int
    
    public init(_ input:Tensor<S>) {
        output = input
        id = nextId()
    }
    
    public init(_ shape:Extent) {
        output = Tensor<S>(shape)
        id = nextId()
    }
    
    public func set(input:Tensor<S>) {
        output = input
    }
    
    public func apply() {}
}

public class Sigmoid<S:Storage where S.ElementType:FloatNumericType>: Operation {
    public var inputs:[AnyOp<S>]
    public var output:Tensor<S>
    public var id:Int
    
    public init<Op:Operation where Op.StorageType==S>(input:Op) {
        inputs = [anyOp(input)]
        output = Tensor<StorageType>(inputs[0].output.shape)
        id = nextId()
    }
    
    public func apply() {
        // check that value exists, if not, allocate with proper shape
        sigmoid(inputs[0].output, output: output)
    }
}

public class Linear<S:Storage where S.ElementType:FloatNumericType>: Operation {
    public var inputs:[AnyOp<S>] = []
    public var output:Tensor<S>
    public var id:Int
    
    public var weight:Tensor<S>
    public var bias:Tensor<S>
    
    
    public init(numInputs:Int, numOutputs:Int) {
        weight = uniform(Extent(numOutputs, numInputs))
        bias = zeros(Extent(numOutputs))
        output = zeros(Extent(numOutputs, numInputs))
        id = nextId()
    }
    
    public init<Op:Operation where Op.StorageType==S>(input:Op, numOutputs:Int) {
        inputs = [anyOp(input)]
        
        let inputSize = input.output.shape[0]
        weight = uniform(Extent(numOutputs, inputSize))
        bias = zeros(Extent(numOutputs))
        
        // TOOD: think about letting output be passed as an optional parameter
        output = zeros(bias.shape)
        id = nextId()
    }
    
    public init<Op:Operation where Op.StorageType==S>(input:Op, weight:Tensor<S>, bias:Tensor<S>) {
        self.weight = weight
        self.bias = bias
        
        inputs = [anyOp(input)]
        output = zeros(bias.shape)
        id = nextId()
    }
    
    public func apply() {
        // check values exist, if not allocate
        dot(weight, inputs[0].output, result: output)
        add(output, bias, result: output)
    }
}

public protocol Traversal {
    associatedtype StorageType:Storage
    var ops:[AnyOp<StorageType>] { get }
    func apply()
}

public class SequentialTraversal<S:Storage where S.ElementType:FloatNumericType>: Traversal {
    public var ops:[AnyOp<S>] = []
    
    public init() {}
    
    public init(_ ops:AnyOp<S>...) {
        self.ops = ops
    }
    
    public func add<Op:Operation where Op.StorageType == S>
        (op:Op)
    {
        ops.append(AnyOp<StorageType>(op))
    }
    
    public func apply() {
        for op in ops {
            op.apply()
        }
    }
}

// returns input -> output dependencies
public func calcDependencies<S:Storage>(ops:[AnyOp<S>]) -> ([AnyOp<S>:[AnyOp<S>]], [AnyOp<S>]) {
    var deps:[AnyOp<S>:[AnyOp<S>]] = [:]
    var nodeps = Set<AnyOp<S>>(ops)
    
    for op in ops {
        for input in op.inputs {
            if var dep = deps[input] {
                dep.append(op)
            } else {
                deps[input] = [op]
            }
            
            // if it receives input, remove it from the no-dependency list
            nodeps.remove(op)
        }
    }
    
    return (deps, Array<AnyOp<S>>(nodeps))
}

public class Graph<S:Storage>: Traversal {
    public var ops:[AnyOp<S>] = []
    public var deps:[AnyOp<S>:[AnyOp<S>]] = [:]
    public var inputs:[AnyOp<S>] = []
    var needsBuild:Bool = true
    
    public init() {}
    
    public init(_ ops:AnyOp<S>...) {
        self.ops = ops
        build()
    }
    
    func build() {
        (deps, inputs) = calcDependencies(ops)
        needsBuild = false
    }
    
    public func add<Op:Operation where Op.StorageType == S>
        (op:Op)
    {
        ops.append(anyOp(op))
        needsBuild = true
    }
    
    public func apply() {
        if needsBuild {
            build()
        }
        
        traverse(inputs)
    }
    
    func traverse(sub:[AnyOp<S>]) {
        var next:[AnyOp<S>] = []
        for op in sub {
            op.apply()
            
            if let nextOps = deps[op] {
                next.appendContentsOf(nextOps)
            }
        }
            
        if next.count > 0 {
            traverse(next)
        }
    }
}
