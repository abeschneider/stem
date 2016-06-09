//
//  graph3.swift
//  stem
//
//  Created by Schneider, Abraham R. on 4/10/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation


public protocol Traversal: SequenceType {
    associatedtype StorageType:Storage
    
    init(ops:[Op<StorageType>])
    func generate() -> AnyGenerator<Op<StorageType>>
}

public class SequenceTraversal<S:Storage>: Traversal {
    public typealias StorageType = S
    public typealias Generator = AnyGenerator<Op<S>>
    
    var ops:[Op<S>] = []
    
    required public init(ops:[Op<S>]) {
        self.ops = ops
    }
    
    public func generate() -> Generator {
        var index = 0
        return AnyGenerator {
            if index >= self.ops.count { return nil }
            defer { index += 1 }
            return self.ops[index]
        }
    }
}

protocol Collection: OpType {
    associatedtype StorageType:Storage
    var ops:[Op<StorageType>] { get }
    
    func add(op:Op<StorageType>)
}

public class Graph<T:Traversal>: Op<T.StorageType>, SequenceType {
    typealias S = T.StorageType
    // contains: traversal, nodes
    public var ops:[Op<S>] = []
    
    public init() {
        super.init(inputs: [],
                   output: Tensor<S>(),
                   labels: ["input"])
    }
    
    public init(output:Tensor<S>) {
        super.init(inputs: [],
                   output: output,
                   labels: ["input"])
    }
    
    public init(_ ops:[Op<S>], output:Tensor<S>) {
        super.init(inputs: [],
                   output: output,
                   labels: ["input"])
        self.ops = ops
    }
    
    public init(_ ops:[Op<S>], output:Tensor<S>, labels:[String]) {
        super.init(inputs: [],
                   output: output,
                   labels: labels)
        self.ops = ops
    }
    
    public func add(op:Op<S>) {}
    
    public override func apply() {
        let traversal = T(ops: ops).generate()
        for op in traversal {
            op.apply()
        }
    }
    
    public func generate() -> AnyGenerator<Op<S>> {
        return T(ops: ops).generate()
    }    
}

public func calcDependencies<S:Storage>(ops:[Op<S>]) -> ([Op<S>:[Op<S>]], [Op<S>:[Op<S>]], [Op<S>], [Op<S>]) {
    var fdeps:[Op<S>:[Op<S>]] = [:]
    var bdeps:[Op<S>:[Op<S>]] = [:]
    var roots = [Op<S>]()
    
    for op in ops {
        if op.inputs.count == 0 {
            roots.append(op)
        } else {
            for input in op.inputs {
                if var dep = fdeps[input] {
                    dep.append(op)
                } else {
                    fdeps[input] = [op]
                    bdeps[op] = [input]
                }
            }
        }
    }
    
    let terminals:[Op<S>] = fdeps.filter { $1.count == 0 }.map { $0.0 }
    
    return (fdeps, bdeps, terminals, roots)
}
