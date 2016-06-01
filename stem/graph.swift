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
                   output: nil,
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

public class Sequence<S:Storage>: Op<S>, Collection {
    public typealias StorageType = S
    
    public var ops:[Op<StorageType>] = []
    
    public convenience init(_ ops:Op<S>...) {
        self.init(ops)
    }
    
    public init(_ ops:[Op<S>]) {
        let last = ops.last!
        super.init(inputs: [NoOp<S>()], output: last.output, labels: ["input"])
        
        self.ops = ops
        
        for i in 1..<ops.count {
            ops[i].setInput("input", to: ops[i-1])
        }
    }
    
    public override func apply() {
        for op in ops {
            op.apply()
        }
    }
    
    public func add(op:Op<S>) {
        if ops.count > 0 {
            op.setInput("input", to: ops.last!)
        }
        
        ops.append(op)
        output = op.output
    }
    
    public override func params() -> [Tensor<S>] {
        var flattened:[Tensor<S>] = []
        for op in ops {
            flattened += op.params()
        }
        
        return flattened
    }
}

public class SequenceGradient<S:Storage>: Op<S>, Collection, Gradient {
    public typealias StorageType = S
    public var ops:[Op<StorageType>] = []
    
    public required init(op:Sequence<S>) {
        super.init(inputs: [NoOp<S>(), NoOp<S>(), NoOp<S>()],
                   output: nil,
                   labels: ["op", "input", "gradOutput"])
        
        for op in op.ops {
            if let dop = op as? Differentiable {
                let grad = dop.gradient()
                ops.insert(grad as! Op<S>, atIndex: 0)
            }
        }
        
        output = ops.last!.output
        
        for i in 1..<ops.count {
            ops[i].setInput("gradOutput", to: ops[i-1])
        }
    }
    
    public init() {
        self.ops = []
        super.init(inputs: [NoOp<S>(), NoOp<S>(), NoOp<S>()],
                   output: nil,
                   labels: ["op", "input", "gradOutput"])
    }
    
//    public init(ops:Op<S>...) {
//        self.ops = ops
//        
//        super.init(inputs: [NoOp<S>(), NoOp<S>(), NoOp<S>()],
//                   output: ops.last!.output,
//                   labels: ["op", "input", "gradOutput"])
//        
//        for i in 1..<ops.count {
//            ops[i].setInput("gradOutput", to: ops[i-1])
//        }
//    }
    
    public override func apply() {
        for op in ops {
            op.apply()
        }
    }
    
    public func add(op:Op<S>) {
        if ops.count > 0 {
            op.setInput("input", to: ops.last!)
        }
        
        ops.append(op)
        output = op.output
    }
    
    public func reset() {
        for op in ops {
            (op as! GradientType).reset()
        }
        
        fill(output!, value: 0)
    }
    
    public override func params() -> [Tensor<S>] {
        var flattened:[Tensor<S>] = []
        for op in ops {
            flattened += op.params()
        }
        
        return flattened
    }
}

extension Sequence:Differentiable {
    public func gradient() -> GradientType {
        return SequenceGradient<S>(op: self)
    }
}

protocol Optimizer {
    associatedtype StorageType:Storage
    func optimize() -> Op<StorageType>
}

public class GradientDescentOptimizer<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    var alpha:Symbol<S>
    var forward:Op<S>
    var backward:Op<S>
    var params:[Tensor<S>]
    var gradParams:[Tensor<S>]
    
    // NB: Need to wait for newer version of swift to allow constraints to be placed on
    // Op s.t. its differentiable. Until then, this code can fail at runtime if a
    // non-differentiable op is used
    public init(_ op:Op<S>, alpha:Symbol<S>) {
        self.alpha = alpha
        forward = op
        backward = (op as! Differentiable).gradient() as! Op<S>
        
        params = []
        gradParams = []
        
        super.init(inputs: [alpha],
                   output: forward.output,
                   labels: ["alpha"])
        
        params = forward.params()
        gradParams = backward.params()
    }
    
    public override func apply() {
        (backward as! GradientType).reset()
        
        forward.apply()
        backward.apply()

        // this is a place where having a TensorScalar class might be nice
        let a:S.ElementType = alpha.output![0]
        for (param, gradParam) in Zip2Sequence(params, gradParams) {
            param -= a*gradParam
        }
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
