//
//  SequentialOp.swift
//  stem
//
//  Created by Abraham Schneider on 12/26/16.
//
//

import Foundation
import Tensor

open class SequentialOp<S:Storage>: CollectionOp<S> {
    public init(_ ops:[Op<S>], modify:Bool=true) {
        if modify  {
            for i in 0..<ops.count-1 {
                print("Connecting \(ops[i].id) to \(ops[i+1].id)")
                connect(from: ops[i], to: ops[i+1])
            }
        }
        
        super.init(ops: ops,
                   inputs: [ops.first!],
                   outputs: [ops.last!],
                   ordering: SequentialOrdering())
        
        setAction("input", action: self.inputSet)
    }
    
    // FIXME: had to get rid of `modify` to avoid compiler segfaulting
    public convenience init(_ ops:Op<S>...) {
        self.init(ops)
    }
    
    public  init() {
        super.init(ops: [],
                   inputs: [],
                   outputs: [],
                   ordering: SequentialOrdering())
        
        setAction("input", action: self.inputSet)
    }
    
    // required for Copyable
    public required convenience init(op:Op<S>, shared:Bool) {
        let cop = op as! SequentialOp<S>
        let ops = cop.ops.map { copy(op: $0, shared: shared) }
        
        self.init(ops, modify: true)
    }
    
    public subscript(index:Int) -> Op<S> {
        get { return ops[index] }
    }
    
    func inputSet(_ label:String, input:[Source<S>]) {
//        for op in ops {
//            if let action = op.inputActions[label] {
//                action(label, input)
//            }
//        }
        
//        if let first = ops.first {
//            if let action = first.inputActions[label] {
//                connect(from: input, to: first)
//                action(label, input)
//            }
//        }
        
        setInput(to: input[0])
        if let op = ops.first {
            let inputValues:[Source<S>] = inputs["input"]!
            op.inputs["input"] = inputValues
            if let action = op.inputActions[label] {
                action(label, input)
            }
        }
    }
    
    public func append(_ op:Op<S>) {
        //        if let last = ops.last!
        let lastOp = ops.last
        ops.append(op)
        
        if let last = lastOp {
            connect(from: last, "output", to: op, "input")
        } else {
            let input:[Source<S>] = inputs["input"]!
            op.inputs["input"] = input
        }
        
        outputs["output"] = [op.output]
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
