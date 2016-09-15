////
////  sequence.swift
////  stem
////
////  Created by Schneider, Abraham R. on 6/5/16.
////  Copyright © 2016 none. All rights reserved.
////
//
//import Foundation
//
//public class Parallel<S:Storage>: Op<S>, Collection {
//    public typealias StorageType = S
//    
//    public var ops:[Op<StorageType>] = []
//    
//    public override init() {
////        super.init(inputs: [NoOp<S>()], output: Tensor<S>(), labels: ["input"])
////        super.init(inputs: [("input", InputType(NoOp<S>()))],
////                   outputs: [("outputs", Tensor<S>())])
//        super.init()
//        connect(from: NoOp<S>(), "output", to: self, "input")
//        outputs["output"] = Tensor<S>()
//        ops = []
//    }
//
//    public convenience init(_ ops:Op<S>...) {
//        self.init(ops)
//    }
//    
//    public init(_ ops:[Op<S>]) {
//        let last = ops.last!
////        super.init(inputs: [NoOp<S>()],
////                   output: last.output,
////                   labels: ["input"])
////        super.init(inputs: [("input", InputType(NoOp<S>()))],
////                   outputs: last.outputs)
//
//        super.init()
//        connect(from: NoOp<S>(), "output", to: self, "input")
//        let lastOutputs:[Tensor<S>] = last.outputs["output"]!
//        outputs["output"] = lastOutputs
//        
//        self.ops = ops
//        
//        setAction("input", action: self.inputSet)
//    }
//    
//    public required init(op:Op<S>, shared:Bool) {
//        let seq = op as! Sequence<S>
//        for op in seq.ops {
//            ops.append(copy(op, shared: shared))
//        }
//        
////        super.init(inputs: [NoOp<S>()],
////                   output: Tensor<S>(op.output.shape),
////                   labels: ["input"])
////        super.init(inputs: [("input", InputType(NoOp<S>()))],
////                   outputs: [("output", Tensor<S>(op.output.shape))])
//
//        super.init()
//        connect(from: NoOp<S>(), "output", to: self, "input")
//        outputs["output"] = Tensor<S>(op.output.shape)
//        
//        setAction("input", action: self.inputSet)
//    }
//    
//    func inputSet(key:String, value:[Op<S>]) {
//        for op in ops {
//            op.setInput("input", to: value)
//        }
//        
//        // need to allow multiple outputs or concat? 
//    }
//    
//    public override func apply() {
//        for op in ops {
//            op.apply()
//        }
//    }
//    
//    public func add(op:Op<S>) {
//        if ops.count > 0 {
//            op.setInput("input", to: ops.last!)
//        }
//        
//        ops.append(op)
////        output = op.output
//        
//        // Q: This will cause a problem if the sequence changes after
//        // another Op points to the output. It wasn't an issue previously
//        // when holding the input Ops, but with the change of holding just
//        // the Tensors does. Is this acceptable, or does a fix like an
//        // observer needed?
//        outputs["output"] = op.output
//    }
//    
//    public override func params() -> [Tensor<S>] {
//        var flattened:[Tensor<S>] = []
//        for op in ops {
//            flattened += op.params()
//        }
//        
//        return flattened
//    }
//    
//    // TODO: this needs to be rewritten to deal with multiple input/outputs
//    public override var description: String {
//        let className = String(Mirror(reflecting: self).subjectType)
//        let inputShapes = inputs["input"].map { $0.output().shape.dims }
////        let inputShapes = inputs.map {
////            switch $0 {
////            case .TensorValue(let op):
////                op.shape.dims
////            case .TensorArray:
////                assertionFailure()
////            }
////        }
//        
//        let value = ops.map { String($0) }.joinWithSeparator("\n")
//        return "<\(className): inputs=\(inputShapes), outputs=\(output.shape.dims)> {\n\(value)\n}"
//    }
//}
//
//public class Sequence<S:Storage>: Op<S>, Collection {
//    public typealias StorageType = S
//    
//    public var ops:[Op<S>] = []
//    
////    public init() {
//////        super.init(inputs: [NoOp<S>()], output: Tensor<S>(), labels: ["input"])
////        super.init(inputs: [("input", NoOp<S>())],
////                   outputs: [("output", Tensor<S>())])
////        ops = []
////    }
//    
//    public convenience init(_ ops:Op<S>...) {
//        self.init(ops)
//    }
//    
//    public init(_ ops:[Op<S>]) {
//        let last = ops.last!
////        super.init(inputs: [("input", InputType(NoOp<S>()))],
////                   outputs: last.outputs)
//        super.init()
//        connect(from: NoOp<S>(), "output", to: self, "input")
//        let lastOutputs:[Tensor<S>] = last.outputs["output"]!
//        outputs["output"] = lastOutputs
//        
//        self.ops = ops
//        
//        for i in 1..<ops.count {
//            print("setInput for: \(ops[i])")
//            ops[i].setInput("input", to: ops[i-1])
//        }
//        
//        output = ops.last!.output
//        
//        setAction("input", action: self.inputSet)
//    }
//    
//    public required init(op:Op<S>, shared:Bool) {
//        let seq = op as! Sequence<S>
//        for op in seq.ops {
//            ops.append(copy(op, shared: shared))
//        }
//        
////        super.init(inputs: [("input", InputType(NoOp<S>()))],
////                   outputs: [("output", Tensor<S>(op.output.shape))])
//        super.init()
//        connect(from: NoOp<S>(), "output", to: self, "input")
//        outputs["output"] = Tensor<S>(op.output.shape)
//        
//        for i in 1..<ops.count {
//            ops[i].setInput("input", to: ops[i-1])
//        }
//        
//        output = ops.last!.output
//        
//        setAction("input", action: self.inputSet)
//    }
//    
//    func inputSet(key:String, value:[Op<S>]) {
//        // need to change first item in sequence
//        ops[0].setInput("input", to: value)
//    }
//    
//    public override func apply() {
//        for op in ops {
//            op.apply()
//        }
//    }
//    
//    public func add(op:Op<S>) {
//        if ops.count > 0 {
//            op.setInput("input", to: ops.last!)
//        }
//        
//        ops.append(op)
//        output = op.output
//    }
//    
//    public override func params() -> [Tensor<S>] {
//        var flattened:[Tensor<S>] = []
//        for op in ops {
//            flattened += op.params()
//        }
//        
//        return flattened
//    }
//    
//    public override var description: String {
//        let className = String(Mirror(reflecting: self).subjectType)
//        let inputShapes = inputs["input"].map { $0.output().shape.dims }
////        let inputShapes = inputs.map {
////            switch $0 {
////            case .TensorValue(let op):
////                op.shape.dims
////            case .TensorArray:
////                assertionFailure()
////            }
////        }
//        
//        let value = ops.map { String($0) }.joinWithSeparator("\n")
//        return "<\(className): inputs=\(inputShapes), outputs=\(output.shape.dims)> {\n\(value)\n}"
//    }
//}
//
//public class SequenceGradient<S:Storage>: Op<S>, Collection, Gradient {
//    public typealias StorageType = S
//    public var ops:[Op<StorageType>] = []
//    
//    public required init(op:Sequence<S>) {
////        super.init(inputs: [NoOp<S>(), NoOp<S>(), NoOp<S>()],
////                   output: Tensor<S>(),
////                   labels: ["op", "input", "gradOutput"])
//
////        super.init(inputs: [("op", InputType(NoOp<S>())),
////                            ("input", InputType(NoOp<S>())),
////                            ("gradOutput", InputType(NoOp<S>()))],
////                   outputs: [("output", Tensor<S>())])
//        super.init()
//        connect(from: NoOp<S>(), "output", to: self, "op")
//        connect(from: NoOp<S>(), "output", to: self, "input")
//        connect(from: NoOp<S>(), "output", to: self, "gradOutput")
//        outputs["output"] = Tensor<S>()
//        
//        for op in op.ops {
//            if let dop = op as? Differentiable {
//                let grad = dop.gradient()
//                ops.insert(grad as! Op<S>, atIndex: 0)
//            }
//        }
//        
//        output = ops.last!.output
//        
//        for i in 1..<ops.count {
//            ops[i].setInput("gradOutput", to: ops[i-1])
//        }
//        
//        setAction("gradOutput", action: self.gradOutputSet)
//    }
//    
//    public override init() {
//        self.ops = []
////        super.init(inputs: [NoOp<S>(), NoOp<S>(), NoOp<S>()],
////                   output: Tensor<S>(),
////                   labels: ["op", "input", "gradOutput"])
////        super.init(inputs: [("op", InputType<S>(NoOp<S>())),
////                            ("input", InputType<S>(NoOp<S>())),
////                            ("gradOutput", InputType<S>(NoOp<S>()))],
////                   outputs: [("output", Tensor<S>())])
//
//        super.init()
//        connect(from: NoOp<S>(), "output", to: self, "op")
//        connect(from: NoOp<S>(), "output", to: self, "input")
//        connect(from: NoOp<S>(), "output", to: self, "gradOutput")
//        outputs["output"] = Tensor<S>()
//        
//
//        setAction("gradOutput", action: self.gradOutputSet)
//    }
//    
//    func gradOutputSet(key:String, value:[Op<S>]) {
//        ops.first!.setInput("gradOutput", to: value)
//    }
//    
//    public override func apply() {
//        for op in ops {
//            op.apply()
//        }
//    }
//    
//    public func add(op:Op<S>) {
//        if ops.count > 0 {
//            op.setInput("input", to: ops.last!)
//        }
//        
//        ops.append(op)
//        output = op.output
//    }
//    
//    public override func reset() {
//        for op in ops {
//            (op as! GradientType).reset()
//        }
//        
//        fill(output, value: 0)
//    }
//    
//    public override func params() -> [Tensor<S>] {
//        // TODO: move this to constructor and inputSet (no need to recompute every time)
//        var flattened:[Tensor<S>] = []
//        for op in ops {
//            flattened += op.params()
//        }
//        
//        return flattened
//    }
//}
//
//extension Sequence:Differentiable {
//    public func gradient() -> GradientType {
//        return SequenceGradient<S>(op: self)
//    }
//}
//
//public func unroll<S:Storage where S.ElementType:FloatNumericType>(ops:[Op<S>], count:Int) -> Sequence<S> {
//    var unrolled = [Op<S>?](count: count*ops.count, repeatedValue: nil)
//    
//    var c = 0
//    for _ in 0..<count {
//        for op in ops {
//            unrolled[c] = copy(op, shared: true)
//            c += 1
//        }
//    }
//    
//    return Sequence<S>(unrolled.map { $0! })
//}
//
//public func unroll<S:Storage where S.ElementType:FloatNumericType>(op:Op<S>, count:Int) -> Sequence<S> {
//    var unrolled = [Op<S>?](count: count, repeatedValue: nil)
//    
//    for i in 0..<count {
//        unrolled[i] = copy(op, shared: true)
//    }
//    
//    return Sequence<S>(unrolled.map { $0! })
//}
//
//// causes segfault from compiler
////public func unroll<S:Storage>(ops:Op<S>..., count:Int) -> Sequence<S> {
////    let seq:Sequence<S> = unroll(ops, count: count)
////    return seq
////}
//
