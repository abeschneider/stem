//
//  graph3.swift
//  stem
//
//  Created by Schneider, Abraham R. on 4/10/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation


//
////protocol GraphModule {
////    associatedtype StorageType:Storage
////    
////    var inputs:Int { get }
////    var outputs:Int { get }
////    
////    func apply(inputs:[Tensor<StorageType>], outputs: [Tensor<StorageType>])
////}
//

public class Symbol<S:Storage> {
    public var value:Tensor<S>?
    public var set:Bool
    
    public init() {
        set = false
    }
    
    public init(shape:Extent) {
        value = Tensor<S>(shape)
        set = false
    }
    
    public init(value:Tensor<S>, copy makeCopy:Bool=false) {
        if makeCopy {
            self.value = copy(value)
        } else {
            self.value = value
        }
        set = false
    }
    
    public func setValue(value:Tensor<S>, copy makeCopy:Bool=false) {
        if makeCopy {
            self.value = copy(value)
        } else {
            self.value = value
        }
        set = true
    }
    
    public func reset() {
        set = false
    }
}

//public class SymbolArray<StorageType:Storage> {
//    var symbols:[Symbol<StorageType>]
//    var isSet:[Bool]
//    
//    public init(symbols:[Symbol<StorageType>]) {
//        self.symbols = symbols
//        isSet = [Bool](count: symbols.count, repeatedValue: false)
//    }
//    
//    public func reset() {
//        for i in 0..<isSet.count {
//            isSet[i] = false
//        }
//    }
//    
//    public subscript(index:Int) -> Symbol<StorageType> {
//        get {
//           return symbols[index]
//        }
//        set {
//            symbols[index] = newValue
//            isSet[index] = true
//        }
//    }
//}

public protocol Operation {
    associatedtype StorageType:Storage
    
    var inputs:[Symbol<StorageType>] { get }
    var outputs:[Symbol<StorageType>] { get }

    func ready() -> Bool
    func apply()
}

@noreturn @inline(never)
internal func _abstract(file: StaticString = #file, line: UInt = #line) {
    fatalError("Method must be overridden", file: file, line: line)
}

public class AnyGraphModule<S:Storage>: Operation {
    public typealias StorageType = S
    
//    var inputs:Int { return _box.inputs }
//    var outputs:Int { return _box.outputs }
//    var outputShape:[Extent] { return _box.outputShape }
    public var inputs:[Symbol<StorageType>] { return _box.inputs }
    public var outputs:[Symbol<StorageType>] { return _box.outputs }
    

    
    init<M:Operation where M.StorageType==S>(_ base:M) {
        self._box = _AnyGraphModuleBox(base)
    }
    
//    func apply(inputs:[Tensor<StorageType>], outputs: [Tensor<StorageType>]) {
//        return _box.apply(inputs, outputs: outputs)
//    }
    
    public func ready() -> Bool { return _box.ready() }
    public func apply() { return _box.apply() }
    
    internal let _box: _AnyGraphModuleBoxBase<S>
}

class _AnyGraphModuleBase {}
class _AnyGraphModuleBoxBase<S:Storage>:
    _AnyGraphModuleBase, Operation
{
    
//    internal var inputs:Int { _abstract() }
//    internal var outputs:Int { _abstract() }
//    internal var outputShape:[Extent] { _abstract() }
    internal var inputs:[Symbol<S>] { _abstract() }
    internal var outputs:[Symbol<S>] { _abstract() }

//    internal func apply(inputs:[Tensor<S>], outputs: [Tensor<S>]) {
//        _abstract()
//    }
    
    internal func ready() -> Bool { _abstract() }
    internal func apply() { _abstract() }
}

class _AnyGraphModuleBox<Base:Operation>:
    _AnyGraphModuleBoxBase<Base.StorageType>
{
    internal init(_ base: Base) { self._base = base }
    
//    override var inputs:Int { return _base.inputs }
//    override var outputs:Int { return _base.outputs }
    override var inputs:[Symbol<Base.StorageType>] { return _base.inputs }
    override var outputs:[Symbol<Base.StorageType>] { return _base.outputs }
    
//    override func apply(inputs:[Tensor<Base.StorageType>], outputs: [Tensor<Base.StorageType>]) {
//        _base.apply(inputs, outputs: outputs)
//    }
    
    override func ready() -> Bool { return _base.ready() }
    override func apply() { _base.apply() }
    
    internal var _base: Base
}

public struct Sigmoid<StorageType:Storage where StorageType.ElementType:FloatNumericType>: Operation {
//    var inputs:Int { return 1 }
//    var outputs:Int { return 1 }
//    var outputShape:[Extent] { return [Extent(-1)] }
    public var inputs:[Symbol<StorageType>]
    public var outputs:[Symbol<StorageType>]
    
    public init(input:Symbol<StorageType>) {
        inputs = [input]
        let symbol = Symbol<StorageType>(shape: input.value!.shape)
        outputs = [symbol]
    }
    
    public func ready() -> Bool {
        return inputs[0].set
    }
    
    public func apply() {
        // check that value exists, if not, allocate with proper shape
        sigmoid(inputs[0].value!, output: outputs[0].value!)
    }
//    func apply(inputs:[Tensor<StorageType>], outputs:[Tensor<StorageType>]) {
//        sigmoid(inputs[0], output: outputs[0])
//    }
}

public struct Linear<StorageType:Storage where StorageType.ElementType:FloatNumericType>: Operation {
//    var inputs:Int { return 1  }
//    var outputs:Int { return 1 }
//    var outputShape:[Extent]
    public var inputs:[Symbol<StorageType>]
    public var outputs:[Symbol<StorageType>]
    
//    var weight:Matrix<StorageType>
//    var bias:Vector<StorageType>
    var weight:Symbol<StorageType>
    var bias:Symbol<StorageType>
    
    public init(input:Symbol<StorageType>, weight:Symbol<StorageType>, bias:Symbol<StorageType>) {
        self.weight = weight
        self.bias = bias
        
        inputs = [input]
        let symbol = Symbol<StorageType>(shape: bias.value!.shape)
        outputs = [symbol]
    }
    
    public func ready() -> Bool {
        return inputs[0].set
    }

//    func apply(inputs:[Tensor<StorageType>], outputs:[Tensor<StorageType>]) {
    public func apply() {
        // check values exist, if not allocate
        dot(weight.value!, inputs[0].value!, result: outputs[0].value!)
        add(outputs[0].value!, bias.value!, result: outputs[0].value!)
    }
}

//public struct Variable<StorageType:Storage>: Operation {
//    public var inputs:[Symbol<StorageType>]
//    public var outputs:[Symbol<StorageType>]
//    
//    public init(tensor:Tensor<StorageType>) {
//        let symbol = Symbol<StorageType>(value:tensor)
//        outputs = [symbol]
//        inputs = []
//    }
//    
//    public func apply() {
//        // do nothing
//    }
//}

public protocol Traversal {
    associatedtype StorageType:Storage
    var ops:[AnyGraphModule<StorageType>] { get }
    func add<Op:Operation where Op.StorageType == StorageType>(op:Op)
    func update()
}

public class SequentialTraversal<StorageType:Storage>: Traversal {
    public var ops:[AnyGraphModule<StorageType>]

    public init() {
        // can't figure out a way to allow variadic parameters with
        // varying generic types, so for now can only create a
        // Traversal using `add`
        self.ops = [] //ops.map { AnyGraphModule<StorageType>($0) }
    }
    
    public func add<Op:Operation where Op.StorageType == StorageType>
        (op:Op)
    {
        ops.append(AnyGraphModule<StorageType>(op))
    }
    
    public func update() {
        for op in ops {
            op.apply()
        }
    }
}

//public class Graph<StorageType:Storage>: Operation {
//    public var inputs:[Symbol<StorageType>]
//    public var outputs:[Symbol<StorageType>]
//    
//    public var ops:[AnyGraphModule<StorageType>]
//
//    public init() {
//        inputs = []
//        outputs = []
//        ops = []
//    }
//    
//    public func add(op:AnyGraphModule<StorageType>) {
//        ops.append(op)
//    }
//    
//    public func ready() -> Bool {
//        return !(ops.map { $0.ready() }).contains(false)
//    }
//    
//    public func apply() {
//        //prioritize based on dependencies?
//        for op in ops {
//            if op.ready() {
//                op.apply()
//            }
//        }
//    }
//}

//struct Port<StorageType:Storage> {
//    var input:AnyGraphModule<StorageType>
//    var output:AnyGraphModule<StorageType>
//
//    var inputIndex:Int
//    var outputIndex:Int
//    
//    init(_ input:AnyGraphModule<StorageType>, _ inputIndex:Int, _ output:AnyGraphModule<StorageType>, _ outputIndex:Int) {
//        self.input = input
//        self.inputIndex = inputIndex
//        self.output = output
//        self.outputIndex = outputIndex
//    }
//}

//class Connector<StorageType:Storage> {
////    var from:[AnyGraphModule<StorageType>]
//    var target:AnyGraphModule<StorageType>
//    var outputs:[Tensor<StorageType>]
//    var ports:[Port<StorageType>]
////    var expectedIncoming:Int
//    var incoming:Int
//    
//    init(target:AnyGraphModule<StorageType>) {
//        self.target = target
//        incoming = 0
//    }
//    
//    func connect(source:AnyGraphModule<StorageType>, sourceIndex:Int, targetIndex:Int) {
//        let port = Port<StorageType>(source, sourceIndex, target, targetIndex)
//        ports.append(port)
//    }
//    
//    func isReady() -> Bool {
//        return false
//    }
//}

//class Graph<StorageType:Storage> {
//    var ops:[String: AnyGraphModule<StorageType>] = [:]
////    var connectors:[Connector<StorageType>] = []
//    var connectors:[String: Connector<StorageType>] = [:]
//    
//    func add(name:String, op:AnyGraphModule<StorageType>) {
//        ops[name] = op
//    }
//    
//    func connect(source:String, sourceIndex:Int, target:String, targetIndex:Int) {
//        let sourceOp = ops[source]!
//        let targetOp = ops[target]!
//        
//        if let connector = connectors[target] {
//            connector.connect(sourceOp, sourceIndex: sourceIndex, targetIndex: targetIndex)
//        } else {
//            let connector = Connector(target: targetOp)
//            connectors[target] = connector
//            connector.connect(sourceOp, sourceIndex: sourceIndex, targetIndex: targetIndex)
//        }
////        connectors.append(connector)
//    }
//    
//    func update() {
//        for connector in connectors {
//            if connector.isReady() {
//                // toOp.apply()
//            }
//        }
//    }
//}

/*

 
 let linear = Linear<D>(inputs: 5, outputs: 5)
 let sigmoid = Sigmoid<D>()
 
 let g = Graph()
 g.add("l1", linear)
 g.add("s1", sigmoid)
 g.connect("l1", "s1")
 
*/