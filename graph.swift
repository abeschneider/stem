//
//  graph.swift
//  stem
//
//  Created by Abe Schneider on 12/24/15.
//  Copyright © 2015 none. All rights reserved.
//

import Foundation

// figure out way to build collection of modules and specify their input/output
// relationships

/*

var g = Graph()
g.add(linear1)
g.add(sigmoid1)
g.add(linear2)
g.add(sigmoid2)

// In theory can do automatically, but for now only allow
// connections that make sense. Each module will have to
// define it's input/output relationships (thus split/merge modules will be needed)
g.connect(linear1, to: sigmoid1)
g.connect(sigmoid1, to: linear2)
g.connect(linear2, to: sigmoid2)


var g = Graph()
g.add(input1)
g.add(input2)
g.add(concat)
g.add(sigmoid1)
g.add(split)
g.add(output1)
g.add(output2)


// merge allows up to N input connections
g.connect(input1, to: concat)
g.connect(input2, to: concat)

g.connect(merge, to: sigmoid1)
g.connect(sigmoid1, to: split)

// split allows for N output connections
g.connect(split, to: output1)
g.connect(split, to: output2)

// can implement a RAE by:
//  1. adding above code
//  2. adding correct number of connections to form tree

TODO:
1. Module -> ForwardOp
2. GradientModule -> BackwardOp
3. make: GraphModule
    a. inputShape
    b. outputShape
    c. numInputs
    d. numOutputs
4. make: Graph
    a. collection of GraphModules
    b. connections between modules
    c. connect(input:GraphModule, to:GraphModule)
        i. needs to verify numInputs and numOutputs is correct
*/

//struct Variable {
//    // is there a way to keep any type of Tensor regardless of StorageType?
//}

protocol GraphModule {
    typealias StorageType:Storage
    
    // If nil, then module has no preferred shape
//    var inputShape:Extent? { get }
//    var outputShape:Extent? { get }
    var output:Tensor<StorageType>? { get set }
    
    // forward direction
    func connect(to to:AnyGraphModule<StorageType>) throws
    
    // backward direction
    func connect(from from:AnyGraphModule<StorageType>) throws
    
    func forward(input:Tensor<StorageType>?)
    func backward(gradOutput:Tensor<StorageType>?)
    
    // returns the output shape for a given input shape
    func outputShape(inputShape:Extent) -> Extent
}

//struct GraphModuleWithStorage<S:Storage>: GraphModule {
//    typealias StorageType = S
//    
//    let _numInputs:() -> Int
//    
//    var numInputs:Int
//    
//    init<M:GraphModule>(module:M) {
//        _numInputs = () -> Int { return module.numInputs }
//        _numOutputs = module.numOutputs
//    }
//}

enum ConnectionError: ErrorType {
    case CannotAcceptInput
    case CannotAcceptOutput
}

@noreturn @inline(never)
internal func _abstract(file: StaticString = __FILE__, line: UInt = __LINE__) {
    fatalError("Method must be overridden", file: file, line: line)
}

class AnyGraphModule<S:Storage>: GraphModule {
    typealias StorageType = S
    
    var output:Tensor<StorageType>? {
        get {
            return _box.output
        }
        
        set {
            _box.output = newValue
        }
    }
    
    init<M:GraphModule where M.StorageType == S>(_ base:M) {
        self._box = _AnyGraphModuleBox(base)
    }
    
    func connect(to to:AnyGraphModule<S>) throws {
        try _box.connect(to:to)
    }
    
    func connect(from from:AnyGraphModule<S>) throws {
        try _box.connect(from:from)
    }

    
    func forward(input:Tensor<StorageType>?=nil) {
        _box.forward(input)
    }
    
    func backward(gradOutput:Tensor<StorageType>?=nil) {
        _box.backward(gradOutput)
    }
    
    func outputShape(inputShape:Extent) -> Extent {
        return _box.outputShape(inputShape)
    }
    
    internal let _box: _AnyGraphModuleBoxBase<S>
}

class _AnyGraphModuleBase {}
class _AnyGraphModuleBoxBase<S:Storage>:
    _AnyGraphModuleBase, GraphModule
{
    internal var output:Tensor<S>? {
        get {
            _abstract()
        }
        
        set {
            _abstract()
        }
    }
    
    internal func forward(input:Tensor<S>?=nil) { _abstract() }
    internal func backward(gradOutput:Tensor<S>?) { _abstract() }
    internal func connect(to to:AnyGraphModule<S>) throws { _abstract() }
    internal func connect(from from:AnyGraphModule<S>) throws { _abstract() }
    internal func outputShape(inputShape: Extent) -> Extent { _abstract() }
}

class _AnyGraphModuleBox<Base:GraphModule>:
    _AnyGraphModuleBoxBase<Base.StorageType>
{
    internal init(_ base: Base) { self._base = base }
    
    override var output:Tensor<Base.StorageType>? {
        get {
            return _base.output
        }
        
        set {
            _base.output = newValue
        }
    }
    
    override func forward(input:Tensor<Base.StorageType>?=nil) {
        _base.forward(input)
    }
    
    override func backward(gradOutput:Tensor<Base.StorageType>?=nil) {
        _base.backward(gradOutput)
    }
    
    internal override func connect(to to:AnyGraphModule<Base.StorageType>) throws {
        try _base.connect(to: to)
    }
    
    internal override func connect(from from:AnyGraphModule<Base.StorageType>) throws {
        try _base.connect(from: from)
    }
    
    internal override func outputShape(inputShape: Extent) -> Extent {
        return _base.outputShape(inputShape)
    }
    
    internal var _base: Base
}

func connect<M1, M2 where M1:GraphModule, M2:GraphModule, M1.StorageType == M2.StorageType>(from from:M1, to:M2) throws {
    try from.connect(to: AnyGraphModule<M2.StorageType>(to))
    try to.connect(from: AnyGraphModule<M1.StorageType>(from))
}

class Group<S:Storage>: GraphModule {
    typealias StorageType = S
    
    var modules:[AnyGraphModule<S>]
    
    var toModule:AnyGraphModule<S>?
    var fromModule:AnyGraphModule<S>?
    
    var output:Tensor<S>? {
        get {
            return modules[modules.count-1].output
        }
        
        set {
            modules[modules.count-1].output = newValue
        }
    }
    
    init() {
        self.modules = []
    }

    func add<M:GraphModule where M.StorageType == S>(module: M) {
        self.modules.append(AnyGraphModule<S>(module))
    }
    
    func connect(to to:AnyGraphModule<S>) throws {
        toModule = to
    }
    
    func connect(from from:AnyGraphModule<S>) throws {
        fromModule = from
    }
    
    func forward(input:Tensor<S>?=nil) {
        _abstract()
    }
    
    func backward(gradOutput:Tensor<S>?=nil) {
        _abstract()
    }
    
    func outputShape(inputShape: Extent) -> Extent {
        _abstract()
    }
}

class Sequence<S:Storage>: Group<S> {
//    init(modules: NSArray) {
//        super.init()
//        for module in modules {
//            add(module)
//        }
//    }
    
    override init() {
        super.init()
    }
    
//    func add(module: AnyObject) {
//        add(module as! AnyGraphModule<S>)
//    }
    
    override func add<M : GraphModule where M.StorageType == S>(module: M) {
        self.modules.append(AnyGraphModule<S>(module))
        
        if self.modules.count > 1 {
            let last = self.modules.count-1
            try! self.modules[last-1].connect(to: self.modules[last])
            try! self.modules[last].connect(from: self.modules[last-1])
        }
    }
    
    override func forward(input:Tensor<S>?=nil) {
        modules.first!.forward(input)
    }
    
    override func backward(gradOutput:Tensor<S>?=nil) {
        modules.last!.backward(gradOutput)
    }
    
    override func outputShape(inputShape: Extent) -> Extent {
        return modules.last!.outputShape(inputShape)
    }
}

class Linear<S:Storage>: GraphModule {
    typealias StorageType = S
    
    var output:Tensor<S>? {
        get { return linear.output }
        set { linear.output = newValue }
    }
    
    var linear:LinearOp<S>
    var grad:LinearGradient<S>
    
    var toModule:AnyGraphModule<S>?
    var fromModule:AnyGraphModule<S>?
    
//    var inputShape:Extent? { return Extent(linear.shape[0]) }
//    var outputShape:Extent? { return Extent(linear.shape[1]) }
 
    init(numInputs:Int, numOutputs:Int) {
        linear = LinearOp<S>(input_size: numInputs, output_size: numOutputs)
        grad = LinearGradient<S>(input_size: numInputs, output_size: numOutputs)
    }
    
    init(weight:Matrix<S>) {
        linear = LinearOp<S>(weight: weight)
        grad = LinearGradient<S>(input_size: weight.shape[0], output_size: weight.shape[1])
    }
    
    func connect(to to:AnyGraphModule<S>) throws {
        toModule = to
    }
    
    func connect(from from:AnyGraphModule<S>) throws {
        fromModule = from
    }

    func forward(input:Tensor<S>?=nil) {
        linear.apply(input!)
        
        if let module = toModule {
            module.forward(output!)
        }
    }
    
    func backward(gradOutput:Tensor<S>?=nil) {
        if let module = fromModule {
            module.backward(gradOutput)
        }
    }
    
    func outputShape(inputShape: Extent) -> Extent {
        return inputShape
    }
}

// should not have `variable`, but instead take inptu from `forward`
class Variable<S:Storage>: GraphModule {
    typealias StorageType = S
    
    var output:Tensor<S>?
    
    var toModule:AnyGraphModule<S>?
    var fromModule:AnyGraphModule<S>?
//    var variable:Tensor<S>
    
    // does not take any input
//    var inputShape:Extent? { return Extent(0) }
    
    // output size of variable we're wrapping
//    var outputShape:Extent? { return variable.shape }
    
//    init(variable:Tensor<S>) {
//        self.variable = variable
//    }
    
    func connect(to to:AnyGraphModule<S>) throws {
        toModule = to
    }
    
    func connect(from from:AnyGraphModule<S>) throws {
        throw ConnectionError.CannotAcceptInput
    }
    
    func forward(input:Tensor<S>?=nil) {
        if let module = toModule {
            module.forward(input)
        }
    }
    
    func backward(gradOutput:Tensor<S>?=nil) {
        // TODO: should allow boolean to decide if we back propagate to
        // tensor or not
        if let module = fromModule {
            module.backward(gradOutput)
        }
    }
    
    func outputShape(inputShape: Extent) -> Extent {
        return Extent(0)
    }
}

class L2Criterion<S:Storage where S.ElementType:NumericType>: GraphModule {
    typealias StorageType = S
    
    var output:Tensor<S>?
    
    var fromModule:AnyGraphModule<S>?
    var error:S.ElementType?
    var target:Tensor<S>
    
    init(target:Tensor<S>) {
        self.target = target
    }

    func connect(to to:AnyGraphModule<S>) throws {
        throw ConnectionError.CannotAcceptOutput
    }
    
    func connect(from from:AnyGraphModule<S>) throws {
        fromModule = from
    }
    
    func forward(input:Tensor<S>?=nil) {
        let s:S.ElementType = sum(target - input!)
        error = S.ElementType(0.5)*(s^S.ElementType(2.0))
    }
    
    func backward(gradOutput:Tensor<S>?=nil) {
        let input = fromModule!.output!
        let gradError = input - target
        
        if let module = fromModule {
            module.backward(gradError)
        }
    }
    
    func outputShape(inputShape: Extent) -> Extent {
        return Extent(1)
    }
}

/*class Concat<S:Storage>: GraphModule {
    var toModule:GraphModule?
    var fromModules:[GraphModule]
    var numInputs:Int { return fromModules.count }
    var numOutputs:Int { return 1 }
    
    // these have to be extracted by the passed modules
    var inputShape:Extent
    var outputShape:Extent
    
    init(modules:[GraphModule]) {
        self.fromModules = []
        inputShape = Extent()
        outputShape = Extent()
    }
    
    func connect(to:GraphModule) throws {
        toModule = to
    }
    
    func forward<S:Storage>(input:Tensor<S>?) {
    
    }
    
    func backward<S:Storage>(input:Tensor<S>) {
        
    }
}

class Split: GraphModule {
    var toModules:[GraphModule]
    var fromModule:GraphModule?
}
*/