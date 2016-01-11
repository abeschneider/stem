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
    
    // forward direction
    func connect(to to:AnyGraphModule<StorageType>) throws
    
    // backward direction
    func connect(from from:AnyGraphModule<StorageType>) throws
    
    func forward(input:Tensor<StorageType>?)
    func backward(input:Tensor<StorageType>?, gradOutput:Tensor<StorageType>?)
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
    
    init<M:GraphModule where M.StorageType == S>(_ base:M) {
        self._box = _AnyGraphModuleBox(base)
    }
    
    func connect(to to:AnyGraphModule<S>) throws {
        try _box.connect(to:to)
    }
    
    func connect(from from:AnyGraphModule<S>) throws {
        try _box.connect(from:from)
    }

    
    func forward(input:Tensor<StorageType>?) {
        _box.forward(input)
    }
    
    func backward(input:Tensor<StorageType>?, gradOutput:Tensor<StorageType>?) {
        _box.backward(input, gradOutput: gradOutput)
    }
    
    internal let _box: _AnyGraphModuleBoxBase<S>
}

class _AnyGraphModuleBase {}
class _AnyGraphModuleBoxBase<S:Storage>:
    _AnyGraphModuleBase, GraphModule
{
    internal func forward(input:Tensor<S>?) { _abstract() }
    internal func backward(input:Tensor<S>?, gradOutput:Tensor<S>?) { _abstract() }
    internal func connect(to to:AnyGraphModule<S>) throws { _abstract() }
    internal func connect(from from:AnyGraphModule<S>) throws { _abstract() }

}

class _AnyGraphModuleBox<Base:GraphModule>:
    _AnyGraphModuleBoxBase<Base.StorageType>
{
    internal init(_ base: Base) { self._base = base }
    
    override func forward(input:Tensor<Base.StorageType>?) {
        _base.forward(input)
    }
    
    override func backward(input:Tensor<Base.StorageType>?, gradOutput:Tensor<Base.StorageType>?) {
        _base.backward(input, gradOutput: gradOutput)
    }
    
    internal override func connect(to to:AnyGraphModule<Base.StorageType>) throws {
        try _base.connect(to: to)
    }
    internal override func connect(from from:AnyGraphModule<Base.StorageType>) throws {
        try _base.connect(from: from)
    }
    
    internal var _base: Base
}

func connect<M1, M2 where M1:GraphModule, M2:GraphModule, M1.StorageType == M2.StorageType>(from from:M1, to:M2) throws {
    try from.connect(to: AnyGraphModule<M2.StorageType>(to))
    try to.connect(from: AnyGraphModule<M1.StorageType>(from))
}

class Linear<S:Storage>: GraphModule {
    typealias StorageType = S
    
    var linear:LinearOp<S>
    var grad:LinearGradient<S>
    
    var toModule:AnyGraphModule<S>?
    var fromModule:AnyGraphModule<S>?
    
    var inputShape:Extent? { return Extent(linear.shape[0]) }
    var outputShape:Extent? { return Extent(linear.shape[1]) }
 
    init(numInputs:Int, numOutputs:Int) {
        linear = LinearOp<S>(input_size: numInputs, output_size: numOutputs)
        grad = LinearGradient<S>(input_size: numInputs, output_size: numOutputs)
    }
    
    func connect(to to:AnyGraphModule<S>) throws {
        toModule = to
    }
    
    func connect(from from:AnyGraphModule<S>) throws {
        fromModule = from
    }

    func forward(input:Tensor<S>?) {
        if let module = toModule {
            module.forward(input)
        }
    }
    
    func backward(input:Tensor<S>?, gradOutput:Tensor<S>?) {
        if let module = fromModule {
            module.backward(input, gradOutput: gradOutput)
        }
    }
}

class Variable<S:Storage>: GraphModule {
    typealias StorageType = S
    
    var toModule:AnyGraphModule<S>?
    var fromModule:AnyGraphModule<S>?
    var variable:Tensor<S>
    
    // does not take any input
    var inputShape:Extent? { return Extent(0) }
    
    // output size of variable we're wrapping
    var outputShape:Extent? { return variable.shape }
    
    init(variable:Tensor<S>) {
        self.variable = variable
    }
    
    func connect(to to:AnyGraphModule<S>) throws {
        toModule = to
    }
    
    func connect(from from:AnyGraphModule<S>) throws {
        throw ConnectionError.CannotAcceptInput
    }
    
    func forward(input:Tensor<S>?) {
        // Input nodes cannot take input themselves
        assert(input == nil)
        
        if let module = toModule {
            module.forward(variable)
        }
    }
    
    func backward(input:Tensor<S>?, gradOutput:Tensor<S>?) {
        // TODO: should allow boolean to decide if we back propagate to
        // tensor or not
        if let module = fromModule {
            module.backward(input, gradOutput: gradOutput)
        }
    }
}

class Criterion<S:Storage>: GraphModule {
    typealias StorageType = S
    typealias CostFunction = (input:Tensor<S>, vars:[Tensor<S>]) -> S.ElementType
    
//    var inputShape:Extent? { return fromModule!.outputShape 
//    var outputShape:Extent? { return Extent(0) }
    
    var fromModule:AnyGraphModule<S>?
    var cost:CostFunction
    var value:S.ElementType?
    var vars:[Tensor<S>]
    
    init(cost:CostFunction, vars:[Tensor<S>] = []) {
        self.cost = cost
        self.vars = []
    }

    func connect(to to:AnyGraphModule<S>) throws {
        throw ConnectionError.CannotAcceptOutput
    }
    
    func connect(from from:AnyGraphModule<S>) throws {
        fromModule = from
    }
    
    func forward(input:Tensor<S>?) {
        value = cost(input: input!, vars: vars)
    }
    
    func backward(input:Tensor<S>?, gradOutput:Tensor<S>?) {
        if let module = fromModule {
            module.backward(input, gradOutput: nil)
        }
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