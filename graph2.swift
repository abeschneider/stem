//
//  graph2.swift
//  stem
//
//  Created by Abe Schneider on 2/2/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation

enum ConnectionError: ErrorType {
    case CannotAcceptInput
    case CannotAcceptOutput
}

protocol GraphModule {
    associatedtype StorageType:Storage
    
    // If nil, then module has no preferred shape
    //    var inputShape:Extent? { get }
    //    var outputShape:Extent? { get }
    var output:Tensor<StorageType>? { get set }
    
    func forward(input:Tensor<StorageType>?) -> Tensor<StorageType>
    func backward(gradOutput:Tensor<StorageType>?) -> Tensor<StorageType>
    
    // returns the output shape for a given input shape
    func outputShape(inputShape:Extent) -> Extent
}

@noreturn @inline(never)
internal func _abstract(file: StaticString = #file, line: UInt = #line) {
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
    
    init<M:GraphModule where M.StorageType==S>(_ base:M) {
        self._box = _AnyGraphModuleBox(base)
    }
    
    func forward(input:Tensor<StorageType>?=nil) -> Tensor<StorageType> {
        return _box.forward(input)
    }
    
    func backward(gradOutput:Tensor<StorageType>?=nil) -> Tensor<StorageType> {
        return _box.backward(gradOutput)
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
    
    internal func forward(input:Tensor<S>?=nil) -> Tensor<S> { _abstract() }
    internal func backward(gradOutput:Tensor<S>?) -> Tensor<S> { _abstract() }
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
    
    override func forward(input:Tensor<Base.StorageType>?=nil) -> Tensor<Base.StorageType> {
        return _base.forward(input)
    }
    
    override func backward(gradOutput:Tensor<Base.StorageType>?=nil) -> Tensor<Base.StorageType> {
        return _base.backward(gradOutput)
    }
    
    internal override func outputShape(inputShape: Extent) -> Extent {
        return _base.outputShape(inputShape)
    }
    
    internal var _base: Base
}


protocol Container {
    associatedtype StorageType
    func add<M:GraphModule where M.StorageType==StorageType>(module:M)
}

class Sequence<S:Storage>: GraphModule, Container {
    typealias StorageType = S
    
    var output:Tensor<S>?
    var gradInput:Tensor<S>?
    var modules:[AnyGraphModule<S>] = []

    func forward(input:Tensor<S>?) -> Tensor<S> {
        output = modules[0].forward(input!)
        for i in 1..<modules.count {
            output = modules[i].forward(output)
        }
        
        return output!
    }
    
    func backward(gradOutput: Tensor<StorageType>?) -> Tensor<S> {
        gradInput = modules.last!.backward(gradOutput!)
        for i in (1..<modules.count).reverse() {
            gradInput = modules[i].backward(gradInput)
        }
        
        return gradInput!
    }
    
    func outputShape(inputShape: Extent) -> Extent {
        return modules.last!.outputShape(inputShape)
    }
    
    func add<M:GraphModule where M.StorageType==S>(module:M) {
        modules.append(AnyGraphModule<S>(module))
    }
}

// takes N copies and concats them into a single input
class Concat<S:Storage>: GraphModule, Container {
    typealias StorageType = S
    
    var output:Tensor<S>?
    var gradInput:Tensor<S>?
    var modules:[AnyGraphModule<S>] = []
    var moduleShapes:[Extent] = []
    
    func forward(input:Tensor<S>?) -> Tensor<S> {
        let outputs = modules.map { $0.forward(input) }
        moduleShapes = outputs.map { $0.shape }
        output = try! concat(outputs)
        return output!
    }
    
    func backward(gradOutput: Tensor<StorageType>?) -> Tensor<S> {
        // use outputShape for each module to figure out which slice of gradOutput to use
        // and apply to each module
        var start = 0
        var end = 0
        for (i, module) in modules.enumerate() {
            end += moduleShapes[i][0]
            gradInput![start..<end] = module.backward(gradOutput![start..<end])
            start += moduleShapes[i][0]
        }
        return gradInput!
    }
    
    func addInput<M:GraphModule where M.StorageType==S>(module:M) {
        modules.append(AnyGraphModule<S>(module))
    }
    
    func outputShape(inputShape: Extent) -> Extent {
        // size is the result of all the modules outputShape added together
        return Extent(0)
    }
    
    func add<M:GraphModule where M.StorageType==S>(module:M) {
        // need to verify output dimensions make sense
        modules.append(AnyGraphModule<S>(module))
    }
}


// takes a single input, and forwards it to each of its children
//class Split<S:Storage>: GraphModule, Container {
//    typealias StorageType = S
//    
//    var output:Tensor<S>?
//    var gradInput:Tensor<S>?
//    var modules:[AnyGraphModule<S>] = []
//
//    func forward(input:Tensor<S>?) -> Tensor<S> {
//        // calculate shape
//        var size = Extent(input!.shape)
//        let moduleShapes = modules.map { $0.outputShape(input!.shape)[0] }
//        size[1] = moduleShapes.reduce(0, combine: +)
//        
//        var start = 0
//        var end = 0
//        for module in modules {
//            let moduleShape = module.outputShape(input!.shape)
//            end += moduleShape[0]
//            // copy output
////            output![start..<end] = 
//            module.forward(input[start..<end])
//            start += moduleShape[0]
//        }
//        
//        output = input!
//        return output!
//    }
//    
//    func backward(gradOutput: Tensor<StorageType>?) -> Tensor<S> {
//        // copy slices of gradOutput and call backward per module
//    }
//
//    func outputShape(inputShape: Extent) -> Extent {
//    }
//    
//    func add<M:GraphModule where M.StorageType==S>(module:M) {
//    }
//}
