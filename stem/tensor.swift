//
//  tensor.swift
//  stem
//
//  Created by Abe Schneider on 11/10/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation
import Accelerate

//public struct All {}
//public let all = All()

enum TensorError: ErrorType {
    case IllegalOperation
    case SizeMismatch(lhs:Extent, rhs:Extent)
    case IllegalAxis(axis:Int)
}

public protocol TensorIndex {
    var TensorRange: Range<Int> { get }
}

extension Int : TensorIndex {
    public var TensorRange: Range<Int> {
        get {
//            return Range<Int>(start: self, end: self+1)
            return self..<(self+1)
        }
    }
}

extension Range : TensorIndex {
    public var TensorRange: Range<Int> {
        get {
//            return Range<Int>(start: self.startIndex as! Int, end: self.endIndex as! Int)
            // why the cast?
            return (self.startIndex as! Int)..<(self.endIndex as! Int)
        }
    }
}

public struct All : TensorIndex {
    public var TensorRange: Range<Int> {
        get {
//            return Range<Int>(start: 0, end: NSIntegerMax)
            return 0..<0
        }
    }
}

public struct TensorStorageIndex<StorageType:Storage>: GeneratorType {
    var tensor:Tensor<StorageType>
    var indices:[Int]

    init(_ tensor:Tensor<StorageType>) {
        self.tensor = tensor
        indices = [Int](count: tensor.shape.count, repeatedValue: 0)
    }

    public mutating func next() -> Int? {
        let last = tensor.shape.count-1
        if indices[last] >= tensor.shape[last] {
            var d:Int = tensor.shape.count - 1
            while d >= 0 && indices[d] >= tensor.shape[d] {
                // at the end, so return no results left
                if d == 0 { return nil }

                // reset current index
                indices[d] = 0

                // increment next offset
                indices[d-1] += 1

                // go to next dimension
                d -= 1
            }
        }

        // TODO: This is expensive if the next index gives currentOffset+1.
        // Need to figure out a way of shortcuting this calculation when possible.
        let value = tensor.calculateOffset(indices)
        indices[last] += 1
        return value
    }
}

public class Tensor<StorageType:Storage> {
    public typealias ViewType = StorageView<StorageType>
    
    public var storage:StorageType
    
    // defined the bounds set within the storage
    var internalShape:Extent
    
    // offset within storage
    var offset:Int
    
    // external shape
    public var shape:Extent {
        get { return view.shape }
    }
    
    // view into storage
    public var view:ViewType
    
    // order to traverse the dimensions
    public var dimIndex:[Int]
    
    // step size to increment within storage for each dimension
    public var stride:[Int]
    
    public var transposed:Bool
    
    public init(array:[StorageType.ElementType], shape:Extent, offset:Int?=nil) {
        storage = StorageType(array: array)
        internalShape = shape
        self.stride = storage.calculateStride(shape)
        dimIndex = Tensor.calculateOrder(shape.count)

        if let o = offset {
            self.offset = o
        } else {
            self.offset = 0
        }
        
        view = ViewType(shape: shape, offset: Array<Int>(count: shape.count, repeatedValue: 0))
        transposed = false
    }
    
    public init(storage:StorageType, shape:Extent, view:StorageView<StorageType>?=nil, offset:Int?=nil) {
        self.storage = storage
        internalShape = shape
        self.stride = storage.calculateStride(shape)
        dimIndex = Tensor.calculateOrder(shape.count)
        
        if let o = offset {
            self.offset = o
        } else {
            self.offset = 0
        }

        if let v = view {
            self.view = v
        } else {
            self.view = ViewType(shape: shape, offset: Array<Int>(count: shape.count, repeatedValue: 0))
        }
        
        transposed = false
    }
    
    public init(shape:Extent) {
        storage = StorageType(size: shape.elements)
        internalShape = shape
        offset = 0
        self.stride = storage.calculateStride(shape)
        dimIndex = Tensor.calculateOrder(shape.count)

        view = ViewType(shape: shape, offset: Array<Int>(count: shape.count, repeatedValue: 0))
        transposed = false
    }
    
    public init(_ tensor:Tensor, window:[Range<Int>]) {
        storage = tensor.storage
        internalShape = tensor.internalShape
        offset = 0
        stride = tensor.stride

        let viewShape = Extent(window.map { $0.last! - $0.first! + 1 })
        view = ViewType(shape: viewShape, offset: window.map { $0.first! })
        dimIndex = (0..<tensor.internalShape.count).map { tensor.internalShape.count-$0-1 }
        transposed = false
    }
    
    public init(_ tensor:Tensor, dimIndex:[Int]?=nil, view:StorageView<StorageType>?=nil, copy:Bool=false) {
        if copy {
            storage = tensor.storage
        } else {
            storage = tensor.storage
        }
        
        internalShape = tensor.internalShape
        offset = 0
        stride = tensor.stride
        
        if let d = dimIndex {
            self.dimIndex = d
        } else {
            self.dimIndex = tensor.dimIndex
        }
        
        if let v = view {
            self.view = v
        } else {
            self.view = ViewType(shape: tensor.shape, offset: tensor.view.offset)
        }
        
        transposed = false
    }

    static func calculateOrder(dims:Int) -> [Int] {
        return (0..<dims).map { dims-$0-1 }
    }
    
    func calculateOffset() -> Int {
        var pos = offset
        for i in 0..<shape.count {
            pos += view.offset[dimIndex[i]]*stride[i]
        }
        
        return pos
    }

    func calculateOffset(indices:[Int]) -> Int {
        var pos = offset
        for i in 0..<indices.count {
            pos += (indices[dimIndex[i]]+view.offset[dimIndex[i]])*stride[i]
        }
        
        return pos
    }
    
    public subscript(indices:[Int]) -> StorageType.ElementType {
        get { return storage[calculateOffset(indices)] }
        set { storage[calculateOffset(indices)] = newValue }
    }
    
    public subscript(indices:Int...) -> StorageType.ElementType {
        get { return storage[calculateOffset(indices)] }
        set { storage[calculateOffset(indices)] = newValue }
    }
    
    public subscript(ranges:[TensorIndex]) -> Tensor {
        get {
            return Tensor(self, window: ranges.map { $0.TensorRange })
        }
        set {
            let view = Tensor(self, window: ranges.map { $0.TensorRange })
            copy(from: newValue, to: view)
        }
    }
    
    public subscript(ranges:TensorIndex...) -> Tensor {
        get { return self[ranges] }
        set {
            let view = Tensor(self, window: ranges.map { $0.TensorRange })
            copy(from: newValue, to: view)
        }
    }
    
    public func transpose() -> Tensor<StorageType> {
        let newDimIndex = Array(dimIndex.reverse())
        let newShape = Extent(view.shape.reverse())
        let newOffset = Array(view.offset.reverse())
        let newView = StorageView<StorageType>(shape: newShape, offset: newOffset)
        return Tensor(self, dimIndex: newDimIndex, view: newView)
    }
    
    public func reshape(newShape:Extent) -> Tensor {
        // verify the total number of elements is conserved
        assert(newShape.elements == internalShape.elements)
        return Tensor(storage: storage, shape: newShape)
    }
    
    // generates indices of view in storage
    public func storageIndices() -> GeneratorSequence<TensorStorageIndex<StorageType>> {
        return GeneratorSequence<TensorStorageIndex<StorageType>>(TensorStorageIndex<StorageType>(self))
    }
}

extension Tensor {
    // general case
    private func elementToString(v:StorageType.ElementType) -> String {
        return String(v)
    }
    
    private func elementToString(v:Double) -> String {
        return String(format: "%2.3f", v)
    }
    
    private func elementToString(v:Float) -> String {
        return String(format: "%2.3f", v)
    }

    private func elementToString(v:Int) -> String {
        return String(format: "%d", v)
    }
    
    private func convertToString(var indices:[Int], dim:Int) -> String {
        if dim == internalShape.count-1 {
            // last dimension, convert values to string
            let values:[String] = (0..<shape[dim]).map({(i:Int) -> String in
                indices[dim] = i
                return elementToString(self[indices])
            })
            return "[\(values.joinWithSeparator(",\t"))]"
        } else {
            let values:[String] = (0..<shape[dim]).map({(i:Int) -> String in
                indices[dim] = i
                
                var indent:String
                if i > 0 {
                    indent = String(count: dim+1, repeatedValue: " " as Character)
                } else {
                    indent = ""
                }
                
                return "\(indent)\(convertToString(indices, dim: dim+1))"
            })
            return "[\(values.joinWithSeparator("\n"))]"
        }
    }
}

extension Tensor: CustomStringConvertible {
    public var description: String {
        get {
            let indices = (0..<internalShape.count).map { _ in 0 }
            return convertToString(indices, dim: 0)
        }
    }
}

func copy<StorageType:Storage>(from from:[[StorageType.ElementType]], to:Matrix<StorageType>) {
//    assert(to.shape == from.shape)

    var toIndices = to.storageIndices()
    for i in 0..<from.count {
        for j in 0..<from[i].count {
            to.storage[toIndices.next()!] = from[i][j]
        }
    }
//    let zippedIndices = zip(from.storageIndices(), to.storageIndices())
//    for (i, j) in zippedIndices {
//        to.storage[j] = from.storage[i]
//    }
}

func copy<StorageType:Storage>(from from:Tensor<StorageType>, to:Tensor<StorageType>) {
    assert(to.shape == from.shape)
    
    let zippedIndices = zip(from.storageIndices(), to.storageIndices())
    for (i, j) in zippedIndices {
        to.storage[j] = from.storage[i]
    }
}

public func copy<StorageType>(tensor:Tensor<StorageType>) -> Tensor<StorageType> {
    return Tensor<StorageType>(tensor, copy: true)
}

func fill<StorageType:Storage>(tensor:Tensor<StorageType>, value:StorageType.ElementType) {
    for i in tensor.storageIndices() {
        tensor.storage[i] = value
    }
}

// concats two tensors along the given axis (0: rows, 1: cols, etc.)
func concat<StorageType:Storage>(tensor1:Tensor<StorageType>, _ tensor2:Tensor<StorageType>, axis:Int=0) throws -> Tensor<StorageType> {
    // verify other dimensions match
    let maxDims = max(tensor1.shape.count, tensor2.shape.count)
    
    if axis >= maxDims {
        throw TensorError.IllegalAxis(axis: axis)
    }
    
    for i in 0..<maxDims {
        if (i != axis) {
            if tensor1.shape[i] != tensor2.shape[i] {
                throw TensorError.SizeMismatch(lhs: tensor1.shape, rhs: tensor2.shape)
            }
        }
    }
    
    var shape = tensor1.shape
    shape[axis] += tensor2.shape[axis]
    
    let result = Tensor<StorageType>(shape: shape)
    var rpos = result.storageIndices()
    
    for pos in tensor1.storageIndices() {
        result.storage[rpos.next()!] = tensor1.storage[pos]
    }
    
    for pos in tensor2.storageIndices() {
        result.storage[rpos.next()!] = tensor2.storage[pos]
    }
    
    return result
}

infix operator ⊕ { associativity left precedence 140 }
public func ⊕<StorageType:Storage>(tensor1:Tensor<StorageType>, tensor2:Tensor<StorageType>) throws -> Tensor<StorageType> {
    return try concat(tensor1, tensor2)
}

func concat<StorageType:Storage>(tensor1:Tensor<StorageType>, _ tensor2:Tensor<StorageType>, _ tensor3:Tensor<StorageType>, _ rest:Tensor<StorageType>..., axis:Int=0) throws -> Tensor<StorageType> {
    var result = try concat(tensor1, tensor2, axis: axis)
    result = try concat(result, tensor3, axis:axis)
    for i in 0..<rest.count {
        result = try concat(result, rest[i], axis: axis)
    }
    
    return result
}

func concat<StorageType:Storage>(tensors:[Tensor<StorageType>], axis:Int=0) throws -> Tensor<StorageType> {
    var result = try concat(tensors[0], tensors[1], axis: axis)
    for i in 2..<tensors.count {
        result = try concat(result, tensors[i], axis: axis)
    }
    
    return result
}

func vstack<StorageType:Storage>(tensor1:Tensor<StorageType>, _ tensor2:Tensor<StorageType>) throws -> Tensor<StorageType> {
    return try concat(tensor1, tensor2, axis: 0)
}

func hstack<StorageType:Storage>(tensor1:Tensor<StorageType>, _ tensor2:Tensor<StorageType>) throws -> Tensor<StorageType> {
    return try concat(tensor1, tensor2, axis: 1)
}

func map<StorageType:Storage>(
    tensor:Tensor<StorageType>,
    fn:(StorageType.ElementType) -> StorageType.ElementType) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(shape: tensor.shape)
    for i in tensor.storageIndices() {
        result.storage[i] = fn(tensor.storage[i])
    }
    
    return result
}
