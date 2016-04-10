//
//  tensor.swift
//  stem
//
//  Created by Abe Schneider on 11/10/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation
import Accelerate

infix operator ⊕ { associativity left precedence 140 }

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
        get { return self..<(self+1) }
    }
}

extension Range : TensorIndex {
    public var TensorRange: Range<Int> {
        get {
            // In theory we should be able to constrain Range.Element to
            // Int, but currently Swift doesn't allow constraints to go
            // down to a single type nor does it allow constraints for
            // extensions conforming to a protocol. Thus, we need to
            // manually cast to an Int here.
            return (self.startIndex as! Int)..<(self.endIndex as! Int)
        }
    }
}

public let all:Range = 0..<0

public struct IndexGenerator: GeneratorType {
    var indices:[Int]
    var shape:Extent
    var dimIndex:[Int]

    public init(_ shape:Extent, dimIndex:[Int]?=nil) {
        self.shape = shape
        indices = [Int](count: shape.count, repeatedValue: 0)
        
        if let dims = dimIndex {
            self.dimIndex = dims
        } else {
            self.dimIndex = (0..<shape.count).map { $0 }
        }
    }
    
    public mutating func next() -> [Int]? {
        if indices[dimIndex[0]] >= shape[dimIndex[0]] {
            var d:Int = 0
            
            // loop until we no longer overflow
            while d <= shape.count && indices[dimIndex[d]] >= shape[dimIndex[d]] {
                // at the end, so return no results left
                if d == dimIndex.count-1 { return nil }
                
                // reset current index
                indices[dimIndex[d]] = 0
                
                // increment next offset
                indices[dimIndex[d+1]] += 1
                
                // go to next dimension
                d += 1
            }
        }
        
        let value = indices
        indices[dimIndex[0]] += 1
        return value
    }
}

// TODO: remove, IndexGenerator supersedes this (+ tensor.calculateOffset)
//public struct TensorStorageIndex<StorageType:Storage>: GeneratorType {
//    var tensor:Tensor<StorageType>
//    var indices:[Int]
//    var index:Int
//    var last:Int
//
//    init(_ tensor:Tensor<StorageType>) {
//        self.tensor = tensor
//        indices = [Int](count: tensor.shape.count, repeatedValue: 0)
//        index = 0
//        last = tensor.shape.count-1
//    }
//
//    public mutating func next() -> Int? {
//        if indices[last] >= tensor.shape[last] {
//            var d:Int = tensor.shape.count - 1
//            
//            // loop until we no longer overflow
//            while d >= 0 && indices[d] >= tensor.shape[d] {
//                // at the end, so return no results left
//                if d == 0 { return nil }
//
//                // reset current index
//                indices[d] = 0
//
//                // increment next offset
//                indices[d-1] += 1
//
//                // go to next dimension
//                d -= 1
//            }
//        }
//
//        let value = tensor.calculateOffset(indices)
//        indices[last] += 1
//        
//        return value
//    }
//}

public protocol TensorType {
    var shape:Extent { get }
}

public class TensorScalar<StorageType:Storage>: TensorType {
    public var shape:Extent { return Extent(1) }
    public var value:StorageType.ElementType
    
    public init(_ value:StorageType.ElementType) {
        self.value = value
    }
    
    public subscript(indices:[Int]) -> StorageType.ElementType {
        get {
            precondition(indices.count == 1)
            precondition(indices[0] == 0)
            return value
        }
        set {
            precondition(indices.count == 1)
            precondition(indices[0] == 0)
            value = newValue
        }
    }
    
    public subscript(indices:Int) -> StorageType.ElementType {
        get {
            precondition(indices == 0)
            return value
        }
        set {
            precondition(indices == 0)
            value = newValue
        }
    }
}

extension Float {
    init<StorageType:Storage where StorageType.ElementType:FloatNumericType>(_ scalar:TensorScalar<StorageType>) {
        // TODO: this is dangerous
        self.init(scalar.value as! Float)
    }
}

public class Tensor<StorageType:Storage>: TensorType {
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
    
    // convenience accessor to generate a transposed view
    public var T:Tensor<StorageType> {
        get { return transpose() }
    }
    
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
    }
    
    public init(shape:Extent) {
        storage = StorageType(size: shape.elements)
        internalShape = shape
        offset = 0
        self.stride = storage.calculateStride(shape)
        dimIndex = Tensor.calculateOrder(shape.count)

        view = ViewType(shape: shape, offset: Array<Int>(count: shape.count, repeatedValue: 0))
    }
    
    public init(_ tensor:Tensor, window:[Range<Int>]) {
        storage = tensor.storage
        internalShape = tensor.internalShape
        offset = 0
        stride = tensor.stride

        let viewShape = Extent(window.enumerate().map {
            if $0.1.first == nil || $0.1.last == nil {
                return tensor.shape[$0.0]
            }
            
            return $0.1.last! - $0.1.first! + 1
        })
        
        view = ViewType(shape: viewShape, offset: window.map { $0.first != nil ? $0.first! : 0})
        dimIndex = (0..<tensor.internalShape.count).map { tensor.internalShape.count-$0-1 }
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
    }

    static func calculateOrder(dims:Int) -> [Int] {
        return (0..<dims).map { dims-$0-1 }
    }
    
    public func calculateOffset() -> Int {
        var pos = offset
        for i in 0..<shape.count {
            pos += view.offset[dimIndex[i]]*stride[i]
        }
        
        return pos
    }

    public func calculateOffset(indices:[Int]) -> Int {
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
        precondition(newShape.elements == internalShape.elements, "Cannot change number of elements in Tensor.")
        
        return Tensor(storage: storage, shape: newShape)
    }
    
    // generates indices of view in storage
//    public func storageIndices() -> GeneratorSequence<TensorStorageIndex<StorageType>> {
//        return GeneratorSequence<TensorStorageIndex<StorageType>>(TensorStorageIndex<StorageType>(self))
//    }
    public func indices() -> GeneratorSequence<IndexGenerator> {
        let dimOrder = (0..<shape.count).map { shape.count-$0-1 }
        return GeneratorSequence<IndexGenerator>(IndexGenerator(shape, dimIndex: dimOrder)) //, dimIndex: dimIndex))
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
    
    private func convertToString(indices:[Int], dim:Int) -> String {
        var idx = indices
        
        if dim == internalShape.count-1 {
            // last dimension, convert values to string
            let values:[String] = (0..<shape[dim]).map({(i:Int) -> String in
                idx[dim] = i
                return elementToString(self[idx])
            })
            return "[\(values.joinWithSeparator(",\t"))]"
        } else {
            let values:[String] = (0..<shape[dim]).map({(i:Int) -> String in
                idx[dim] = i
                
                var indent:String
                if i > 0 {
                    indent = String(count: dim+1, repeatedValue: " " as Character)
                } else {
                    indent = ""
                }
                
                return "\(indent)\(convertToString(idx, dim: dim+1))"
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

func copy<StorageType:Storage>(from from:[[StorageType.ElementType]], to:Matrix<StorageType>)  {
    precondition(to.shape[0] != from.count || to.shape[1] != from[0].count,
                 "Destination and source must be the same size")

    var toIndices = to.indices()
    for i in 0..<from.count {
        for j in 0..<from[i].count {
            to[toIndices.next()!] = from[i][j]
        }
    }
}

func copy<StorageType:Storage>(from from:Tensor<StorageType>, to:Tensor<StorageType>) {
    precondition(to.shape == from.shape, "Destination and source must be the same size")
    
    let zippedIndices = zip(from.indices(), to.indices())
    for (i, j) in zippedIndices {
        to[j] = from[i]
    }
}

public func copy<StorageType>(tensor:Tensor<StorageType>) -> Tensor<StorageType> {
    return Tensor<StorageType>(tensor, copy: true)
}

func fill<StorageType:Storage>(tensor:Tensor<StorageType>, value:StorageType.ElementType) {
    for i in tensor.indices() {
        tensor[i] = value
    }
}

// concats two tensors along the given axis (0: rows, 1: cols, etc.)
func concat<StorageType:Storage>(tensor1:Tensor<StorageType>, _ tensor2:Tensor<StorageType>, axis:Int=0) -> Tensor<StorageType> {
    // verify other dimensions match
    let maxDims = max(tensor1.shape.count, tensor2.shape.count)
    
//    if axis >= maxDims {
//        throw TensorError.IllegalAxis(axis: axis)
//    }
    precondition(axis < maxDims, "Axis is greater than number of dimensions")
    
    for i in 0..<maxDims {
        if (i != axis) {
            if tensor1.shape[i] != tensor2.shape[i] {
//                throw TensorError.SizeMismatch(lhs: tensor1.shape, rhs: tensor2.shape)
                precondition(tensor1.shape[i] == tensor2.shape[i],
                             "Dimensions of tensors do not match")
            }
        }
    }
    
    var shape = tensor1.shape
    shape[axis] += tensor2.shape[axis]
    
    let result = Tensor<StorageType>(shape: shape)
    var rpos = result.indices()
    
    for pos in tensor1.indices() {
//        result.storage[rpos.next()!] = tensor1.storage[pos]
        result[rpos.next()!] = tensor1[pos]
    }
    
//    for pos in tensor2.storageIndices() {
    for pos in tensor2.indices() {
//        result.storage[rpos.next()!] = tensor2.storage[pos]
        result[rpos.next()!] = tensor2[pos]
    }
    
    return result
}

public func ⊕<StorageType:Storage>(tensor1:Tensor<StorageType>, tensor2:Tensor<StorageType>)  -> Tensor<StorageType> {
    return concat(tensor1, tensor2)
}

func concat<StorageType:Storage>(tensor1:Tensor<StorageType>, _ tensor2:Tensor<StorageType>, _ tensor3:Tensor<StorageType>, _ rest:Tensor<StorageType>..., axis:Int=0) -> Tensor<StorageType> {
    var result = concat(tensor1, tensor2, axis: axis)
    result = concat(result, tensor3, axis:axis)
    for i in 0..<rest.count {
        result = concat(result, rest[i], axis: axis)
    }
    
    return result
}

func concat<StorageType:Storage>(tensors:[Tensor<StorageType>], axis:Int=0)
    -> Tensor<StorageType>
{
    var result = concat(tensors[0], tensors[1], axis: axis)
    for i in 2..<tensors.count {
        result = concat(result, tensors[i], axis: axis)
    }
    
    return result
}

func vstack<StorageType:Storage>(tensor1:Tensor<StorageType>, _ tensor2:Tensor<StorageType>)
    -> Tensor<StorageType>
{
    return concat(tensor1, tensor2, axis: 0)
}

func hstack<StorageType:Storage>(tensor1:Tensor<StorageType>, _ tensor2:Tensor<StorageType>)
    -> Tensor<StorageType>
{
    return concat(tensor1, tensor2, axis: 1)
}

func map<StorageType:Storage>(
    tensor:Tensor<StorageType>,
    fn:(StorageType.ElementType) -> StorageType.ElementType) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(shape: tensor.shape)
    for i in tensor.indices() {
        result[i] = fn(tensor[i])
    }
    
    return result
}
