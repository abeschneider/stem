//
//  tensor.swift
//  stem
//
//  Created by Abe Schneider on 11/10/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation
import Accelerate

struct IllegalOperation: ErrorType {}

public protocol TensorIndex {
    var TensorRange: Range<Int> { get }
}

extension Int : TensorIndex {
    public var TensorRange: Range<Int> {
        get {
            return Range<Int>(start: self, end: self+1)
        }
    }
}

extension Range : TensorIndex {
    public var TensorRange: Range<Int> {
        get {
            return Range<Int>(start: self.startIndex as! Int, end: self.endIndex as! Int)
        }
    }
}

public struct TensorStorageIndex<StorageType:Storage>: GeneratorType {
    var tensor:Tensor<StorageType>
    var indices:[Int]

    init(_ tensor:Tensor<StorageType>) {
        self.tensor = tensor
        indices = [Int](count: tensor.shape.dims, repeatedValue: 0)
    }

    public mutating func next() -> Int? {
        let last = tensor.shape.dims-1
        if indices[last] >= tensor.shape[last] {
            var d:Int = tensor.shape.dims - 1
            while d >= 0 && indices[d] >= tensor.shape[d] {
                // at the end, so return no results left
                if d == 0 { return nil }

                // reset current index
                indices[d] = 0

                // increment next offset
                ++indices[d-1]

                // go to next dimension
                --d
            }
        }

        // TODO: This is expensive if the next index gives currentOffset+1.
        // Need to figure out a way of shortcuting this calculation when possible.
        let value = tensor.calculateOffset(indices)
        ++indices[last]
        return value
    }
}


// TODO: can this can be defined as parameterized by Storage, and
// delegated by view? Otherwise, two Tensors with different view types
// will also be different Tensor types
public class Tensor<StorageType:Storage> {
    public typealias ViewType = StorageView<StorageType>
    
    public var storage:StorageType
    
    // reshape affects this
    // TODO: need to make this into a function-var to allow
    // dimIndex to affect this.. or some aux. variable
    var internalShape:Extent
    
    // offset within storage
    var offset:Int
    
    // external shape
    var shape:Extent {
        get { return view.shape }
    }
    
    // view into storage
    public var view:ViewType
    
    // order to traverse the dimensions
    public var dimIndex:[Int]
    
    // step size to increment within storage for each dimension
    public var stride:[Int]

    
    // forward shape from view
//    public var shape:Extent { return view.shape }
    public var transposed:Bool
    
    public init(array:[StorageType.ElementType], shape:Extent, offset:Int?=nil) {
        storage = StorageType(array: array)
        internalShape = shape
        self.stride = storage.calculateStride(shape)
        dimIndex = Tensor.calculateOrder(shape.dims)

        if let o = offset {
            self.offset = o
        } else {
            self.offset = 0
        }
        
        view = ViewType(shape: shape, offset: Array<Int>(count: shape.dims, repeatedValue: 0))
        transposed = false
    }
    
    public init(storage:StorageType, shape:Extent, view:StorageView<StorageType>?=nil, offset:Int?=nil) {
        self.storage = storage
        internalShape = shape
        self.stride = storage.calculateStride(shape)
        dimIndex = Tensor.calculateOrder(shape.dims)
        
        if let o = offset {
            self.offset = o
        } else {
            self.offset = 0
        }

        if let v = view {
            self.view = v
        } else {
            self.view = ViewType(shape: shape, offset: Array<Int>(count: shape.dims, repeatedValue: 0))
        }
        
        transposed = false
    }
    
    public init(shape:Extent) {
        storage = StorageType(size: shape.elements)
        internalShape = shape
        offset = 0
        self.stride = storage.calculateStride(shape)
        dimIndex = Tensor.calculateOrder(shape.dims)

        view = ViewType(shape: shape, offset: Array<Int>(count: shape.dims, repeatedValue: 0))
        transposed = false
    }
    
    public init(_ tensor:Tensor, window:[Range<Int>]) {
        storage = tensor.storage
        internalShape = tensor.internalShape
        offset = 0
        stride = tensor.stride

        let viewShape = Extent(window.map { $0.last! - $0.first! + 1 })
        view = ViewType(shape: viewShape, offset: window.map { $0.first! })
        dimIndex = (0..<tensor.internalShape.dims).map { tensor.internalShape.dims-$0-1 }
        transposed = false
    }
    
    public init(_ tensor:Tensor, dimIndex:[Int]?=nil, view:StorageView<StorageType>?=nil) {
        storage = tensor.storage
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
        for i in 0..<shape.dims {
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
    }
    
    public subscript(ranges:TensorIndex...) -> Tensor {
        get { return self[ranges] }
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
    private func convertToString(var indices:[Int], dim:Int) -> String {
        if dim == internalShape.dims-1 {
            // last dimension, convert values to string
            let values:[String] = (0..<shape[dim]).map({(i:Int) -> String in
                indices[dim] = i
                return String(format: "%2.3f", self[indices] as! Double)
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
            let indices = (0..<internalShape.dims).map { _ in 0 }
            return convertToString(indices, dim: 0)
        }
    }
}

func copy<StorageType:Storage>(source:Tensor<StorageType>, _ destination:Tensor<StorageType>) {
    assert(destination.shape == source.shape)
    for i in source.storageIndices() {
        destination.storage[i] = source.storage[i]
    }
}

func fill<StorageType:Storage>(tensor:Tensor<StorageType>, value:StorageType.ElementType) {
    for i in tensor.storageIndices() {
        tensor.storage[i] = 0
    }
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
