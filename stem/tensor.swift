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

public enum TensorType {
    case Tensor
    case Vector
    case RowVector
    case ColumnVector
    case Matrix
    case Cube
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
    
    public init(_ shape:Extent, order:DimensionOrder) {
        self.shape = shape
        indices = [Int](count: shape.count, repeatedValue: 0)

        switch order {
        case .ColumnMajor:
            dimIndex = (0..<shape.count).map { shape.count-$0-1 }
            break
        case .RowMajor:
            dimIndex = (0..<shape.count).map { $0 }
            break
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

// returns blocks of indices .. blocks are from the major axis
//public struct BlockIndexGenerator: GeneratorType {
//    var indices:[Int]
//    var shape:Extent
//    var dimIndex:[Int]
//    
//    public init(_ shape:Extent, dimIndex:[Int]?=nil) {
//        self.shape = shape
//        indices = [Int](count: shape.count, repeatedValue: 0)
//        
//        if let dims = dimIndex {
//            self.dimIndex = dims
//        } else {
//            self.dimIndex = (0..<shape.count).map { $0 }
//        }
//    }
//    
//    public init(_ shape:Extent, order:DimensionOrder) {
//        self.shape = shape
//        indices = [Int](count: shape.count, repeatedValue: 0)
//        
//        switch order {
//        case .ColumnMajor:
//            dimIndex = (0..<shape.count).map { shape.count-$0-1 }
//            break
//        case .RowMajor:
//            dimIndex = (0..<shape.count).map { $0 }
//            break
//        }
//    }
//    
//    public mutating func next() -> [[Int]]? {
//        for i in 0..<shape[dimIndex[0]] {
//            
//        }
//    }
//}

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
        set {
            precondition(view.shape.elements == newValue.elements, "Number of elements must match")
            view.shape = newValue
        }
    }
    
    // view into storage
    public var view:ViewType
    
    // order to traverse the dimensions
    public var dimIndex:[Int]
    
    // step size to increment within storage for each dimension
    public var stride:[Int]
    
    // TODO: look into determining this at initialization
    public var type:TensorType {
        if shape.span == 1 {
            if shape[0] > 1 {
                return .RowVector
            } else {
                return .ColumnVector
            }
        } else if shape.span == 2 {
            return .Matrix
        } else if shape.span == 3 {
            return .Cube
        }
        
        return .Tensor
    }
    
    // convenience accessor to generate a transposed view
    public var T:Tensor<StorageType> {
        get { return transpose() }
    }
    
    /**
     Creates a vector along a specified axis.
     
     - Parameter array: contents of array
     - Parameter axis: axis vector lies on
 
     */
    public convenience init(_ array:[StorageType.ElementType], axis:Int=0) {
        var shapeValues = [Int](count: axis+1, repeatedValue: 1)
        shapeValues[axis] = array.count
        let shape = Extent(shapeValues)
        
        self.init(array: array, shape: shape)
    }
    
    /**

     Creates a row vector
     
     - Parameter rowvector: contents of array
     */
    public convenience init(rowvector array:[StorageType.ElementType]) {
        let cols = array.count
        let shape = Extent(1, cols)
        
        self.init(array: array, shape: shape)
    }
    
    public convenience init(colvector array:[StorageType.ElementType]) {
        let rows = array.count
        let shape = Extent(rows, 1)

        self.init(array: array, shape: shape)
    }

    public convenience init(_ array:[[StorageType.ElementType]]) {
        let rows = array.count
        let cols = array[0].count
        
        self.init(Extent(rows, cols))
        
        var index = indices(.ColumnMajor)
        for i in 0..<rows {
            for j in 0..<cols {
                self[index.next()!] = array[i][j]
            }
        }
    }
    
    public init(_ shape:Extent, value:StorageType.ElementType=0) {
        storage = StorageType(size: shape.elements, value: value)
        internalShape = shape
        offset = 0
        self.stride = calculateStride(Extent(storage.calculateOrder(shape.dims)))
        dimIndex = storage.calculateOrder(shape.count)
        
        view = ViewType(shape: shape, offset: Array<Int>(count: shape.count, repeatedValue: 0))
    }
    
    public init(_ shape:Extent, storage:StorageType, offset:Int=0) {
//        storage = StorageType(size: shape.elements, value: value)
        self.storage = StorageType(storage: storage, offset: offset)
        internalShape = shape
        self.offset = 0
        self.stride = calculateStride(Extent(storage.calculateOrder(shape.dims)))
        dimIndex = storage.calculateOrder(shape.count)
        
        view = ViewType(shape: shape, offset: Array<Int>(count: shape.count, repeatedValue: 0))
    }
    
    init(array:[StorageType.ElementType], shape:Extent, offset:Int?=nil) {
        storage = StorageType(array: array)
        internalShape = shape
        self.stride = calculateStride(Extent(storage.calculateOrder(shape.dims)))
        dimIndex = storage.calculateOrder(shape.count)

        if let o = offset {
            self.offset = o
        } else {
            self.offset = 0
        }
        
        view = ViewType(shape: shape, offset: Array<Int>(count: shape.count, repeatedValue: 0))
    }
    
    init(storage:StorageType, shape:Extent, view:StorageView<StorageType>?=nil, offset:Int?=nil) {
        self.storage = storage
        internalShape = shape
        self.stride = calculateStride(Extent(storage.calculateOrder(shape.dims)))
        dimIndex = storage.calculateOrder(shape.count)
        
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
    
    init(_ tensor:Tensor, window:[Range<Int>]) {
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
        dimIndex = tensor.storage.calculateOrder(viewShape.count)
    }
    
    init(_ tensor:Tensor, dimIndex:[Int]?=nil, view:StorageView<StorageType>?=nil, stride: [Int]?=nil, copy:Bool=false) {
        if copy {
//            storage = tensor.storage
            storage = StorageType(size: tensor.shape.elements, value: 0)
            for i in 0..<tensor.shape.elements {
                storage[i] = tensor.storage[i]
            }
        } else {
            storage = tensor.storage
        }
        
        internalShape = tensor.internalShape
        offset = 0
        
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
        
        if let s = stride {
            self.stride = s
        } else {
            self.stride = tensor.stride
        }
    }
    
    init(tensor:Tensor, shape:Extent, stride:[Int]) {
        storage = tensor.storage
        internalShape = shape
        offset = 0
        self.stride = stride
        
        // check if we need to increase the size of tensor.view.offset
        if tensor.view.offset.count < shape.count {
            let diff = shape.count - tensor.view.offset.count
            for _ in 0..<diff {
                tensor.view.offset.append(0)
            }
        }
        
        self.view = ViewType(shape: shape, offset: tensor.view.offset)
        dimIndex = storage.calculateOrder(shape.count)
    }    

    public func calculateOffset() -> Int {
        var pos = offset
        for i in 0..<shape.count {
            let di = dimIndex[i]
            pos += view.offset[di]*stride[i]
        }
        
        return pos
    }

    // TODO: modified to allow less indices than dimensions to
    // be specified (e.g. v: [3x1] can be indexed as v[i]
    public func calculateOffset(indices:[Int]) -> Int {
        var pos = offset
        let size = min(indices.count, dimIndex.count)
        for i in 0..<size { // was dimIndex.count
            let di = dimIndex[i] // + start
            pos += (indices[di]+view.offset[di])*stride[i]
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
        get {
            return self[ranges]
        }
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
        return Tensor(self, dimIndex: newDimIndex, view: newView, stride: stride)
    }
    
    public func reshape(newShape:Extent) -> Tensor {
        precondition(newShape.elements == internalShape.elements, "Cannot change number of elements in Tensor.")
        
        return Tensor(storage: storage, shape: newShape)
    }
    
    // Defaults to given indices in native layout (to allow for better performance). However,
    // if consistency in traversal between storage types is required, the order can be specified
    public func indices(order:DimensionOrder?=nil) -> GeneratorSequence<IndexGenerator> {
        if let o = order {
            return GeneratorSequence<IndexGenerator>(IndexGenerator(shape, order: o))
        } else {
            return GeneratorSequence<IndexGenerator>(IndexGenerator(shape, order: storage.order))
        }
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

public func ones<S:Storage>(shape:Extent) -> Tensor<S> {
    return Tensor<S>(shape, value: S.ElementType(1))
}

public func zeros<S:Storage>(shape:Extent) -> Tensor<S> {
    return Tensor<S>(shape, value: S.ElementType(0))
}

//public func diagIndices(shape:Extent) -> 

// TODO: change to support N dimensions (requires diagIndices)
public func eye<S:Storage>(size:Int) -> Tensor<S> {
    let tensor:Tensor<S> = zeros(Extent(size, size))
    for i in 0..<size {
        tensor[i, i] = 1
    }
    
    return tensor
}

// TODO: rewrite so we don't have to reverse at the end
public func calculateBroadcastStride<S>(tensor:Tensor<S>, shape:Extent) -> [Int] {
    var stride = [Int](count: shape.count, repeatedValue: 0)
    
    // if the dimensions grow, we want to offset where values are are placed
    let start = shape.count - tensor.shape.count
    let tensorStride = tensor.storage.calculateOrder(tensor.stride)
    
    for i in 0..<tensor.shape.count {
        if shape[i+start] == tensor.shape[i] {
            stride[i+start] = tensorStride[i]
        } else if tensor.shape[i] != 1 {
            assertionFailure("Cannot broadcast on dimension \(i)")
        }
    }
    
    return tensor.storage.calculateOrder(stride)
}

public func broadcast<S>(tensor:Tensor<S>, shape:Extent) -> Tensor<S> {
    let newStride = calculateBroadcastStride(tensor, shape: shape)
    return Tensor<S>(tensor: tensor, shape: shape, stride: newStride)
}

public func broadcast<S>(left:Tensor<S>, _ right:Tensor<S>) -> (Tensor<S>, Tensor<S>) {
    if left.shape.count < right.shape.count {
        return (broadcast(left, shape: right.shape), right)
    } else {
        return (left, broadcast(right, shape: left.shape))
    }
}

func copy<StorageType:Storage>(from from:[[StorageType.ElementType]], to:Tensor<StorageType>)  {
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
    let maxDims = max(tensor1.shape.count, tensor2.shape.count)
    
    // verify other dimensions match
    precondition(axis < maxDims, "Axis is greater than number of dimensions")
    
    for i in 0..<maxDims {
        if (i != axis) {
            if tensor1.shape[i] != tensor2.shape[i] {
                precondition(tensor1.shape[i] == tensor2.shape[i],
                             "Dimensions of tensors do not match")
            }
        }
    }
    
    var shape = tensor1.shape
    shape[axis] += tensor2.shape[axis]
    
    let result = Tensor<StorageType>(shape)
    var rpos = result.indices()
    
    for pos in tensor1.indices() {
        result[rpos.next()!] = tensor1[pos]
    }
    
    for pos in tensor2.indices() {
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
    let result = Tensor<StorageType>(tensor.shape)
    for i in tensor.indices() {
        result[i] = fn(tensor[i])
    }
    
    return result
}
