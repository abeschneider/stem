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

enum TensorError: Error {
    case illegalOperation
    case sizeMismatch(lhs:Extent, rhs:Extent)
    case illegalAxis(axis:Int)
}

public enum TensorType {
    case tensor
    case vector
    case rowVector
    case columnVector
    case matrix
    case cube
}

// TensorIndex provides a mechanism to allow either an integer or a range
// be used in a Tensor's subscript
public protocol TensorIndex {
    var TensorRange: CountableRange<Int> { get }
}

extension Int : TensorIndex {
    public var TensorRange: CountableRange<Int> {
        get { return self..<(self+1) }
    }
}

extension Range : TensorIndex {
    public var TensorRange: CountableRange<Int> {
//        get { return self as! CountableRange<Int> }
        get {
            return (self.lowerBound as! Int)..<(self.upperBound as! Int)
        }
    }
}

public let all:Range = Int(0)..<Int(0)

public struct IndexGenerator: IteratorProtocol {
    var indices:[Int]
    var shape:Extent
    var dimIndex:[Int]

    public init(_ shape:Extent, dimIndex:[Int]?=nil) {
        self.shape = shape
        indices = [Int](repeating: 0, count: shape.count)
        
        if let dims = dimIndex {
            self.dimIndex = dims
        } else {
            self.dimIndex = (0..<shape.count).map { $0 }
        }
    }
    
    public init(_ shape:Extent, order:DimensionOrder) {
        self.shape = shape
        indices = [Int](repeating: 0, count: shape.count)

        switch order {
        case .columnMajor:
            dimIndex = (0..<shape.count).map { shape.count-$0-1 }
            break
        case .rowMajor:
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

// TODO: it seems unlikely all the initializers are needed, look into
// thinning down the herd
open class Tensor<StorageType:Storage> {
    public typealias ViewType = StorageView<StorageType>
    
    open var storage:StorageType
    
    // convienence variable
    open var dims:Int {
        get { return view.shape.dims.count }
    }
    
    // view into storage
    open var view:ViewType
    
    // convenience variable to access the shape of the view
    open var shape:Extent {
        get { return view.shape }
        set {
            precondition(view.shape.elements == newValue.elements, "Number of elements must match")
            view.shape = newValue
        }
    }
    
    // order to traverse the dimensions
    open var dimIndex:[Int]
    
    // step size to increment within storage for each dimension
    open var stride:[Int]
    
    // TODO: look into determining this at initialization
    open var type:TensorType {
        if shape.span == 1 {
            if shape[0] > 1 {
                return .rowVector
            } else {
                return .columnVector
            }
        } else if shape.span == 2 {
            return .matrix
        } else if shape.span == 3 {
            return .cube
        }
        
        return .tensor
    }
    
    // convenience accessor to generate a transposed view
//    open var T:Tensor<StorageType> {
//        get { return transpose() }
//    }
    
    /**
     Creates a vector along a specified axis.
     
     - Parameter array: contents of array
     - Parameter axis: axis vector lies on
 
     */
    public convenience init(_ array:[StorageType.ElementType], axis:Int=0) {
        var shapeValues = [Int](repeating: 1, count: axis+1)
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
    
    /**
     
     Creates a column vector
     
     - Parameter colvector: contents of array
     */
    public convenience init(colvector array:[StorageType.ElementType]) {
        let rows = array.count
        let shape = Extent(rows, 1)

        self.init(array: array, shape: shape)
    }

    public convenience init(_ array:[[StorageType.ElementType]]) {
        let rows = array.count
        let cols = array[0].count
        
        self.init(Extent(rows, cols))
        
        var index = indices(.columnMajor)
        for i in 0..<rows {
            for j in 0..<cols {
                self[index.next()!] = array[i][j]
            }
        }
    }
    
    public convenience init() {
        self.init(Extent(0))
    }
    
    public init(_ shape:Extent, value:StorageType.ElementType=0) {
        storage = StorageType(size: shape.elements, value: value)
        self.stride = calculateStride(Extent(storage.calculateOrder(shape.dims)))
        dimIndex = storage.calculateOrder(shape.count)
        
        view = ViewType(shape: shape, offset: Array<Int>(repeating: 0, count: shape.count))
    }
        
    init(array:[StorageType.ElementType], shape:Extent) {
        storage = StorageType(array: array)
        self.stride = calculateStride(Extent(storage.calculateOrder(shape.dims)))
        dimIndex = storage.calculateOrder(shape.count)
        view = ViewType(shape: shape, offset: Array<Int>(repeating: 0, count: shape.count))
    }
    
    init(storage:StorageType, shape:Extent, view:StorageView<StorageType>?=nil) { //, offset:Int?=nil) {
        self.storage = storage
        self.stride = calculateStride(Extent(storage.calculateOrder(shape.dims)))
        dimIndex = storage.calculateOrder(shape.count)
        self.view = view ?? ViewType(shape: shape, offset: Array<Int>(repeating: 0, count: shape.count))
    }
    
    init(_ tensor:Tensor, window:[CountableRange<Int>]) {
        storage = tensor.storage
        stride = tensor.stride

        let viewShape = Extent(window.enumerated().map {
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
            self.view = view ?? ViewType(shape: tensor.shape, offset: tensor.view.offset)
            storage = StorageType(size: tensor.shape.elements, value: 0)
            for i in 0..<tensor.shape.elements {
                storage[i] = tensor.storage[i]
            }
        } else {
            // NB: If making a copy, previously defined offset is no longer valid
            self.view = view ?? ViewType(shape: tensor.shape, offset: Array<Int>(repeating: 0, count: tensor.dims))
            storage = tensor.storage
        }
        
        self.view = view ?? ViewType(shape: tensor.shape, offset: copy ? tensor.view.offset : Array<Int>(repeating: 0, count: tensor.dims))
        self.dimIndex = dimIndex ?? tensor.dimIndex
        self.stride = stride ?? tensor.stride
    }
    
    init(tensor:Tensor, shape:Extent, stride:[Int]) {
        storage = tensor.storage
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

    open func calculateOffset() -> Int {
        var pos = 0 //offset
        for i in 0..<shape.count {
            let di = dimIndex[i]
            pos += view.offset[di]*stride[i]
        }
        
        return pos
    }

    open func calculateOffset(_ indices:[Int]) -> Int {
        var pos = 0
        let size = min(indices.count, dimIndex.count)
        for i in 0..<size {
            let di = dimIndex[i] // + start
            pos += (indices[di]+view.offset[di])*stride[i]
        }
        
        return pos
    }
    
    open subscript(indices:[Int]) -> StorageType.ElementType {
        get { return storage[calculateOffset(indices)] }
        set { storage[calculateOffset(indices)] = newValue }
    }
    
    open subscript(indices:Int...) -> StorageType.ElementType {
        get { return storage[calculateOffset(indices)] }
        set { storage[calculateOffset(indices)] = newValue }
    }
    
    open subscript(ranges:[TensorIndex]) -> Tensor {
        get {
            return Tensor(self, window: ranges.map { $0.TensorRange })
        }
        set {
            let view = Tensor(self, window: ranges.map { $0.TensorRange })
            let bvalue = broadcast(newValue, shape: view.shape)
            copy(from: bvalue, to: view)
        }
    }
    
    open subscript(ranges:TensorIndex...) -> Tensor {
        get {
            return self[ranges]
        }
        set {
            let view = Tensor(self, window: ranges.map { $0.TensorRange })
            let bvalue = broadcast(newValue, shape: view.shape)
            copy(from: bvalue, to: view)
        }
    }
    
    open func transpose() -> Tensor<StorageType> {
        let newDimIndex = Array(dimIndex.reversed())
        let newShape = Extent(view.shape.reversed())
        let newOffset = Array(view.offset.reversed())
        let newView = StorageView<StorageType>(shape: newShape, offset: newOffset)
        return Tensor(self, dimIndex: newDimIndex, view: newView, stride: stride)
    }
    
    open func reshape(_ newShape:Extent) -> Tensor {
        precondition(newShape.elements == shape.elements, "Cannot change number of elements in Tensor.")
        
        return Tensor(storage: storage, shape: newShape)
    }
    
    open func resize(_ newShape:Extent) {
        if shape != newShape {
            storage = StorageType(size: newShape.elements, value: 0)
            self.stride = calculateStride(Extent(storage.calculateOrder(newShape.dims)))
            dimIndex = storage.calculateOrder(newShape.count)
            
            view = ViewType(shape: newShape, offset: Array<Int>(repeating: 0, count: newShape.count))
        }
    }
    
    // Defaults to given indices in native layout (to allow for better performance). However,
    // if consistency in traversal between storage types is required, the order can be specified
    open func indices(_ order:DimensionOrder?=nil) -> IteratorSequence<IndexGenerator> {
        let o = order ?? storage.order
        return IteratorSequence<IndexGenerator>(IndexGenerator(shape, order: o))
    }
}

extension Tensor {
    // general case
    fileprivate func elementToString(_ v:StorageType.ElementType) -> String {
        return String(describing: v)
    }
    
    fileprivate func elementToString(_ v:Double) -> String {
        return String(format: "%2.3f", v)
    }
    
    fileprivate func elementToString(_ v:Float) -> String {
        return String(format: "%2.3f", v)
    }

    fileprivate func elementToString(_ v:Int) -> String {
        return String(format: "%d", v)
    }
    
    fileprivate func convertToString(_ indices:[Int], dim:Int) -> String {
        var idx = indices
        
        if dim == shape.count-1 {
            // last dimension, convert values to string
            let values:[String] = (0..<shape[dim]).map({(i:Int) -> String in
                idx[dim] = i
                return elementToString(self[idx])
            })
            return "[\(values.joined(separator: ",\t"))]"
        } else {
            let values:[String] = (0..<shape[dim]).map({(i:Int) -> String in
                idx[dim] = i
                
                let indent:String = i > 0 ? String(repeating: " ", count: dim+1) : ""
                return "\(indent)\(convertToString(idx, dim: dim+1))"
            })
            return "[\(values.joined(separator: "\n"))]"
        }
    }
}

extension Tensor: CustomStringConvertible {
    public var description: String {
        get {
            let indices = (0..<shape.count).map { _ in 0 }
            return convertToString(indices, dim: 0)
        }
    }
}

public func ones<S:Storage>(_ shape:Extent) -> Tensor<S> {
    return Tensor<S>(shape, value: S.ElementType(1))
}

public func zeros<S:Storage>(_ shape:Extent) -> Tensor<S> {
    return Tensor<S>(shape, value: S.ElementType(0))
}

//public func diagIndices(shape:Extent) -> 

// TODO: change to support N dimensions (requires diagIndices)
public func eye<S:Storage>(_ size:Int) -> Tensor<S> {
    let tensor:Tensor<S> = zeros(Extent(size, size))
    for i in 0..<size {
        tensor[i, i] = 1
    }
    
    return tensor
}

// TODO: rewrite so we don't have to reverse at the end
public func calculateBroadcastStride<S>(_ tensor:Tensor<S>, shape:Extent) -> [Int] {
    var stride = [Int](repeating: 0, count: shape.count)
    
    // if the dimensions grow, we want to offset where values are are placed
    let start = shape.count - tensor.shape.count
    let tensorStride = tensor.storage.calculateOrder(tensor.stride)
    
    for i in 0..<tensor.shape.count {
        if shape[i+start] == tensor.shape[i] {
            stride[i+start] = tensorStride[i]
        } else if tensor.shape[i] != 1 {
            assertionFailure("Cannot broadcast from \(tensor.shape.dims) to \(shape.dims)")
        }
    }
    
    return tensor.storage.calculateOrder(stride)
}

public func broadcast<S>(_ tensor:Tensor<S>, shape:Extent) -> Tensor<S> {
    let newStride = calculateBroadcastStride(tensor, shape: shape)
    return Tensor<S>(tensor: tensor, shape: shape, stride: newStride)
}

public func broadcast<S>(_ left:Tensor<S>, _ right:Tensor<S>) -> (Tensor<S>, Tensor<S>) {
    if left.shape < right.shape {
        return (broadcast(left, shape: right.shape), right)
    } else {
        return (left, broadcast(right, shape: left.shape))
    }
}

public func copy<StorageType:Storage>(from:[[StorageType.ElementType]], to:Tensor<StorageType>)  {
    precondition(to.shape[0] != from.count || to.shape[1] != from[0].count,
                 "Destination and source must be the same size")

    var toIndices = to.indices()
    for i in 0..<from.count {
        for j in 0..<from[i].count {
            to[toIndices.next()!] = from[i][j]
        }
    }
}

public func copy<StorageType:Storage>(from:Tensor<StorageType>, to:Tensor<StorageType>) {
    precondition(to.shape == from.shape, "Destination and source must be the same size")
    
    let zippedIndices = zip(from.indices(), to.indices())
    for (i, j) in zippedIndices {
        to[j] = from[i]
    }
}

public func copy<StorageType>(_ tensor:Tensor<StorageType>) -> Tensor<StorageType> {
    return Tensor<StorageType>(tensor, copy: true)
}

public func fill<StorageType:Storage>(_ tensor:Tensor<StorageType>, value:StorageType.ElementType) {
    for i in tensor.indices() {
        tensor[i] = value
    }
}

// concats two tensors along the given axis (0: rows, 1: cols, etc.)
public func concat<S:Storage>(_ tensor1:Tensor<S>, _ tensor2:Tensor<S>, axis:Int=0, to:Tensor<S>?=nil) -> Tensor<S> {
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
    
    let result = to == nil ? Tensor<S>(shape) : to!
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

public func concat<StorageType:Storage>(_ tensor1:Tensor<StorageType>,
                                        _ tensor2:Tensor<StorageType>,
                                        _ tensor3:Tensor<StorageType>,
                                        _ rest:Tensor<StorageType>...,
                                        axis:Int=0) -> Tensor<StorageType>
{
    var result = concat(tensor1, tensor2, axis: axis)
    result = concat(result, tensor3, axis:axis)
    for i in 0..<rest.count {
        result = concat(result, rest[i], axis: axis)
    }
    
    return result
}

public func concat<StorageType:Storage>(_ tensors:[Tensor<StorageType>], axis:Int=0) -> Tensor<StorageType>
{
    var result = concat(tensors[0], tensors[1], axis: axis)
    for i in 2..<tensors.count {
        result = concat(result, tensors[i], axis: axis)
    }
    
    return result
}

public func vstack<StorageType:Storage>(_ tensor1:Tensor<StorageType>, _ tensor2:Tensor<StorageType>) -> Tensor<StorageType>
{
    return concat(tensor1, tensor2, axis: 0)
}

public func hstack<StorageType:Storage>(_ tensor1:Tensor<StorageType>, _ tensor2:Tensor<StorageType>) -> Tensor<StorageType>
{
    return concat(tensor1, tensor2, axis: 1)
}

public func map<StorageType:Storage>(
    _ tensor:Tensor<StorageType>,
    fn:(StorageType.ElementType) -> StorageType.ElementType) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(tensor.shape)
    for i in tensor.indices() {
        result[i] = fn(tensor[i])
    }
    
    return result
}

public func ravel<StorageType:Storage>(_ tensor:Tensor<StorageType>) -> Tensor<StorageType> {
    return tensor.reshape(Extent(tensor.shape.elements))
}
