//
//  tensor.swift
//  stem
//
//  Created by Abe Schneider on 11/10/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation
import Accelerate

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

// TODO: can this can be defined as parameterized by Storage, and
// delegated by view? Otherwise, two Tensors with different view types
// will also be different Tensor types
public class Tensor<StorageType:Storage> {
    public typealias ViewType = StorageView<StorageType>
    
    public var view:ViewType
    
    // forward shape from view
    public var shape:Extent { return view.shape }
    public var transposed:Bool
    
    public init(array:[StorageType.ElementType], shape:Extent) {
        let storage = StorageType(array: array, shape: shape)
        view = ViewType(storage: storage)
        transposed = false
    }
    
    public init(shape:Extent) {
        let storage = StorageType(shape: shape)
        view = ViewType(storage: storage)
        transposed = false
    }
    
    public init(view:ViewType) {
        self.view = view
        transposed = false
    }
    
    public subscript(indices:[Int]) -> StorageType.ElementType {
        get { return view[indices] }
        set { view[indices] = newValue }
    }
    
    public subscript(indices:Int...) -> StorageType.ElementType {
        get { return view[indices] }
        set { view[indices] = newValue }
    }
    
    public subscript(ranges:[TensorIndex]) -> Tensor {
        get {
            let v = ViewType(storage: view.storage, window: ranges.map {$0.TensorRange})
            return Tensor(view:v)
        }
//        set(newValue) {
//            let s = S(storage:self.storage, view:ranges.map {$0.NDArrayRange})
//            s.copy(newValue.storage)
//        }
    }
    
    public subscript(ranges:TensorIndex...) -> Tensor {
        get { return self[ranges] }
//        set { self[ranges] = newValue }
    }
    
    public func transpose() -> Tensor<StorageType> {
        let window = Array(view.window.reverse())
        let dimIndex = Array(view.dimIndex.reverse())
        let newView = StorageView(storage: view.storage, window: window, dimIndex: dimIndex)
        
        return Tensor(view: newView)
    }
}

extension Tensor {
    private func convertToString(var indices:[Int], dim:Int) -> String {
        if dim == shape.dims() - 1 {
//        if view.shape[dim] > 1 {
            // last dimension, convert values to string
            let values:[String] = (0..<shape[dim]).map({(i:Int) -> String in
                indices[dim] = i
                return String(format: "%2.3f", view[indices] as! Double)
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
            let indices = (0..<shape.dims()).map { _ in 0 }
            return convertToString(indices, dim: 0)
        }
    }
}

/*
The following classes add constraints to the Tensor class. This can
be useful for dispatching to functions based on those constraints
(e.g. if a function needs a RowVector, any type of Vector, or a 
Matrix).
*/

// can be either row or column vector
public class Vector<StorageType:Storage>: Tensor<StorageType> {
    public init(_ array:[StorageType.ElementType], transposed:Bool=false) {
        super.init(array: array, shape: Extent(array.count))
        self.transposed = transposed
    }
    
    public init(_ tensor:Tensor<StorageType>) {
        // TODO: should assert that we're being passed a vector
        super.init(view: tensor.view)
    }
    
    public init(rows:Int) {
        super.init(shape: Extent(rows))
    }
    
    public init(cols:Int) {
        super.init(shape: Extent(cols))
        transposed = true
    }
    
    public override init(view:ViewType) {
        super.init(view: view)
    }
    
    public override func transpose() -> Vector {
        let window = Array(view.window.reverse())
        let dimIndex = Array(view.dimIndex.reverse())
        let newView = StorageView(storage: view.storage, window: window, dimIndex: dimIndex)
        return Vector(view: newView)
    }
}

// constrained to be just a column vector
public class ColumnVector<StorageType:Storage>: Vector<StorageType> {
    public init(_ array:[StorageType.ElementType]) {
        super.init(array, transposed: false)
    }
    
    public override init(_ tensor:Tensor<StorageType>) {
        // TODO: should assert that we're being passed a vector
        super.init(view: tensor.view)
    }
    
    public override init(rows:Int) {
        super.init(rows: rows)
    }
    
    public override init(view:ViewType) {
        super.init(view: view)
    }
    
    public override func transpose() -> RowVector<StorageType> {
        let window = Array(view.window.reverse())
        let dimIndex = Array(view.dimIndex.reverse())
        let newView = StorageView(storage: view.storage, window: window, dimIndex: dimIndex)
        return RowVector(view: newView)
    }
}

// constrained to be just a row vector
public class RowVector<StorageType:Storage>: Vector<StorageType> {
    public init(_ array:[StorageType.ElementType]) {
        super.init(array, transposed: true)
    }
    
    public override init(_ tensor:Tensor<StorageType>) {
        // TODO: should assert that we're being passed a vector
        super.init(view: tensor.view)
        transposed = true
    }
    
    public override init(rows:Int) {
        super.init(rows: rows)
    }
    
    public override init(view:ViewType) {
        super.init(view: view)
    }
    
    public override func transpose() -> ColumnVector<StorageType> {
        let window = Array(view.window.reverse())
        let dimIndex = Array(view.dimIndex.reverse())
        let newView = StorageView(storage: view.storage, window: window, dimIndex: dimIndex)
        return ColumnVector(view: newView)
    }
}

public class Matrix<StorageType:Storage>: Tensor<StorageType> {
    public init(_ array:[[StorageType.ElementType]], copyTransposed:Bool=false) {
        var rows = array.count
        var cols = array[0].count
        
//        if (copyTransposed) {
//            (rows, cols) = (cols, rows)
//        }
        
        if (!copyTransposed) {
            super.init(shape: Extent(rows, cols))
        } else {
            super.init(shape: Extent(cols, rows))
        }
        
        // copy array
        var indices = view.storageIndices()
        
        if (!copyTransposed) {
            for i in 0..<rows {
                for j in 0..<cols {
                    let index = indices.next()!
                    view.storage[index] = array[i][j]
                }
            }
        } else {
            for j in 0..<cols {
                for i in 0..<rows {                
                    let index = indices.next()!
                    view.storage[index] = array[i][j]
                }
            }
        }
    }
    
    public init(rows:Int, cols:Int) {
        super.init(shape: Extent(rows, cols))
    }
    
    public override init(view:ViewType) {
        super.init(view: view)
    }
    
    public override func transpose() -> Matrix {
        let window = Array(view.window.reverse())
        let dimIndex = Array(view.dimIndex.reverse())
        let newView = StorageView(storage: view.storage, window: window, dimIndex: dimIndex)
        return Matrix(view: newView)
    }
}

func elementwiseBinaryOp<StorageType:Storage>
    (left:Tensor<StorageType>, _ right:Tensor<StorageType>, result:Tensor<StorageType>,
    op:(left:StorageType.ElementType, right:StorageType.ElementType) -> StorageType.ElementType)
{
    // don't want to compare shape directly (otherwise have to deal with 1x4 vs 4
    // is this the best method?
    assert(left.shape.elements == right.shape.elements)
    assert(left.shape.elements == result.shape.elements)
    
    let indexLeft = left.view.storageIndices()
    let indexRight = right.view.storageIndices()
    var indexResult = result.view.storageIndices()

    // TODO: There should be better syntax to support this use-case
    for (l, r) in Zip2Sequence(GeneratorSequence(indexLeft), GeneratorSequence(indexRight)) {
        let idx = indexResult.next()!
        result.view.storage[idx] = op(left: left.view.storage[l], right: right.view.storage[r])
    }
}

public func subtract<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Tensor<StorageType>, right:Tensor<StorageType>, result:Tensor<StorageType>)
{
    elementwiseBinaryOp(left, right, result: result, op: { return $0 - $1 })
}

public func -<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Tensor<StorageType>, right:Tensor<StorageType>) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(shape: left.shape)
    subtract(left: left, right: right, result: result)
    
    return result
}

public func add<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Tensor<StorageType>, right:Tensor<StorageType>, result:Tensor<StorageType>)
{
    assert(left.shape == right.shape)
    elementwiseBinaryOp(left, right, result: result, op: { $0 + $1 })
}

public func add<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:RowVector<StorageType>, result:Matrix<StorageType>)
{
    // NxM + N
    assert(left.shape[0] == right.shape[0])
    
    let rows = left.shape[0]
    let cols = left.shape[1]
    for i in 0..<cols {
        print("left: \(left[0..<rows, i])")
        print("right: \(right)")
        elementwiseBinaryOp(left[0..<rows, i], right, result: result[0..<rows, i], op: { $0 + $1 })
    }
}

public func +<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Tensor<StorageType>, right:Tensor<StorageType>) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(shape: left.shape)
    add(left: left, right: right, result: result)
    
    return result
}

struct IllegalOperation: ErrorType {}

// completely generic type currently unsupported
public func dot<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Tensor<StorageType>, right:Tensor<StorageType>, result:Tensor<StorageType>) throws
{
    throw IllegalOperation()
}

public func dot<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Vector<StorageType>, right:Vector<StorageType>) -> StorageType.ElementType
{
    assert(left.shape[0] == right.shape[0])
    
    var result:StorageType.ElementType = 0
    
    // per row
    for i in 0..<left.shape[0] {
        // per column
        for j in 0..<right.shape.elements {
            result = result + left[i]*right[j]
        }
    }
    
    return result
}

public func dot<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:Vector<StorageType>, result:Tensor<StorageType>)
{
    // NxM x M -> N
    assert(left.shape[0] == result.shape[0])
    assert(left.shape[1] == right.shape[0])
//    assert(right.shape.elements == result.shape.elements)

    // per row
    for i in 0..<left.shape[0] {
        // per column
        for j in 0..<left.shape[1] {
            result[i] = result[i] + left[i, j]*right[j]
        }
    }
}

//public func dot<StorageType:Storage where StorageType.ElementType:NumericType>
//    (left left:Vector<StorageType>, right:Matrix<StorageType>, result:Tensor<StorageType>)
//{
//    // N x NxM -> M
//    assert(left.shape[0] == right.shape[0])
//    assert(left.shape[1] == result.shape[0])
//    
//    for i in 0..<left.shape[0] {
//        for j in 0..<left.shape[1] {
//            result[j] = result[j] + left[i]*right[j, i]
//        }
//    }
//}

public func dot<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:Matrix<StorageType>, result:Tensor<StorageType>)
{
    print("left.shape = \(left.shape), right.shape = \(right.shape)")
    assert(left.shape[1] == right.shape[0])
//    assert(right.shape.elements == result.shape.elements)

    // NxM x MxK -> NxK
    for n in 0..<left.shape[0] {
        for m in 0..<left.shape[1] {
//            for k in 0..<right.shape[0] {
            for k in 0..<right.shape[1] {
                result[n, k] = result[n, k] + left[n, m]*right[m, k]
            }
//            }
        }
    }
}

public func outer<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Vector<StorageType>, right:Vector<StorageType>, result:Tensor<StorageType>)
{
    assert(left.shape[0] == result.shape[0])
    assert(right.shape[0] == result.shape[1])
    
    for i in 0..<result.shape[0] {
        for j in 0..<result.shape[1] {
            result[i, j] = left[i]*right[j]
        }
    }
}

public func *<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Matrix<StorageType>, right:RowVector<StorageType>) -> Vector<StorageType>
{
    let result = Vector<StorageType>(rows: right.shape[0])
    dot(left: left, right: right, result: result)
    return result
}

public func *<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:ColumnVector<StorageType>, right:RowVector<StorageType>) -> Matrix<StorageType>
{
    let result = Matrix<StorageType>(rows: left.shape[0], cols: right.shape[0])
    outer(left: left, right: right, result: result)
    return result
}

public func *<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:RowVector<StorageType>, right:ColumnVector<StorageType>) -> StorageType.ElementType
{
    return dot(left: left, right: right)
}

//public func *<StorageType:Storage where StorageType.ElementType:NumericType>
//    (left:Matrix<StorageType>, right:Tensor<StorageType>) throws -> Vector<StorageType>
//{
//    throw IllegalOperation()
//}

public func abs<StorageType:Storage where StorageType.ElementType:NumericType>
    (tensor:Tensor<StorageType>) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(shape: tensor.shape)
    for index in tensor.view.storageIndices() {
        result.view.storage[index] = abs(tensor.view.storage[index])
    }
    
    return result
}

public func isClose<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Tensor<StorageType>, _ right:Tensor<StorageType>, eps: StorageType.ElementType) -> Bool
{
    let diff = left - right
    let adiff:Tensor<StorageType> = abs(diff)
    for i in adiff.view.storageIndices() {
        if adiff.view.storage[i] >= eps { return false }
    }
    return true
}

//func ==<V:StorageView where V.StorageType.ElementType:NumericType>
//    (left:Tensor<V>, right:Tensor<V>) -> Bool
//{
//    return isClose(left, right, 10e-4) //V.StorageType.ElementType(10e-4))
//}
