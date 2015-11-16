//
//  tensor.swift
//  stem
//
//  Created by Abe Schneider on 11/10/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation
import Accelerate

protocol TensorIndex {
    var TensorRange: Range<Int> { get }
}

extension Int : TensorIndex {
    var TensorRange: Range<Int> {
        get {
            return Range<Int>(start: self, end: self+1)
        }
    }
}

extension Range : TensorIndex {
    var TensorRange: Range<Int> {
        get {
            return Range<Int>(start: self.startIndex as! Int, end: self.endIndex as! Int)
        }
    }
}

// TODO: can this can be defined as parameterized by Storage, and
// delegated by view? Otherwise, two Tensors with different view types
// will also be different Tensor types
class Tensor<ViewType:StorageView> {
    typealias StorageType = ViewType.StorageType

//    var storage:StorageType
    var view:ViewType
    
    // forward shape from view
    var shape:Extent { return view.shape }
    
    init(array:[StorageType.ElementType], shape:Extent) {
        let storage = StorageType(array: array, shape: shape)
        view = ViewType(storage: storage)
    }
    
    init(shape:Extent) {
        let storage = StorageType(shape: shape)
        view = ViewType(storage: storage)
    }
    
    init(view:ViewType) {
        self.view = view
    }
    
    subscript(indices:[Int]) -> ViewType.StorageType.ElementType {
        get { return view[indices] }
        set { view[indices] = newValue }
    }
    
    subscript(indices:Int...) -> ViewType.StorageType.ElementType {
        get { return view[indices] }
        set { view[indices] = newValue }
    }
    
    subscript(ranges:[TensorIndex]) -> Tensor {
        get {
            let v = ViewType(storage: view.storage, view: ranges.map {$0.TensorRange})
            return Tensor(view:v)
        }
//        set(newValue) {
//            let s = S(storage:self.storage, view:ranges.map {$0.NDArrayRange})
//            s.copy(newValue.storage)
//        }
    }
    
    subscript(ranges:TensorIndex...) -> Tensor {
        get { return self[ranges] }
//        set { self[ranges] = newValue }
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
    var description: String {
        get {
            let indices = (0..<shape.dims()).map { _ in 0 }
            return convertToString(indices, dim: 0)
        }
    }
}

class Vector<ViewType:StorageView>: Tensor<ViewType> {
    // axis on which vector lies along
    var axis:Int
    
    init(_ array:[StorageType.ElementType]) {
        axis = 0
        super.init(array: array, shape: Extent(array.count))
    }
    
    init(_ tensor:Tensor<ViewType>) {
        // TODO: should assert that we're being pass a vector
        axis = tensor.shape.max()
        super.init(view: tensor.view)
    }
    
    init(rows:Int) {
        axis = 0
        super.init(shape: Extent(rows))
    }
    
    init(cols:Int) {
        axis = 1
        super.init(shape: Extent(cols))
    }
}

class Matrix<ViewType:StorageView>: Tensor<ViewType> {
    var transposed:Bool
    
    init(_ array:[[StorageType.ElementType]]) {
        transposed = false
        
        // first allocate space
        // TODO: make sure dims are correct
        super.init(shape: Extent(array.count, array[0].count))
        
        // next copy array
        let indices = view.storageIndices()
        for i in 0..<view.storage.shape[0] {
            for j in 0..<view.storage.shape[1] {
                view.storage[indices.next()!] = array[i][j]
            }
        }
    }
    
    init(rows:Int, cols:Int) {
        transposed = false
        super.init(shape: Extent(rows, cols))
    }
    
    init(_ m:Matrix, transpose:Bool=false) {
        self.transposed = transpose
        super.init(view: m.view)
    }
    
    func transpose() -> Matrix {
        return Matrix(self, transpose: !transposed)
    }
}

func elementwiseBinaryOp<V:StorageView>
    (left:Tensor<V>, _ right:Tensor<V>, result:Tensor<V>,
    op:(left:V.StorageType.ElementType, right:V.StorageType.ElementType) -> V.StorageType.ElementType)
{
    // don't want to compare shape directly (otherwise have to deal with 1x4 vs 4
    // is this the best method?
    assert(left.shape.elements == right.shape.elements)
    assert(left.shape.elements == result.shape.elements)
    
    let indexLeft = left.view.storageIndices()
    let indexRight = right.view.storageIndices()
    
    var i:Int = 0
    for (l, r) in Zip2Sequence(GeneratorSequence(indexLeft), GeneratorSequence(indexRight)) {
        result[i++] = op(left: left[l], right: right[r])
    }
}

func subtract<V:StorageView where V.StorageType.ElementType:NumericType>
    (left left:Tensor<V>, right:Tensor<V>, result:Tensor<V>)
{
    elementwiseBinaryOp(left, right, result: result, op: { return $0 - $1 })
}

func -<V:StorageView where V.StorageType.ElementType:NumericType>
    (left:Tensor<V>, right:Tensor<V>) -> Tensor<V>
{
    let result = Tensor<V>(shape: left.shape)
    subtract(left: left, right: right, result: result)
    
    return result
}

func add<V:StorageView where V.StorageType.ElementType:NumericType>(left left:Tensor<V>, right:Tensor<V>, result:Tensor<V>) {
    elementwiseBinaryOp(left, right, result: result, op: { $0 + $1 })
}

func +<V:StorageView where V.StorageType.ElementType:NumericType>(left:Tensor<V>, right:Tensor<V>) -> Tensor<V> {
    let result = Tensor<V>(shape: left.shape)
    add(left: left, right: right, result: result)
    
    return result
}

func dot<V:StorageView where V.StorageType.ElementType:NumericType>
    (left left:Matrix<V>, right:Vector<V>, result:Vector<V>)
{
    assert(left.shape[1] == right.shape[0])
    assert(right.shape[0] == result.shape[0])
    
    // per row
    for i in 0..<left.shape[0] {
        // per column
        for j in 0..<right.shape[0] {
            result[j] = result[j] + left[i, j]*right[j]
        }
    }
}

func abs<V:StorageView where V.StorageType.ElementType:AbsoluteValuable>(tensor:Tensor<V>) -> Tensor<V> {
    let result = Tensor<V>(shape: tensor.shape)
    for index in tensor.view.storageIndices() {
        result.view.storage[index] = abs(tensor.view.storage[index])
    }
    
    return result
}

func isClose<V:StorageView where V.StorageType.ElementType:NumericType>
    (left:Tensor<V>, _ right:Tensor<V>, eps: V.StorageType.ElementType) -> Bool
{
    let diff = left - right
    let adiff:Tensor<V> = abs(diff)
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
