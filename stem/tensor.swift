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
    
    public init(array:[StorageType.ElementType], shape:Extent) {
        let storage = StorageType(array: array, shape: shape)
        view = ViewType(storage: storage)
    }
    
    public init(shape:Extent) {
        let storage = StorageType(shape: shape)
        view = ViewType(storage: storage)
    }
    
    public init(view:ViewType) {
        self.view = view
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
//                print("\(indices), \(view[indices] as! Double)")
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

public class Vector<StorageType:Storage>: Tensor<StorageType> {
    public init(_ array:[StorageType.ElementType]) {
        super.init(array: array, shape: Extent(array.count))
    }
    
    public init(_ tensor:Tensor<StorageType>) {
        // TODO: should assert that we're being pass a vector
        super.init(view: tensor.view)
    }
    
    public init(rows:Int) {
        super.init(shape: Extent(rows))
    }
    
    public init(cols:Int) {
        super.init(shape: Extent(cols))
    }
}

public class Matrix<StorageType:Storage>: Tensor<StorageType> {
    var transposed:Bool
    
    public init(_ array:[[StorageType.ElementType]]) {
        transposed = false
        
        // first allocate space
        // TODO: make sure dims are correct
        super.init(shape: Extent(array.count, array[0].count))
        
        // next copy array
        let indices = view.storageIndices()
        for i in 0..<view.storage.shape[0] {
            for j in 0..<view.storage.shape[1] {
                let index = indices.next()!
//                print("** (\(i), \(j)=>\(index)): \(array[i][j])")
                view.storage[index] = array[i][j]
            }
        }
    }
    
    public init(rows:Int, cols:Int) {
        transposed = false
        super.init(shape: Extent(rows, cols))
    }
    
    public override init(view:ViewType) {
        transposed = false
        super.init(view: view)
    }
    
//    public init(_ m:Matrix, transpose:Bool=false) {
//        self.transposed = transpose
//        // need to make a new view .. this requires something like:
//        // m.view.make(...)
//        // alternatively, have view.transpose function?
////        let type = Mirror(reflecting:m.view)
////        var view = type.subjectType(storage: m.storage, view: m.window)
//        let viewType = m.view.dynamicType
////        let view = viewType.init(storage: m.view.storage, window: [m.view.window[1], m.view.window[0]])
////        let view:StorageView = m.view.transpose()
////        let view = transposeView(view: m.view)
////        let transposeType = m.view.dynamicType.TransposeType(storage: m.view.storage)
////        let transposeType = viewType.TransposeType.self
////        transposeType(storage: m.view.storage)
////        let view = transposeType(storage: m.view.storage)
////        transposeType.init(storage: m.view.storage, window: m.view.window)
////        let view = transposeType(storage: m.view.storage, window: [m.view.window[1], m.view.window[0]])
//        super.init(view: view.transpose())
//    }
    
//    public func transpose() -> Matrix {
////        let transposeType = view.dynamicType.TransposeType.self.dynamicType
//        return Matrix(view: view.transpose())
//    }    
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
    
    var i:Int = 0
    for (l, r) in Zip2Sequence(GeneratorSequence(indexLeft), GeneratorSequence(indexRight)) {
        result[i++] = op(left: left[l], right: right[r])
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
    elementwiseBinaryOp(left, right, result: result, op: { $0 + $1 })
}

public func +<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Tensor<StorageType>, right:Tensor<StorageType>) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(shape: left.shape)
    add(left: left, right: right, result: result)
    
    return result
}

public func dot<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:Vector<StorageType>, result:Vector<StorageType>)
{
    print(left.shape)
    print(right.shape)
//    assert(left.shape[1] == right.shape[0])
//    assert(right.shape.elements == result.shape.elements)
//
//    // per row
//    for i in 0..<left.shape[0] {
//        // per column
//        for j in 0..<right.shape.elements {
//            result[j] = result[j] + left[i, j]*right[j]
//        }
//    }
}

public func *<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Matrix<StorageType>, right:Vector<StorageType>) -> Vector<StorageType>
{
    let result = Vector<StorageType>(rows: right.shape[0])
    dot(left: left, right: right, result: result)
    return result
}

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
