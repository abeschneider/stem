//
//  tensorops.swift
//  stem
//
//  Created by Abe Schneider on 12/9/15.
//  Copyright © 2015 none. All rights reserved.
//

import Foundation

func elementwiseBinaryOp<StorageType:Storage>
    (left:Tensor<StorageType>, _ right:Tensor<StorageType>, result:Tensor<StorageType>,
    op:(left:StorageType.ElementType, right:StorageType.ElementType) -> StorageType.ElementType)
{
    assert(left.shape.elements == right.shape.elements)
    assert(left.shape.elements == result.shape.elements)
    
    let indexLeft = left.storageIndices()
    let indexRight = right.storageIndices()
    var indexResult = result.storageIndices()
    
    // TODO: There should be better syntax to support this use-case
    for (l, r) in Zip2Sequence(GeneratorSequence(indexLeft), GeneratorSequence(indexRight)) {
        let idx = indexResult.next()!
        result.storage[idx] = op(left: left.storage[l], right: right.storage[r])
    }
}

func elementwiseBinaryOp<StorageType:Storage>
    (left:Tensor<StorageType>, _ right:StorageType.ElementType, result:Tensor<StorageType>,
    op:(left:StorageType.ElementType, right:StorageType.ElementType) -> StorageType.ElementType)
{
    assert(left.shape.elements == result.shape.elements)
    
    var indexResult = result.storageIndices()
    for i in left.storageIndices() {
        let idx = indexResult.next()!
        result.storage[idx] = op(left: left.storage[i], right: right)
    }
}

func elementwiseBinaryOp<StorageType:Storage>
    (left:StorageType.ElementType, _ right:Tensor<StorageType>, result:Tensor<StorageType>,
    op:(left:StorageType.ElementType, right:StorageType.ElementType) -> StorageType.ElementType)
{
    assert(right.shape.elements == result.shape.elements)
    
    var indexResult = result.storageIndices()
    for i in right.storageIndices() {
        let idx = indexResult.next()!
        result.storage[idx] = op(left: left, right: right.storage[i])
    }
}

//
// addition
//

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
        elementwiseBinaryOp(left[0..<rows, i], right, result: result[0..<rows, i], op: { $0 + $1 })
    }
}

public func add<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:ColumnVector<StorageType>, result:Matrix<StorageType>)
{
    // NxM + N
    assert(left.shape[0] == right.shape[0])
    
    let rows = left.shape[0]
    let cols = left.shape[1]
    for i in 0..<rows {
        elementwiseBinaryOp(left[i, 0..<cols], right, result: result[i, 0..<cols], op: { $0 + $1 })
    }
}

public func iadd<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Tensor<StorageType>, right:Tensor<StorageType>)
{
    // general case is currently not supported
    assert(false)
}

public func iadd<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Vector<StorageType>, right:Vector<StorageType>)
{
    // NxM + N
    assert(left.shape.elements == right.shape.elements)
    elementwiseBinaryOp(left, right, result: left, op: { $0 + $1 })
}

public func iadd<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:ColumnVector<StorageType>)
{
    // NxM + N
    assert(left.shape[1] == right.shape[0])
    
    let rows = left.shape[0]
    let cols = left.shape[1]
    for i in 0..<rows {
        elementwiseBinaryOp(left[i, 0..<cols], right, result: left[i, 0..<cols], op: { $0 + $1 })
    }
}

public func iadd<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:RowVector<StorageType>)
{
    // NxM + N
    assert(left.shape[0] == right.shape[0])
    
    let rows = left.shape[0]
    let cols = left.shape[1]
    for i in 0..<cols {
        elementwiseBinaryOp(left[0..<rows, i], right, result: left[0..<rows, i], op: { $0 + $1 })
    }
}

public func +<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Tensor<StorageType>, right:Tensor<StorageType>) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(shape: left.shape)
    add(left: left, right: right, result: result)
    
    return result
}

public func +=<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Tensor<StorageType>, right:Tensor<StorageType>)
{
    try! iadd(left: left, right: right)
}

public func +=<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Vector<StorageType>, right:Vector<StorageType>)
{
    iadd(left: left, right: right)
}

public func +=<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Matrix<StorageType>, right:ColumnVector<StorageType>)
{
    iadd(left: left, right: right)
}

public func +=<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Matrix<StorageType>, right:RowVector<StorageType>)
{
    iadd(left: left, right: right)
}


//
// subtraction
//

public func sub<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Tensor<StorageType>, right:Tensor<StorageType>, result:Tensor<StorageType>)
{
    elementwiseBinaryOp(left, right, result: result, op: { return $0 - $1 })
}

public func sub<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:StorageType.ElementType, right:Tensor<StorageType>, result:Tensor<StorageType>)
{
    elementwiseBinaryOp(left, right, result: result, op: { return $0 - $1 })
}

public func isub<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Tensor<StorageType>, right:Tensor<StorageType>)
{
    // general case is currently not supported
    assert(false)
}

public func isub<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Vector<StorageType>, right:Vector<StorageType>)
{
    // NxM + N
    assert(left.shape.elements == right.shape.elements)
    elementwiseBinaryOp(left, right, result: left, op: { $0 - $1 })
}

public func isub<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:ColumnVector<StorageType>)
{
    // NxM + N
    assert(left.shape[1] == right.shape[0])
    
    let rows = left.shape[0]
    let cols = left.shape[1]
    for i in 0..<rows {
        elementwiseBinaryOp(left[i, 0..<cols], right, result: left[i, 0..<cols], op: { $0 - $1 })
    }
}

public func isub<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:RowVector<StorageType>)
{
    // NxM + N
    assert(left.shape[0] == right.shape[0])
    
    let rows = left.shape[0]
    let cols = left.shape[1]
    for i in 0..<cols {
        elementwiseBinaryOp(left[0..<rows, i], right, result: left[0..<rows, i], op: { $0 - $1 })
    }
}

public func -<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Tensor<StorageType>, right:Tensor<StorageType>) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(shape: left.shape)
    sub(left: left, right: right, result: result)
    
    return result
}

public func -<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:StorageType.ElementType, right:Tensor<StorageType>) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(shape: right.shape)
    sub(left: left, right: right, result: result)
    
    return result
}


public func -=<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Tensor<StorageType>, right:Tensor<StorageType>)
{
    try! isub(left: left, right: right)
}


//
// elementwise division
//

public func div<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Tensor<StorageType>, right:Tensor<StorageType>, result:Tensor<StorageType>)
{
    elementwiseBinaryOp(left, right, result: result, op: { return $0 / $1 })
}


public func div<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Tensor<StorageType>, right:StorageType.ElementType, result:Tensor<StorageType>)
{
    elementwiseBinaryOp(left, right, result: result, op: { return $0 / $1 })
}


public func idiv<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Tensor<StorageType>, right:Tensor<StorageType>)
{
    // general case is currently not supported
    assert(false)
}

public func idiv<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Vector<StorageType>, right:Vector<StorageType>)
{
    // NxM + N
    assert(left.shape.elements == right.shape.elements)
    elementwiseBinaryOp(left, right, result: left, op: { $0 / $1 })
}

public func idiv<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:ColumnVector<StorageType>)
{
    // NxM + N
    assert(left.shape[1] == right.shape[0])
    
    let rows = left.shape[0]
    let cols = left.shape[1]
    for i in 0..<rows {
        elementwiseBinaryOp(left[i, 0..<cols], right, result: left[i, 0..<cols], op: { $0 / $1 })
    }
}

public func idiv<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:RowVector<StorageType>)
{
    // NxM + N
    assert(left.shape[0] == right.shape[0])
    
    let rows = left.shape[0]
    let cols = left.shape[1]
    for i in 0..<cols {
        elementwiseBinaryOp(left[0..<rows, i], right, result: left[0..<rows, i], op: { $0 / $1 })
    }
}

public func idiv<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Tensor<StorageType>, right:StorageType.ElementType)
{
    elementwiseBinaryOp(left, right, result: left, op: { $0 / $1 })
}

public func /<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Tensor<StorageType>, right:Tensor<StorageType>) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(shape: left.shape)
    div(left: left, right: right, result: result)
    
    return result
}

public func /=<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Tensor<StorageType>, right:Tensor<StorageType>)
{
    idiv(left: left, right: right)
}

public func /=<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Tensor<StorageType>, right:StorageType.ElementType)
{
    idiv(left: left, right: right)
}

//
// element-wise multiplication
//

public func mul<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Tensor<StorageType>, right:Tensor<StorageType>, result:Tensor<StorageType>)
{
    elementwiseBinaryOp(left, right, result: result, op: { return $0 * $1 })
}


public func mul<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Tensor<StorageType>, right:StorageType.ElementType, result:Tensor<StorageType>)
{
    elementwiseBinaryOp(left, right, result: result, op: { return $0 * $1 })
}


public func imul<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Tensor<StorageType>, right:Tensor<StorageType>)
{
    // general case is currently not supported
    assert(false)
}

public func imul<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Vector<StorageType>, right:Vector<StorageType>)
{
    // NxM + N
    assert(left.shape.elements == right.shape.elements)
    elementwiseBinaryOp(left, right, result: left, op: { $0 * $1 })
}

public func imul<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:ColumnVector<StorageType>)
{
    // NxM + N
    assert(left.shape[1] == right.shape[0])
    
    let rows = left.shape[0]
    let cols = left.shape[1]
    for i in 0..<rows {
        elementwiseBinaryOp(left[i, 0..<cols], right, result: left[i, 0..<cols], op: { $0 * $1 })
    }
}

public func imul<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:RowVector<StorageType>)
{
    // NxM + N
    assert(left.shape[0] == right.shape[0])
    
    let rows = left.shape[0]
    let cols = left.shape[1]
    for i in 0..<cols {
        elementwiseBinaryOp(left[0..<rows, i], right, result: left[0..<rows, i], op: { $0 * $1 })
    }
}

public func imul<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Tensor<StorageType>, right:StorageType.ElementType)
{
    elementwiseBinaryOp(left, right, result: left, op: { $0 * $1 })
}

public func *<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Tensor<StorageType>, right:Tensor<StorageType>) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(shape: left.shape)
    mul(left: left, right: right, result: result)
    
    return result
}

public func *=<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Tensor<StorageType>, right:Tensor<StorageType>)
{
    imul(left: left, right: right)
}

public func *=<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Tensor<StorageType>, right:StorageType.ElementType)
{
    imul(left: left, right: right)
}

//
// dot product
//

public func dot<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Tensor<StorageType>, right:Tensor<StorageType>, result:Tensor<StorageType>)
{
    // completely generic type currently unsupported
    assert(false)
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
    
    // per row
    for i in 0..<left.shape[0] {
        // per column
        for j in 0..<left.shape[1] {
            result[i] = result[i] + left[i, j]*right[j]
        }
    }
}

public func dot<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:Matrix<StorageType>, result:Tensor<StorageType>)
{
    assert(left.shape[1] == right.shape[0])
    
    // NxM x MxK -> NxK
    for n in 0..<left.shape[0] {
        for m in 0..<left.shape[1] {
            for k in 0..<right.shape[1] {
                result[n, k] = result[n, k] + left[n, m]*right[m, k]
            }
        }
    }
}

public func *<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:RowVector<StorageType>, right:ColumnVector<StorageType>) -> StorageType.ElementType
{
    return dot(left: left, right: right)
}

public func *<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Matrix<StorageType>, right:ColumnVector<StorageType>) -> RowVector<StorageType>
{
    let result = RowVector<StorageType>(cols: left.shape[0])
    dot(left: left, right: right, result: result)
    return result
}

public func *<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Matrix<StorageType>, right:Matrix<StorageType>) -> Vector<StorageType>
{
    let result = Vector<StorageType>(rows: right.shape[0])
    dot(left: left, right: right, result: result)
    return result
}

//
// outer product
//

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
    (left:ColumnVector<StorageType>, right:RowVector<StorageType>) -> Matrix<StorageType>
{
    let result = Matrix<StorageType>(rows: left.shape[0], cols: right.shape[0])
    outer(left: left, right: right, result: result)
    return result
}

public func *<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:StorageType.ElementType, right:Tensor<StorageType>) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(shape: right.shape)
    for i in right.storageIndices() {
        result.storage[i] = right.storage[i]*left
    }
    return result
}

public func *<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Tensor<StorageType>, right:StorageType.ElementType) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(shape: left.shape)
    for i in left.storageIndices() {
        result.storage[i] = left.storage[i]*right
    }
    return result
}

public func abs<StorageType:Storage where StorageType.ElementType:NumericType>
    (tensor:Tensor<StorageType>) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(shape: tensor.shape)
    for index in tensor.storageIndices() {
        result.storage[index] = abs(tensor.storage[index])
    }
    
    return result
}

public func isClose<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Tensor<StorageType>, _ right:Tensor<StorageType>, eps: StorageType.ElementType) -> Bool
{
    let diff = left - right
    let adiff:Tensor<StorageType> = abs(diff)
    for i in adiff.storageIndices() {
        if adiff.storage[i] >= eps { return false }
    }
    return true
}

// Misc (move?)
//public func hist<StorageType:Storage where StorageType.ElementType == Int>
//    (tensor:Tensor<StorageType>, bins:Int) -> Vector<StorageType>
//{
//    let h = Vector<StorageType>(rows: bins)
//    for i in tensor.storageIndices() {
//        let value:Int = tensor.storage[i]
//        h[value] = h[value] + 1
//    }
//
//    return h
//}

public func hist<StorageType:Storage where StorageType.ElementType == Double>
    (tensor:Tensor<StorageType>, bins:Int) -> Vector<StorageType>
{
    let h = Vector<StorageType>(rows: bins)
    let m = max(tensor)
    let delta = m/StorageType.ElementType(bins)
    for i in tensor.storageIndices() {
        let value = tensor.storage[i]
        var bin = Int(value/delta)
        if bin >= bins {  bin = bins-1 }
        h[bin] = h[bin] + 1
    }

    return h
}


// NB: no attempt was made to optimize these, and they need a lot of work to be fully functional

// axis = nil, means sum everything
// axis = Int chooses axis to sum along
public func sum<StorageType:Storage> // where StorageType.ElementType:NumericType>
    (tensor:Tensor<StorageType>, axis:Int?=nil) -> StorageType.ElementType
{
    if let ax = axis {
        // TODO
        assert(false)
        return 0
    } else {
        var total:StorageType.ElementType = 0
        for i in tensor.storageIndices() {
            total = total + tensor.storage[i]
        }
        
        return total
    }
}

public func max<StorageType:Storage where StorageType.ElementType:NumericType>
    (tensor:Tensor<StorageType>, axis:Int?=nil) -> StorageType.ElementType
{
    if let ax = axis {
        // TODO
        return 0
    } else {
        var index = 0
        var value:StorageType.ElementType?
        for i in tensor.storageIndices() {
            if value == nil || tensor.storage[i] > value! {
                index = i
                value = tensor.storage[i]
            }
        }
        return value!
    }
}

infix operator ^ {}

public func ^<StorageType:Storage where StorageType.ElementType == Double>
    (tensor:Tensor<StorageType>, p:Double) -> Tensor<StorageType>
{
    return pow(tensor, p)
}

public func pow<StorageType:Storage where StorageType.ElementType:NumericType>
    (tensor:Tensor<StorageType>, _ power:StorageType.ElementType) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(shape: tensor.shape)
    for i in result.storageIndices() {
        result.storage[i] = StorageType.ElementType.pow(tensor.storage[i], power)
    }
    
    return result
}

func exp<StorageType:Storage where StorageType.ElementType:NumericType>
    (tensor:Tensor<StorageType>) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(shape: tensor.shape)
    for i in result.storageIndices() {
        result.storage[i] = StorageType.ElementType.exp(tensor.storage[i])
    }
    
    return result
}


func norm<StorageType:Storage where StorageType.ElementType:NumericType>
    (tensor:Tensor<StorageType>) -> StorageType.ElementType
{
    let p = pow(tensor, StorageType.ElementType(2.0))
    let s:StorageType.ElementType = sum(p)
    return StorageType.ElementType.sqrt(s)
}
