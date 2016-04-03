//
//  tensorops.swift
//  stem
//
//  Created by Abe Schneider on 12/9/15.
//  Copyright © 2015 none. All rights reserved.
//

import Foundation

infix operator ** { associativity left precedence 200 }
infix operator ⊗ {associativity left precedence 100 }
infix operator ⊙ {associativity left precedence 100 }

func elementwiseBinaryOp<StorageType:Storage>
    (left:Tensor<StorageType>, _ right:Tensor<StorageType>, result:Tensor<StorageType>,
    op:(left:StorageType.ElementType, right:StorageType.ElementType) -> StorageType.ElementType)
{
    precondition(left.shape.elements == right.shape.elements)
    precondition(left.shape.elements == result.shape.elements)
    
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
    precondition(left.shape.elements == result.shape.elements)
    
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
    precondition(right.shape.elements == result.shape.elements)
    
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
    precondition(left.shape == right.shape, "Tensor dimensions must match")
    
    elementwiseBinaryOp(left, right, result: result, op: { $0 + $1 })
}

public func add<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:RowVector<StorageType>, result:Matrix<StorageType>)
{
    // NxM + 1xN
    precondition(left.shape[1] == right.shape[1], "Number of columns must match")
    precondition(result.shape[0] == left.shape[0], "Number of rows must match")
    precondition(result.shape[1] == left.shape[1], "Number of columns must match")
    
    let rows = left.shape[0]
    let cols = left.shape[1]
    for i in 0..<rows {
        elementwiseBinaryOp(left[i, 0..<cols], right, result: result[i, 0..<cols], op: { $0 + $1 })
    }
}

public func add<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:ColumnVector<StorageType>, result:Matrix<StorageType>)
{
    // NxM + M
    precondition(left.shape[0] == right.shape[1], "Vector must have same number columns as rows in Matrix")
    
    let rows = left.shape[0]
    let cols = left.shape[1]
    
    for i in 0..<cols {
        elementwiseBinaryOp(left[0..<rows, i], right, result: result[0..<rows, i], op: { $0 + $1 })
    }
}

public func iadd<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Tensor<StorageType>, right:Tensor<StorageType>)
{
    // TODO: need to just make sure Tensors are same shape
    assert(false)
}

public func iadd<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Vector<StorageType>, right:Vector<StorageType>)
{
    precondition(left.shape.elements == right.shape.elements)
    elementwiseBinaryOp(left, right, result: left, op: { $0 + $1 })
}

public func iadd<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:ColumnVector<StorageType>)
{
    // NxM + Mx1
    precondition(left.shape[0] == right.shape[0], "Number of columns must match")
    
    let rows = left.shape[0]
    let cols = left.shape[1]
    for i in 0..<cols {
        elementwiseBinaryOp(left[0..<rows, i], right, result: left[0..<rows, i], op: { $0 + $1 })
    }
}

public func iadd<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:RowVector<StorageType>)
{
    // NxM + N
    precondition(left.shape[0] == right.shape[0], "Number of rows much match")
    
    let rows = left.shape[0]
    let cols = left.shape[1]
    for i in 0..<rows {
        elementwiseBinaryOp(left[i, 0..<cols], right, result: left[i, 0..<cols], op: { $0 + $1 })
    }

}

public func +<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Tensor<StorageType>, right:Tensor<StorageType>) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(shape: left.shape)
    add(left: left, right: right, result: result)
    
    return result
}

public func +<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:RowVector<StorageType>, right:RowVector<StorageType>) -> RowVector<StorageType>
{
    let result = RowVector<StorageType>(cols: left.shape[1])
    add(left: left, right: right, result: result)
    
    return result
}

public func +<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:ColumnVector<StorageType>, right:ColumnVector<StorageType>) -> ColumnVector<StorageType>
{
    let result = ColumnVector<StorageType>(rows: left.shape[0])
    add(left: left, right: right, result: result)
    
    return result
}



public func +=<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Tensor<StorageType>, right:Tensor<StorageType>)
{
    iadd(left: left, right: right)
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
    precondition(left.shape.elements == right.shape.elements)
    elementwiseBinaryOp(left, right, result: left, op: { $0 - $1 })
}

public func isub<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:ColumnVector<StorageType>)
{
    // NxM + N
    precondition(left.shape[1] == right.shape[0])
    
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
    precondition(left.shape[0] == right.shape[0])
    
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
    isub(left: left, right: right)
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
    // TODO
    assert(false)
}

public func idiv<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Vector<StorageType>, right:Vector<StorageType>)
{
    // NxM + N
    precondition(left.shape.elements == right.shape.elements)
    elementwiseBinaryOp(left, right, result: left, op: { $0 / $1 })
}

public func idiv<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:ColumnVector<StorageType>)
{
    // NxM + N
    precondition(left.shape[1] == right.shape[0])
    
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
    precondition(left.shape[0] == right.shape[0])
    
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
    (left left:Matrix<StorageType>, right:ColumnVector<StorageType>, result:Matrix<StorageType>)
{
    // NxM + N
    precondition(left.shape[0] == right.shape[0])
    
    let rows = left.shape[0]
    let cols = left.shape[1]
    for i in 0..<cols {
        elementwiseBinaryOp(left[0..<rows, i], right, result: result[0..<rows, i], op: { $0 * $1 })
    }
}

public func mul<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:RowVector<StorageType>, result:Matrix<StorageType>)
{
    // NxM + N
    precondition(left.shape[0] == right.shape[0])
    
    let rows = left.shape[0]
    let cols = left.shape[1]
    for i in 0..<rows {
        elementwiseBinaryOp(left[i, 0..<cols], right, result: result[i, 0..<cols], op: { $0 * $1 })
    }
}

public func mul<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Tensor<StorageType>, right:StorageType.ElementType, result:Tensor<StorageType>)
{
    elementwiseBinaryOp(left, right, result: result, op: { return $0 * $1 })
}

public func mul<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:StorageType.ElementType, right:Tensor<StorageType>, result:Tensor<StorageType>)
{
    elementwiseBinaryOp(left, right, result: result, op: { return $0 * $1 })
}

public func imul<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Tensor<StorageType>, right:Tensor<StorageType>)
{
    // TODO
    assert(false)
}

public func imul<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Vector<StorageType>, right:Vector<StorageType>)
{
    // NxM + N
    precondition(left.shape.elements == right.shape.elements)
    elementwiseBinaryOp(left, right, result: left, op: { $0 * $1 })
}

public func imul<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:ColumnVector<StorageType>)
{
    // NxM + N
    precondition(left.shape[1] == right.shape[0])
    
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
    precondition(left.shape[0] == right.shape[0])
    
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

public func *<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Matrix<StorageType>, right:RowVector<StorageType>) -> Tensor<StorageType>
{
    let result = Matrix<StorageType>(shape: left.shape)
    mul(left: left, right: right, result: result)
    
    return result
}

public func *<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Matrix<StorageType>, right:ColumnVector<StorageType>) -> Tensor<StorageType>
{
//    if left.shape[0] != right.shape[0] {
//        throw TensorError.SizeMismatch(lhs: left.shape, rhs: right.shape)
//    }
    precondition(left.shape[0] == right.shape[0], "Number of rows of vector must match number of rows of matrix")
    
    let result = Matrix<StorageType>(shape: left.shape)
    mul(left: left, right: right, result: result)
    
    return result
}

public func *<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:StorageType.ElementType, right:Tensor<StorageType>) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(shape: right.shape)
    mul(left: left, right: right, result: result)

    return result
}

public func *<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Tensor<StorageType>, right:StorageType.ElementType) -> Tensor<StorageType>
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
//    throw TensorError.IllegalOperation
    assertionFailure()
}

// generalized form (must check manually that vectors are properly aligned)
public func dot<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Vector<StorageType>, right:Vector<StorageType>) -> StorageType.ElementType
{
//    if left.shape.elements != right.shape.elements {
//        throw TensorError.SizeMismatch(lhs: left.shape, rhs: right.shape)
//    }
    precondition(left.shape.elements == right.shape.elements, "Number of elements must match")
    
    var result:StorageType.ElementType = 0
    
    let indexLeft = left.storageIndices()
    let indexRight = right.storageIndices()
    
    for (l, r) in Zip2Sequence(GeneratorSequence(indexLeft), GeneratorSequence(indexRight)) {
        result = result + left.storage[l]*right.storage[r]
    }

    return result
}

// transform: nxm * mx1 -> nx1
public func dot<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:ColumnVector<StorageType>, result:ColumnVector<StorageType>)
{
//    if left.shape[1] != right.shape[0] {
//        throw TensorError.SizeMismatch(lhs: left.shape, rhs: right.shape)
//    }
    precondition(left.shape[1] == right.shape[0], "Number of rows in vector must match number of rows of matrix")
    
//    if left.shape[0] != result.shape[0] {
//        throw TensorError.SizeMismatch(lhs: left.shape, rhs: result.shape)
//    }
    precondition(left.shape[0] == result.shape[0], "Number of rows of result must match rows in matrix")
    
    // per row
    for i in 0..<left.shape[0] {
        // per column
        for j in 0..<left.shape[1] {
            result[i] = result[i] + left[i, j]*right[j]
        }
    }
}

// NxM * MxK -> NxK
public func dot<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:Matrix<StorageType>, result:Tensor<StorageType>)
{
//    if left.shape[1] != right.shape[0] {
//        throw TensorError.SizeMismatch(lhs: left.shape, rhs: right.shape)
//    }
    precondition(left.shape[1] == right.shape[0], "Number of columns must match number of rows in matrices")
    
    for n in 0..<left.shape[0] {
        for m in 0..<left.shape[1] {
            for k in 0..<right.shape[1] {
                result[n, k] = result[n, k] + left[n, m]*right[m, k]
            }
        }
    }
}

public func ⊙<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:RowVector<StorageType>, right:ColumnVector<StorageType>) -> StorageType.ElementType
{
    return dot(left: left, right: right)
}

public func ⊙<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Matrix<StorageType>, right:ColumnVector<StorageType>) -> ColumnVector<StorageType>
{
    let result = ColumnVector<StorageType>(rows: left.shape[0])
    dot(left: left, right: right, result: result)
    return result
}

public func ⊙<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Matrix<StorageType>, right:Matrix<StorageType>) -> Matrix<StorageType>
{
    let result = Matrix<StorageType>(rows: left.shape[0], cols: right.shape[1])
    try dot(left: left, right: right, result: result)
    return result
}

//
// outer product
//

public func outer<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Vector<StorageType>, right:Vector<StorageType>, result:Tensor<StorageType>)
{
    let indexLeft = left.storageIndices()
    let indexRight = right.storageIndices()
    let indexResult = result.storageIndices()
    
    var o = GeneratorSequence(indexResult)
    for l in GeneratorSequence(indexLeft) {
        for r in GeneratorSequence(indexRight ){
            let pos = o.next()!
            result.storage[pos] = left.storage[l]*right.storage[r]
        }
    }
}

public func ⊗<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Vector<StorageType>, right:Vector<StorageType>) -> Matrix<StorageType>
{
    let result = Matrix<StorageType>(rows: left.shape.elements, cols: right.shape.elements)
    outer(left: left, right: right, result: result)
    return result
}

public func ⊗<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:ColumnVector<StorageType>, right:RowVector<StorageType>) -> Matrix<StorageType>
{
    let result = Matrix<StorageType>(rows: left.shape[0], cols: right.shape[1])
    outer(left: left, right: right, result: result)
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


// provide two versions of this .. one that returns a scalar and one that returns a Tensor?
//public func reduce<StorageType:Storage>
//    (tensor:Tensor<StorageType>, axis:Int) -> Tensor<StorageType>
//{
//    var shape = tensor.shape
//    shape[axis] = 1
//    var result = Tensor<StorageType>(shape: shape)
//    
//}

// axis = nil, means sum everything
// axis = Int chooses axis to sum along
public func sum<StorageType:Storage> // where StorageType.ElementType:NumericType>
    (tensor:Tensor<StorageType>, axis:Int?=nil) -> StorageType.ElementType
{
    if let ax = axis {
        // need to verify is valid
        
        var shape = tensor.shape
        shape[ax] = 1
        var result = Tensor<StorageType>(shape: shape)
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

public func **<StorageType:Storage where StorageType.ElementType:FloatNumericType>
    (tensor:Tensor<StorageType>, power:StorageType.ElementType) -> Tensor<StorageType>
{
    return pow(tensor, power)
}

public func pow<StorageType:Storage where StorageType.ElementType:FloatNumericType>
    (tensor:Tensor<StorageType>, _ power:StorageType.ElementType) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(shape: tensor.shape)
    for i in result.storageIndices() {
        result.storage[i] = StorageType.ElementType.pow(tensor.storage[i], power)
    }
    
    return result
}

func exp<StorageType:Storage where StorageType.ElementType:FloatNumericType>
    (tensor:Tensor<StorageType>) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(shape: tensor.shape)
    for i in result.storageIndices() {
        result.storage[i] = StorageType.ElementType.exp(tensor.storage[i])
    }
    
    return result
}


func norm<StorageType:Storage where StorageType.ElementType:FloatNumericType>
    (tensor:Tensor<StorageType>, axis:Int?=nil) -> StorageType.ElementType
{
    let p = pow(tensor, StorageType.ElementType(2.0))
    let s:StorageType.ElementType = sum(p, axis: axis)
    return StorageType.ElementType.sqrt(s)
}
