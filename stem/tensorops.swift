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
    
    var indexResult = result.indices()
    
    // TODO: There should be better syntax to support this use-case
    for (lhs, rhs) in Zip2Sequence(left.indices(), right.indices()) {
        let idx = indexResult.next()!
        result[idx] = op(left: left[lhs], right: right[rhs])
    }
}

func elementwiseBinaryOp<StorageType:Storage>
    (left:Tensor<StorageType>, _ right:StorageType.ElementType, result:Tensor<StorageType>,
    op:(left:StorageType.ElementType, right:StorageType.ElementType) -> StorageType.ElementType)
{
    precondition(left.shape.elements == result.shape.elements)
    
    var indexResult = result.indices()
    for i in left.indices() {
        let idx = indexResult.next()!
        result[idx] = op(left: left[i], right: right)
    }
}

// TODO: provide code to allow broadcasting (need to check dimensions against each other)
func elementwiseBinaryOp<StorageType:Storage>
    (left:StorageType.ElementType, _ right:Tensor<StorageType>, result:Tensor<StorageType>,
    op:(left:StorageType.ElementType, right:StorageType.ElementType) -> StorageType.ElementType)
{
//    precondition(right.shape.elements == result.shape.elements)
    
    if right.shape.elements == result.shape.elements {
        var indexResult = result.indices()
        for i in right.indices() {
            let idx = indexResult.next()!
            result[idx] = op(left: left, right: right[i])
        }
    } else {
        // broadcast
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

func isVector(type:TensorType) -> Bool {
    return type == .Vector || type == .RowVector || type == .ColumnVector
}

func isMatrix(type:TensorType) -> Bool {
    return type == .Matrix
}

func isCube(type:TensorType) -> Bool {
    return type == .Cube
}


// TODO: benchmark this against specific cases .. not sure
// the specific cases are faster (and therefore should potentially
// be removed)
public func dot<S:Storage where S.ElementType:NumericType>
    (left left:Tensor<S>, right:Tensor<S>, result:Tensor<S>)
{
    precondition(left.shape.span < 3)
    precondition(right.shape.span < 3)
    precondition(left.shape[1] == right.shape[0], "Number of rows in vector must match number of rows of matrix")
    precondition(left.shape[0] == result.shape[0], "Number of rows of result must match rows in matrix")
    
    for n in 0..<left.shape[0] {
        for m in 0..<right.shape[0] {
            for k in 0..<right.shape[1] {
                result[n, k] = result[n, k] + left[n, m]*right[m, k]
            }
        }
    }
}

public func dot<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Vector<StorageType>, right:Vector<StorageType>) -> StorageType.ElementType
{
//    if left.shape.elements != right.shape.elements {
//        throw TensorError.SizeMismatch(lhs: left.shape, rhs: right.shape)
//    }
    precondition(left.shape.elements == right.shape.elements, "Number of elements must match")
    
    var result:StorageType.ElementType = 0
    
    let indexLeft = left.indices()
    let indexRight = right.indices()
    
    for (l, r) in Zip2Sequence(indexLeft, indexRight) {
        result = result + left[l]*right[r]
    }

    return result
}

// transform: nxm * mx1 -> nx1
public func dot<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Matrix<StorageType>, right:ColumnVector<StorageType>, result:ColumnVector<StorageType>)
{
    precondition(left.shape[1] == right.shape[0], "Number of rows in vector must match number of rows of matrix")
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
    dot(left: left, right: right, result: result)
    return result
}

//
// outer product
//

public func outer<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Vector<StorageType>, right:Vector<StorageType>, result:Tensor<StorageType>)
{
    let indexLeft = left.indices()
    let indexRight = right.indices()
    var indexResult = result.indices()
    
//    var o = GeneratorSequence(indexResult)
    for l in indexLeft {
        for r in indexRight {
            let pos = indexResult.next()!
            result[pos] = left[l]*right[r]
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
    for index in tensor.indices() {
        result[index] = abs(tensor[index])
    }
    
    return result
}

public func isClose<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Tensor<StorageType>, _ right:Tensor<StorageType>, eps: StorageType.ElementType) -> Bool
{
    let diff = left - right
    let adiff:Tensor<StorageType> = abs(diff)
    for i in adiff.indices() {
        if adiff[i] >= eps { return false }
    }
    return true
}


public func hist<StorageType:Storage where StorageType.ElementType == Double>
    (tensor:Tensor<StorageType>, bins:Int) -> Vector<StorageType>
{
    let h = Vector<StorageType>(rows: bins)
    let m = max(tensor)
    let delta = m/StorageType.ElementType(bins)
    for i in tensor.indices() {
        let value = tensor[i]
        var bin = Int(value/delta)
        if bin >= bins {  bin = bins-1 }
        h[bin] = h[bin] + 1
    }

    return h
}

func reduce<StorageType:Storage>(tensor:Tensor<StorageType>,
            axis:Int,
            op:(StorageType.ElementType, StorageType.ElementType) -> StorageType.ElementType)
    -> Tensor<StorageType>
{
    precondition(axis < tensor.shape.count)
    
    // calculate new shape
    let reduced:[(index:Int, element:Int)] = tensor.shape
        .enumerate()
        .filter { $0.index != axis }
    
    let indices = reduced.map { $0.index }
    let newShape = reduced.map { $0.element }
    
    // create tensor to hold results
    let result = Tensor<StorageType>(shape: Extent(newShape))
    
    // index `i` is the axis we are summing along
    for i in 0..<tensor.shape[axis] {
        let oldIndices = GeneratorSequence(IndexGenerator(tensor.shape, dimIndex: indices))
        let newIndices = GeneratorSequence(IndexGenerator(result.shape))
        
        for (var tensorIndex, resultIndex) in Zip2Sequence(oldIndices, newIndices) {
            tensorIndex[axis] = i
            let tensorOffset:Int = tensor.calculateOffset(tensorIndex)            
            let resultOffset:Int = result.calculateOffset(resultIndex)
            
            result.storage[resultOffset] = op(result.storage[resultOffset], tensor.storage[tensorOffset])
        }
    }
    
    return result
}

func reduce<StorageType:Storage>(tensor:Tensor<StorageType>,
            op:(StorageType.ElementType, StorageType.ElementType) -> StorageType.ElementType)
    -> StorageType.ElementType
{
    var result:StorageType.ElementType = 0
    
    let indices = GeneratorSequence(IndexGenerator(tensor.shape))
    for index in indices {
        result = op(result, tensor[index])
    }
    
    return result
}

public func sum<StorageType:Storage>
    (tensor:Tensor<StorageType>, axis:Int) -> Tensor<StorageType>
{
    return reduce(tensor, axis: axis, op: +)
}

public func sum<StorageType:Storage>(tensor:Tensor<StorageType>) -> StorageType.ElementType {
    return reduce(tensor, op: +)
}

public func max<StorageType:Storage where StorageType.ElementType:NumericType>
    (tensor:Tensor<StorageType>, axis:Int) -> Tensor<StorageType>
{
    return reduce(tensor, axis: axis, op: max)
}

public func max<StorageType:Storage where StorageType.ElementType:NumericType>
    (tensor:Tensor<StorageType>) -> StorageType.ElementType
{
    return reduce(tensor, op: max)
}

public func min<StorageType:Storage where StorageType.ElementType:NumericType>
    (tensor:Tensor<StorageType>, axis:Int) -> Tensor<StorageType>
{
    return reduce(tensor, axis: axis, op: min)
}

public func min<StorageType:Storage where StorageType.ElementType:NumericType>
    (tensor:Tensor<StorageType>) -> StorageType.ElementType
{
    return reduce(tensor, op: min)
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
    for i in result.indices() {
        result[i] = StorageType.ElementType.pow(tensor[i], power)
    }
    
    return result
}

func exp<StorageType:Storage where StorageType.ElementType:FloatNumericType>
    (tensor:Tensor<StorageType>) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(shape: tensor.shape)
    for i in result.indices() {
        result[i] = StorageType.ElementType.exp(tensor[i])
    }
    
    return result
}

func sqrt(tensor:TensorType) -> TensorType
{
    preconditionFailure()
}

func sqrt<StorageType:Storage where StorageType.ElementType:FloatNumericType>
    (tensor:Tensor<StorageType>) -> Tensor<StorageType>
{
    let indices = tensor.indices()
    let result = Tensor<StorageType>(shape: tensor.shape)
    for index in indices {
        result[index] = StorageType.ElementType.sqrt(tensor[index])
    }
    
    return result
}

//func sqrt<StorageType:Storage where StorageType.ElementType:FloatNumericType>
//    (scalar:TensorScalar<StorageType>) -> TensorScalar<StorageType>
//{
//    return TensorScalar<StorageType>(StorageType.ElementType.sqrt(scalar.value))
//}

func norm<StorageType:Storage where StorageType.ElementType:FloatNumericType>
    (tensor:Tensor<StorageType>, axis:Int) -> Tensor<StorageType>
{
    let p = pow(tensor, StorageType.ElementType(2.0))
    let s = sum(p, axis: axis)
    return sqrt(s)
}

func sigmoid<StorageType:Storage where StorageType.ElementType:FloatNumericType>
    (input:Tensor<StorageType>, output:Tensor<StorageType>)
{
    precondition(input.shape == output.shape)
    for index in input.indices() {
        output[index] = StorageType.ElementType(1.0) / (StorageType.ElementType(1.0) + StorageType.ElementType.exp(-input[index]))
    }
}
