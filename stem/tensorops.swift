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
    (left left:Tensor<StorageType>, right:Tensor<StorageType>) throws
{
    // general case is currently not supported
    throw IllegalOperation()
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

public func isub<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Tensor<StorageType>, right:Tensor<StorageType>) throws
{
    // general case is currently not supported
    throw IllegalOperation()
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

public func -=<StorageType:Storage where StorageType.ElementType:NumericType>
    (left:Tensor<StorageType>, right:Tensor<StorageType>)
{
    try! isub(left: left, right: right)
}


//
// dot product
//

public func dot<StorageType:Storage where StorageType.ElementType:NumericType>
    (left left:Tensor<StorageType>, right:Tensor<StorageType>, result:Tensor<StorageType>) throws
{
    // completely generic type currently unsupported
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
