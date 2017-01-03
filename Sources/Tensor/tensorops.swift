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

/**
 Performs binary operation on `lhs` and `rhs`, storing results in `result`.
 
 - Parameter lhs: Tensor
 - Parameter rhs: Tensor
 - Parameter result: Tensor to place results of addition
 - Parameter op: Operation to perform
 */
func elementwiseBinaryOp<S:Storage>
    (_ left:Tensor<S>, _ right:Tensor<S>, result:Tensor<S>,
    op:(_ left:S.ElementType, _ right:S.ElementType) -> S.ElementType)
{
    precondition(left.shape.elements == right.shape.elements)
    precondition(left.shape.elements == result.shape.elements)
    
    var indexResult = result.indices()
    
    // TODO: There should be better syntax to support this use-case
    for (lhs, rhs) in zip(left.indices(), right.indices()) {
        let idx = indexResult.next()!
        result[idx] = op(left[lhs], right[rhs])
    }
}

/**
 Performs binary operation on `lhs` and `rhs`, storing results in `result`.
     
 - Parameter lhs: Tensor
 - Parameter rhs: NumericType
 - Parameter result: Tensor to place results of addition
 - Parameter op: Operation to perform
 */
func elementwiseBinaryOp<S:Storage>
    (_ left:Tensor<S>, _ right:S.ElementType, result:Tensor<S>,
    op:(_ left:S.ElementType, _ right:S.ElementType) -> S.ElementType)
{
    precondition(left.shape.elements == result.shape.elements)
    
    var indexResult = result.indices()
    for i in left.indices() {
        let idx = indexResult.next()!
        result[idx] = op(left[i], right)
    }
}

/**
 Performs binary operation on `lhs` and `rhs`, storing results in `result`.

 - Parameter lhs: NumericType
 - Parameter rhs: Tensor
 - Parameter result: Tensor to place results of addition
 - Parameter op: Operation to perform
 */
func elementwiseBinaryOp<StorageType:Storage>
    (_ left:StorageType.ElementType, _ right:Tensor<StorageType>, result:Tensor<StorageType>,
    op:(_ left:StorageType.ElementType, _ right:StorageType.ElementType) -> StorageType.ElementType)
{
    precondition(right.shape.elements == result.shape.elements)
    
    var indexResult = result.indices()
    for i in right.indices() {
        let idx = indexResult.next()!
        result[idx] = op(left, right[i])
    }
}

//
// Addition
//

/**
 Adds `lhs` to `rhs` and puts resulting value in `result`. If their shapes don't match, calls `broadcast`.
 
 - Parameter lhs: Tensor
 - Parameter rhs: Tensor
 - Parameter result: Tensor to place results of addition
 */
public func add<S:Storage>
    (_ lhs:Tensor<S>, _ rhs:Tensor<S>, result:Tensor<S>) where S.ElementType:NumericType
{
    if lhs.shape == rhs.shape {
        elementwiseBinaryOp(lhs, rhs, result: result, op: { $0 + $1 })
    } else {
        let (l, r) = broadcast(lhs, rhs)
        elementwiseBinaryOp(l, r, result: result, op: { $0 + $1 })
    }
}

/**
 Adds `lhs` to `rhs` and puts resulting value in `result`.
 
 - Parameter lhs: Tensor
 - Parameter rhs: NumericType
 - Parameter result: Tensor to place results of addition
 */
public func add<S:Storage>
    (_ lhs:Tensor<S>, _ rhs:S.ElementType, result:Tensor<S>) where S.ElementType:NumericType
{
    elementwiseBinaryOp(lhs, rhs, result: result, op: { $0 + $1 })
}

/**
 Adds `lhs` to `rhs` and puts resulting value in `result`.
 
 - Parameter lhs: NumericType
 - Parameter rhs: Tensor
 - Parameter result: Tensor to place results of addition
 */
public func add<S:Storage>
    (_ lhs:S.ElementType, _ rhs:Tensor<S>, result:Tensor<S>) where S.ElementType:NumericType
{
    elementwiseBinaryOp(lhs, rhs, result: result, op: { $0 + $1 })
}


/**
 Adds `rhs` to `lhs` in-place. If their shapes don't match, calls `broadcast`.
 
 - Parameter lhs: Tensor
 - Parameter rhs: Tensor
 */
public func iadd<S:Storage>
    (_ lhs:Tensor<S>, _ rhs:Tensor<S>) where S.ElementType:NumericType
{
    if lhs.shape == rhs.shape {
        elementwiseBinaryOp(lhs, rhs, result: lhs, op: { $0 + $1 })
    } else {
        let (l, r) = broadcast(lhs, rhs)
        elementwiseBinaryOp(l, r, result: lhs, op: { $0 + $1 })
    }
}

/**
 Adds `rhs` to `lhs` in-place.
 
 - Parameter lhs: Tensor
 - Parameter rhs: NumericType
 */
public func iadd<S:Storage>
    (_ lhs:Tensor<S>, _ rhs:S.ElementType) where S.ElementType:NumericType
{
    elementwiseBinaryOp(lhs, rhs, result: lhs, op: { $0 + $1 })
}

/**
 Returns the addition of `lhs` to `rhs`. If their shapes don't match, calls broadcast.
 
 - Parameter lhs: Tensor
 - Parameter rhs: Tensor
 - Returns: Results of `lhs` + `rhs`
 */
public func +<S:Storage>
    (lhs:Tensor<S>, rhs:Tensor<S>) -> Tensor<S> where S.ElementType:NumericType
{
    let result = Tensor<S>(lhs.shape)
    add(lhs, rhs, result: result)
    
    return result
}

/**
 Returns the addition of `lhs` to `rhs`. If their shapes don't match, calls broadcast.
 
 - Parameter lhs: Tensor
 - Parameter rhs: NumericType
 - Returns: Results of `lhs` + `rhs`
 */
public func +<S:Storage>
    (lhs:Tensor<S>, rhs:S.ElementType) -> Tensor<S> where S.ElementType:NumericType
{
    let result = Tensor<S>(lhs.shape)
    add(lhs, rhs, result: result)
    
    return result
}

/**
 Returns the addition of `lhs` to `rhs`. If their shapes don't match, calls broadcast.
 
 - Parameter lhs: Tensor
 - Parameter rhs: NumericType
 - Returns: Results of `lhs` + `rhs`
 */
public func +<S:Storage>
    (lhs:S.ElementType, rhs:Tensor<S>) -> Tensor<S> where S.ElementType:NumericType
{
    let result = Tensor<S>(rhs.shape)
    add(lhs, rhs, result: result)
    
    return result
}

/**
 Adds `rhs` to `lhs` in-place. If their shapes don't match, calls `broadcast`.
 
 - Parameter lhs: Tensor
 - Parameter rhs: Tensor
 */
public func +=<S:Storage>
    (lhs:Tensor<S>, rhs:Tensor<S>) where S.ElementType:NumericType
{
    iadd(lhs, rhs)
}

/**
 Adds `rhs` to `lhs` in-place. If their shapes don't match, calls `broadcast`.
 
 - Parameter lhs: Tensor
 - Parameter rhs: Tensor
 */
public func +=<S:Storage>
    (lhs:Tensor<S>, rhs:S.ElementType) where S.ElementType:NumericType
{
    iadd(lhs, rhs)
}

//
// Subtraction
//

/**
 Subtracts `rhs` to `lhs` in-place. If their shapes don't match, calls `broadcast`.
 
 - Parameter lhs: Tensor
 - Parameter rhs: Tensor
 */
public func sub<StorageType:Storage>
    (_ lhs:Tensor<StorageType>, _ rhs:Tensor<StorageType>, result:Tensor<StorageType>) where StorageType.ElementType:NumericType
{
    if lhs.shape == rhs.shape {
        elementwiseBinaryOp(lhs, rhs, result: result, op: { return $0 - $1 })
    } else {
        let (l, r) = broadcast(lhs, rhs)
        elementwiseBinaryOp(l, r, result: result, op: { return $0 - $1 })
    }
    
}

public func sub<StorageType:Storage>
    (_ lhs:StorageType.ElementType, _ rhs:Tensor<StorageType>, result:Tensor<StorageType>) where StorageType.ElementType:NumericType
{
    elementwiseBinaryOp(lhs, rhs, result: result, op: { return $0 - $1 })
}

public func isub<StorageType:Storage>
    (_ lhs:Tensor<StorageType>, _ rhs:Tensor<StorageType>) where StorageType.ElementType:NumericType
{
    if lhs.shape == rhs.shape {
        elementwiseBinaryOp(lhs, rhs, result: lhs, op: { $0 - $1 })
    } else {
        let (l, r) = broadcast(lhs, rhs)
        elementwiseBinaryOp(l, r, result: lhs, op: { $0 - $1 })
    }
}

public func -<StorageType:Storage>
    (lhs:Tensor<StorageType>, rhs:Tensor<StorageType>) -> Tensor<StorageType> where StorageType.ElementType:NumericType
{
    let result = Tensor<StorageType>(lhs.shape)
    sub(lhs, rhs, result: result)
    
    return result
}

public func -<StorageType:Storage>
    (lhs:StorageType.ElementType, rhs:Tensor<StorageType>) -> Tensor<StorageType> where StorageType.ElementType:NumericType
{
    let result = Tensor<StorageType>(rhs.shape)
    sub(lhs, rhs, result: result)
    
    return result
}


public func -=<StorageType:Storage>
    (lhs:Tensor<StorageType>, rhs:Tensor<StorageType>) where StorageType.ElementType:NumericType
{
    isub(lhs, rhs)
}


//
// elementwise division
//

public func div<StorageType:Storage>
    (_ lhs:Tensor<StorageType>, _ rhs:Tensor<StorageType>, result:Tensor<StorageType>) where StorageType.ElementType:NumericType
{
    elementwiseBinaryOp(lhs, rhs, result: result, op: { return $0 / $1 })
}


public func div<StorageType:Storage>
    (_ lhs:Tensor<StorageType>, _ rhs:StorageType.ElementType, result:Tensor<StorageType>) where StorageType.ElementType:NumericType
{
    elementwiseBinaryOp(lhs, rhs, result: result, op: { return $0 / $1 })
}


public func idiv<StorageType:Storage>
    (_ lhs:Tensor<StorageType>, _ rhs:Tensor<StorageType>) where StorageType.ElementType:NumericType
{
    if lhs.shape == rhs.shape {
        elementwiseBinaryOp(lhs, rhs, result: lhs, op: { $0 / $1 })
    } else {
        let (l, r) = broadcast(lhs, rhs)
        elementwiseBinaryOp(l, r, result: lhs, op: { $0 / $1 })
    }
}

public func idiv<StorageType:Storage>
    (_ lhs:Tensor<StorageType>, _ rhs:StorageType.ElementType) where StorageType.ElementType:NumericType
{
    elementwiseBinaryOp(lhs, rhs, result: lhs, op: { $0 / $1 })
}

public func /<StorageType:Storage>
    (lhs:Tensor<StorageType>, rhs:Tensor<StorageType>) -> Tensor<StorageType> where StorageType.ElementType:NumericType
{
    let shape = max(lhs.shape, rhs.shape)
    let result = Tensor<StorageType>(shape)
    div(lhs, rhs, result: result)
    return result
}

public func /=<StorageType:Storage>
    (lhs:Tensor<StorageType>, rhs:Tensor<StorageType>) where StorageType.ElementType:NumericType
{
    idiv(lhs, rhs)
}

public func /=<StorageType:Storage>
    (lhs:Tensor<StorageType>, rhs:StorageType.ElementType) where StorageType.ElementType:NumericType
{
    idiv(lhs, rhs)
}

//
// element-wise multiplication
//

public func mul<StorageType:Storage>
    (_ lhs:Tensor<StorageType>, _ rhs:Tensor<StorageType>, result:Tensor<StorageType>) where StorageType.ElementType:NumericType
{
    if lhs.shape == rhs.shape {
        elementwiseBinaryOp(lhs, rhs, result: result, op: { return $0 * $1 })
    } else {
        let (l, r) = broadcast(lhs, rhs)
        elementwiseBinaryOp(l, r, result: result, op: { return $0 * $1 })
    }
}

public func mul<StorageType:Storage>
    (_ lhs:Tensor<StorageType>, rhs:StorageType.ElementType, result:Tensor<StorageType>) where StorageType.ElementType:NumericType
{
    elementwiseBinaryOp(lhs, rhs, result: result, op: { return $0 * $1 })
}

public func mul<StorageType:Storage>
    (_ lhs:StorageType.ElementType, _ rhs:Tensor<StorageType>, result:Tensor<StorageType>) where StorageType.ElementType:NumericType
{
    elementwiseBinaryOp(lhs, rhs, result: result, op: { return $0 * $1 })
}

public func mul<StorageType:Storage>
    (_ lhs:Tensor<StorageType>, _ rhs:StorageType.ElementType, result:Tensor<StorageType>) where StorageType.ElementType:NumericType
{
    elementwiseBinaryOp(lhs, rhs, result: result, op: { return $0 * $1 })
}

public func imul<StorageType:Storage>
    (_ lhs:Tensor<StorageType>, _ rhs:Tensor<StorageType>) where StorageType.ElementType:NumericType
{
    if lhs.shape == rhs.shape {
        elementwiseBinaryOp(lhs, rhs, result: lhs, op: { $0 * $1 })
    } else {
        let (l, r) = broadcast(lhs, rhs)
        elementwiseBinaryOp(l, r, result: lhs, op: { $0 * $1 })
    }
}

public func imul<StorageType:Storage>
    (_ lhs:Tensor<StorageType>, _ rhs:StorageType.ElementType) where StorageType.ElementType:NumericType
{
    elementwiseBinaryOp(lhs, rhs, result: lhs, op: { $0 * $1 })
}

public func *<StorageType:Storage>
    (lhs:Tensor<StorageType>, rhs:Tensor<StorageType>) -> Tensor<StorageType> where StorageType.ElementType:NumericType
{
    let shape = max(lhs.shape, rhs.shape)
    let result = Tensor<StorageType>(shape)
    
    mul(lhs, rhs, result: result)
    
    return result
}

public func *<StorageType:Storage>
    (lhs:StorageType.ElementType, rhs:Tensor<StorageType>) -> Tensor<StorageType> where StorageType.ElementType:NumericType
{
    let result = Tensor<StorageType>(rhs.shape)
    mul(lhs, rhs, result: result)

    return result
}

public func *<StorageType:Storage>
    (lhs:Tensor<StorageType>, rhs:StorageType.ElementType) -> Tensor<StorageType> where StorageType.ElementType:NumericType
{
    let result = Tensor<StorageType>(lhs.shape)
    mul(lhs, rhs: rhs, result: result)

    return result
}

public func *=<StorageType:Storage>
    (lhs:Tensor<StorageType>, rhs:Tensor<StorageType>) where StorageType.ElementType:NumericType
{
    imul(lhs, rhs)
}

public func *=<StorageType:Storage>
    (lhs:Tensor<StorageType>, rhs:StorageType.ElementType) where StorageType.ElementType:NumericType
{
    imul(lhs, rhs)
}


//
// matrix multiplication
//

public func matmul<S:Storage>
    (_ lhs:Tensor<S>, _ rhs:Tensor<S>, addTo result:Tensor<S>) where S.ElementType:NumericType
{
    precondition(lhs.shape.dims.count == 2)
    precondition(rhs.shape.dims.count == 2)
    precondition(result.shape == Extent(lhs.shape[0], rhs.shape[1]))
    
    // nxp * pxm (lhs.shape[1] == rhs.shape[0])
    for i in 0..<lhs.shape[0] {
        for j in 0..<rhs.shape[1] {
            for k in 0..<lhs.shape[1] {
                result[i, j] = result[i, j] + lhs[i, k]*rhs[k, j]
            }
        }
    }
}

public func matmul<S:Storage>
    (_ lhs:Tensor<S>, _ rhs:Tensor<S>, result:Tensor<S>) where S.ElementType:NumericType
{
    precondition(lhs.shape.dims.count == 2)
    precondition(rhs.shape.dims.count == 2)
    precondition(result.shape == Extent(lhs.shape[0], rhs.shape[1]))
    
    // nxp * pxm (lhs.shape[1] == rhs.shape[0])
    fill(result, value: 0)
    for i in 0..<lhs.shape[0] {
        for j in 0..<rhs.shape[1] {
            for k in 0..<lhs.shape[1] {
                result[i, j] += lhs[i, k]*rhs[k, j]
            }
        }
    }
}

//
// dot product
//

public func isVector(_ type:TensorType) -> Bool {
    return type == .vector || type == .rowVector || type == .columnVector
}

public func isMatrix(_ type:TensorType) -> Bool {
    return type == .matrix
}

public func isCube(_ type:TensorType) -> Bool {
    return type == .cube
}


// TODO: benchmark this against specific cases .. not sure
// the specific cases are faster (and therefore should potentially
// be removed)
public func dot<S:Storage>
    (_ lhs:Tensor<S>, _ rhs:Tensor<S>, result:Tensor<S>) where S.ElementType:FloatNumericType
{
    precondition(lhs.shape.span < 3)
    precondition(rhs.shape.span < 3)
    precondition(lhs.shape[1] == rhs.shape[0], "Number of rows in vector must match number of rows of matrix")
    precondition(lhs.shape[0] == result.shape[0], "Number of rows of result must match rows in matrix")
 
    fill(result, value: 0)
    for n in 0..<lhs.shape[0] {
        for m in 0..<rhs.shape[0] {
            for k in 0..<rhs.shape[1] {
                result[n, k] = result[n, k] + lhs[n, m]*rhs[m, k]
            }
        }
    }
}

public func dot<S:Storage>
    (_ lhs:Tensor<S>, _ rhs:Tensor<S>, addTo result:Tensor<S>) where S.ElementType:FloatNumericType
{
    precondition(lhs.shape.span < 3)
    precondition(rhs.shape.span < 3)
    precondition(lhs.shape[1] == rhs.shape[0], "Number of rows in vector must match number of rows of matrix")
    precondition(lhs.shape[0] == result.shape[0], "Number of rows of result must match rows in matrix")
    
    for n in 0..<lhs.shape[0] {
        for m in 0..<rhs.shape[0] {
            for k in 0..<rhs.shape[1] {
                result[n, k] = result[n, k] + lhs[n, m]*rhs[m, k]
            }
        }
    }
}

// TODO: make version of `dot` that returns a result vector
public func dot<S:Storage>
    (_ lhs:Tensor<S>, _ rhs:Tensor<S>) -> S.ElementType where S.ElementType:NumericType
{
    precondition(isVector(lhs.type) && isVector(rhs.type))
    precondition(lhs.shape.elements == rhs.shape.elements, "Number of elements must match")

    var result:S.ElementType = 0

    for (l, r) in zip(lhs.indices(), rhs.indices()) {
        result = result + lhs[l]*rhs[r]
    }

    return result
}

public func ⊙<S:Storage>
    (lhs:Tensor<S>, rhs:Tensor<S>) -> S.ElementType where S.ElementType:NumericType
{
    return dot(lhs, rhs)
}

public func ⊙<S:Storage>
    (lhs:Tensor<S>, rhs:Tensor<S>) -> Tensor<S> where S.ElementType:FloatNumericType
{
//    let result = Tensor<S>(shape: Extent(left.shape[0], 1))
    let result = Tensor<S>(Extent(lhs.shape[0], rhs.shape[1]))
    dot(lhs, rhs, result: result)
    return result
}

//
// outer product
//

public func outer<StorageType:Storage>
    (_ lhs:Tensor<StorageType>, _ rhs:Tensor<StorageType>, result:Tensor<StorageType>) where StorageType.ElementType:NumericType
{
    precondition(isVector(lhs.type))
    precondition(isVector(rhs.type))
    
    let indexLeft = lhs.indices()
    let indexRight = rhs.indices()
    var indexResult = result.indices()
    
    for l in indexLeft {
        for r in indexRight {
            let pos = indexResult.next()!
            result[pos] = lhs[l]*rhs[r]
        }
    }
}

public func outer<StorageType:Storage>
    (_ lhs:Tensor<StorageType>, _ rhs:Tensor<StorageType>, addTo result:Tensor<StorageType>) where StorageType.ElementType:NumericType
{
    precondition(isVector(lhs.type))
    precondition(isVector(rhs.type))
    precondition(result.shape.elements == lhs.shape.elements*rhs.shape.elements)
    
    let indexLeft = lhs.indices()
    let indexRight = rhs.indices()
    var indexResult = result.indices()
    
    for l in indexLeft {
        for r in indexRight {
            let pos = indexResult.next()!
            result[pos] = result[pos] + lhs[l]*rhs[r]
        }
    }
}

public func ⊗<StorageType:Storage>
    (lhs:Tensor<StorageType>, rhs:Tensor<StorageType>) -> Tensor<StorageType> where StorageType.ElementType:NumericType
{
    let result = Tensor<StorageType>(Extent(lhs.shape.elements, rhs.shape.elements))
    outer(lhs, rhs, result: result)
    return result
}

public func abs<StorageType:Storage>
    (_ tensor:Tensor<StorageType>) -> Tensor<StorageType> where StorageType.ElementType:NumericType
{
    let result = Tensor<StorageType>(tensor.shape)
    for index in tensor.indices() {
        result[index] = abs(tensor[index])
    }
    
    return result
}

public func isClose<StorageType:Storage>
    (_ lhs:Tensor<StorageType>, _ rhs:Tensor<StorageType>, eps: StorageType.ElementType) -> Bool where StorageType.ElementType:NumericType
{
    let diff = lhs - rhs
    let adiff:Tensor<StorageType> = abs(diff)
    for i in adiff.indices() {
        if adiff[i] >= eps { return false }
    }
    return true
}

public func isClose<StorageType:Storage>
    (_ lhs:Tensor<StorageType>, _ rhs:Array<StorageType.ElementType>, eps: StorageType.ElementType) -> Bool where StorageType.ElementType:NumericType
{
    var k = 0
    for i in lhs.indices() {
        let diff = abs(lhs[i] - rhs[k])
        if diff > eps { return false }
        
        k += 1
    }
    
    return true
}

public func hist<StorageType:Storage>
    (_ t:Tensor<StorageType>, bins:Int) -> Tensor<StorageType> where StorageType.ElementType == Double
{
    let h = Tensor<StorageType>(Extent(bins))
    let m = max(t)
    let delta = m/StorageType.ElementType(bins)
    for i in t.indices() {
        let value = t[i]
        var bin = Int(value/delta)
        if bin >= bins {  bin = bins-1 }
        h[bin] = h[bin] + 1
    }

    return h
}

func reduce<S:Storage>(
    _ tensor:Tensor<S>,
    axis:Int,
    result:Tensor<S>,
    op:(S.ElementType, S.ElementType) -> S.ElementType)
{
    precondition(axis < tensor.shape.count)
    
    let reduced:[(offset:Int, element:Int)] = tensor.shape
        .enumerated()
        .filter { return $0.offset != axis }
    
    let indices = reduced.map { $0.offset }
    
    // index `i` is the axis we are summing along
    for i in 0..<tensor.shape[axis] {
        let oldIndices = IteratorSequence(IndexGenerator(tensor.shape, dimIndex: indices))
        let newIndices = IteratorSequence(IndexGenerator(result.shape))
        
        for (var tensorIndex, resultIndex) in zip(oldIndices, newIndices) {
            tensorIndex[axis] = i
            let tensorOffset:Int = tensor.calculateOffset(tensorIndex)            
            let resultOffset:Int = result.calculateOffset(resultIndex)
            
            result.storage[resultOffset] = op(result.storage[resultOffset], tensor.storage[tensorOffset])
        }
    }
}

public func reduce<S:Storage>(_ tensor:Tensor<S>,
            axis:Int,
            op:(S.ElementType, S.ElementType) -> S.ElementType)
    -> Tensor<S>
{
    // calculate new shape
    let reduced:[(offset:Int, element:Int)] = tensor.shape
        .enumerated()
        .filter { $0.offset != axis }
    
    let newShape = reduced.map { $0.element }
    let result = Tensor<S>(Extent(newShape))

    reduce(tensor, axis: axis, result: result, op: op)
    return result
}


public func reduce<StorageType:Storage>(_ tensor:Tensor<StorageType>,
            op:(StorageType.ElementType, StorageType.ElementType) -> StorageType.ElementType)
    -> StorageType.ElementType
{
    var result:StorageType.ElementType = 0
    
    let indices = IteratorSequence(IndexGenerator(tensor.shape))
    for index in indices {
        result = op(result, tensor[index])
    }
    
    return result
}

public func sum<StorageType:Storage>
    (_ tensor:Tensor<StorageType>, axis:Int) -> Tensor<StorageType>
{
    return reduce(tensor, axis: axis, op: +)
}

public func sum<S:Storage>
    (_ tensor:Tensor<S>, axis:Int, result:Tensor<S>)
{
    reduce(tensor, axis: axis, result: result, op: +)
}

public func sum<StorageType:Storage>(_ tensor:Tensor<StorageType>) -> StorageType.ElementType {
    return reduce(tensor, op: +)
}

/*
 hO=(h+2pH−kH)/sY+1
 w0=(w+2pW−kW)/sX+1
 */
// currently only supports 'valid' mode; the kernel cannot exceed the bounds of the image
public func calculateConv2DSize<S:Storage>(input:Tensor<S>, kernel:Tensor<S>, stride:[Int], padding:[Int]) -> Extent {
    var rows = (input.shape[0] - kernel.shape[0] + 2*padding[0]) + 1
    rows /= stride[0]
    var cols = (input.shape[1] - kernel.shape[1] + 2*padding[1]) + 1
    cols /= stride[1]
    return Extent(rows, cols)
}

public func conv2d<S:Storage>(_ input:Tensor<S>,
                                kernel:Tensor<S>,
                                stride:[Int]=[1, 1],
                                padding:[Int]=[0, 0],
                                paddingValue:S.ElementType=0,
                                flip:Bool=true,
                                addTo:Tensor<S>)
{
    let centerX = kernel.shape[1] / 2
    let centerY = kernel.shape[0] / 2
    
//    var rows = (input.shape[0] - kernel.shape[0] + 2*padding[0]) + 1
//    rows /= stride[0]
//    var cols = (input.shape[1] - kernel.shape[1] + 2*padding[1]) + 1
//    cols /= stride[1]
    let outputShape = calculateConv2DSize(input: input, kernel: kernel, stride: stride, padding: padding)
    
//    let outputShape = Extent(rows, cols)
//    let out = Tensor<S>(outputShape)
    
    for i in 0..<outputShape[0] {
        for j in 0..<outputShape[1] {
            for k in 0..<kernel.shape[0] {                
                let kflipped = flip ? kernel.shape[0] - k - 1 : k
                for l in 0..<kernel.shape[1] {
                    let lflipped = flip ? kernel.shape[1] - l - 1 : l
                    
                    let m = (i + (k - centerY) - padding[0])*stride[0] + centerY
                    let n = (j + (l - centerX) - padding[1])*stride[1] + centerX
                    
                    let o = i+padding[0]/2
                    let p = j+padding[1]/2
                    
                    if (m >= 0 && m < input.shape[0] && n >= 0 && n < input.shape[1]) {
                        let input_mn:S.ElementType = input[m, n]
                        let kernel_kl:S.ElementType = kernel[kflipped, lflipped]

                        // FIXME: addTo.+= dispatches incorrectly .. this is caused by an unknown
                        // interaction of the generics and the dispatch system (needs investigation)
                        addTo[o, p] = addTo[o, p] + input_mn*kernel_kl
                    } else {
                        addTo[o, p] = addTo[o, p] + paddingValue * kernel[kflipped, lflipped]
                    }
                }
            }
        }
    }
    
//    return out
}

public func conv2d<S:Storage>(_ input:Tensor<S>,
                   kernel:Tensor<S>,
                   stride:[Int]=[1, 1],
                   padding:[Int]=[0, 0],
                   paddingValue:S.ElementType=0,
                   flip:Bool=true) -> Tensor<S> where S.ElementType:NumericType
{
    let outputShape = calculateConv2DSize(input: input, kernel: kernel, stride: stride, padding: padding)
    let out = Tensor<S>(outputShape)
    
    conv2d(input, kernel:kernel, stride:stride, padding:padding, paddingValue:paddingValue, flip:flip, addTo:out)
    return out
}


public func max<StorageType:Storage>
    (_ tensor:Tensor<StorageType>, axis:Int) -> Tensor<StorageType> where StorageType.ElementType:NumericType
{
    return reduce(tensor, axis: axis, op: max)
}

public func max<StorageType:Storage>
    (_ tensor:Tensor<StorageType>) -> StorageType.ElementType where StorageType.ElementType:NumericType
{
    return reduce(tensor, op: max)
}

public func min<StorageType:Storage>
    (_ tensor:Tensor<StorageType>, axis:Int) -> Tensor<StorageType> where StorageType.ElementType:NumericType
{
    return reduce(tensor, axis: axis, op: min)
}

public func min<StorageType:Storage>
    (_ tensor:Tensor<StorageType>) -> StorageType.ElementType where StorageType.ElementType:NumericType
{
    return reduce(tensor, op: min)
}

public func **<StorageType:Storage>
    (tensor:Tensor<StorageType>, power:StorageType.ElementType) -> Tensor<StorageType> where StorageType.ElementType:FloatNumericType
{
    return pow(tensor, power)
}

public func pow<StorageType:Storage>
    (_ tensor:Tensor<StorageType>, _ power:StorageType.ElementType) -> Tensor<StorageType> where StorageType.ElementType:FloatNumericType
{
    let result = Tensor<StorageType>(tensor.shape)
    pow(tensor, power, result: result)
    return result
}

public func pow<StorageType:Storage>
    (_ tensor:Tensor<StorageType>, _ power:StorageType.ElementType, result:Tensor<StorageType>) where StorageType.ElementType:FloatNumericType
{
    for i in result.indices() {
        result[i] = StorageType.ElementType.pow(tensor[i], power)
    }
}

public func exp<StorageType:Storage>
    (_ tensor:Tensor<StorageType>) -> Tensor<StorageType> where StorageType.ElementType:FloatNumericType
{
    let result = Tensor<StorageType>(tensor.shape)
    for i in result.indices() {
        result[i] = StorageType.ElementType.exp(tensor[i])
    }
    
    return result
}

public func log<StorageType:Storage>
    (_ tensor:Tensor<StorageType>) -> Tensor<StorageType> where StorageType.ElementType:FloatNumericType
{
    let result = Tensor<StorageType>(tensor.shape)
    for i in result.indices() {
        result[i] = StorageType.ElementType.log(tensor[i])
    }
    
    return result
}

func sqrt(_ tensor:TensorType) -> TensorType
{
    preconditionFailure()
}

public func sqrt<StorageType:Storage>
    (_ tensor:Tensor<StorageType>) -> Tensor<StorageType> where StorageType.ElementType:FloatNumericType
{
    let indices = tensor.indices()
    let result = Tensor<StorageType>(tensor.shape)
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

public func norm<StorageType:Storage>
    (_ tensor:Tensor<StorageType>, axis:Int) -> Tensor<StorageType> where StorageType.ElementType:FloatNumericType
{
    let p = pow(tensor, StorageType.ElementType(2.0))
    let s = sum(p, axis: axis)
    return sqrt(s)
}

public func sigmoid<S:Storage>
    (_ input:Tensor<S>, output:Tensor<S>) where S.ElementType:FloatNumericType
{
    precondition(input.shape == output.shape)
    for index in input.indices() {
        output[index] = S.ElementType(1.0) / (S.ElementType(1.0) + S.ElementType.exp(-input[index]))
    }
}

public func sigmoid<S:Storage>
    (_ input:Tensor<S>) -> Tensor<S> where S.ElementType:FloatNumericType
{
//    precondition(input.shape == output.shape)
    let output:Tensor<S> = zeros(input.shape)
    for index in input.indices() {
        output[index] = S.ElementType(1.0) / (S.ElementType(1.0) + S.ElementType.exp(-input[index]))
    }
    
    return output
}

public func tanh<S:Storage>
    (_ input:Tensor<S>, output:Tensor<S>) where S.ElementType:FloatNumericType
{
    precondition(input.shape == output.shape)
    for index in input.indices() {
        output[index] = S.ElementType.tanh(input[index])
    }
}


public func log<S:Storage>
    (_ input:Tensor<S>, result:Tensor<S>) where S.ElementType:FloatNumericType
{
    for (i1, i2) in zip(input.indices(), result.indices()) {
        result[i2] = S.ElementType.log(input[i1])
    }
}

// TODO: should employ inputMax trick to make sure no 0s occur
public func logsoftmax<S:Storage>
    (_ input:Tensor<S>, result:Tensor<S>) where S.ElementType:FloatNumericType
{
    // log[ exp(input) / sum(exp(input)) ]
    // = log(exp(input)) - log(sum(exp(input)))
    // = input - log(sum(exp(input)))
    let s = Tensor<S>([sum(exp(input))])
    let logsum = log(s)
    sub(input, logsum, result: result)
}

