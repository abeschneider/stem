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

func elementwiseBinaryOp<S:Storage>
    (left:Tensor<S>, _ right:Tensor<S>, result:Tensor<S>,
    op:(left:S.ElementType, right:S.ElementType) -> S.ElementType)
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

func elementwiseBinaryOp<S:Storage>
    (left:Tensor<S>, _ right:S.ElementType, result:Tensor<S>,
    op:(left:S.ElementType, right:S.ElementType) -> S.ElementType)
{
    precondition(left.shape.elements == result.shape.elements)
    
    var indexResult = result.indices()
    for i in left.indices() {
        let idx = indexResult.next()!
        result[idx] = op(left: left[i], right: right)
    }
}

func elementwiseBinaryOp<StorageType:Storage>
    (left:StorageType.ElementType, _ right:Tensor<StorageType>, result:Tensor<StorageType>,
    op:(left:StorageType.ElementType, right:StorageType.ElementType) -> StorageType.ElementType)
{
    precondition(right.shape.elements == result.shape.elements)
    
    var indexResult = result.indices()
    for i in right.indices() {
        let idx = indexResult.next()!
        result[idx] = op(left: left, right: right[i])
    }
}

//
// addition
//

/**
    Adds `lhs` to `rhs` and puts resulting value in `result`. If their shapes don't match, calls `broadcast`.
 
    - Parameter lhs: Tensor
    - Parameter rhs: Tensor
    - Parameter result: Tensor to place results of addition
 */
public func add<S:Storage where S.ElementType:NumericType>
    (lhs:Tensor<S>, _ rhs:Tensor<S>, result:Tensor<S>)
{
    if lhs.shape == rhs.shape {
        elementwiseBinaryOp(lhs, rhs, result: result, op: { $0 + $1 })
    } else {
        let (l, r) = broadcast(lhs, rhs)
        elementwiseBinaryOp(l, r, result: result, op: { $0 + $1 })
    }
}

/**
 Adds `lhs` to `rhs` and puts resulting value in `result`. If their shapes don't match, calls `broadcast`.
 
 - Parameter lhs: Tensor
 - Parameter rhs: Scalar
 - Parameter result: Tensor to place results of addition
 */
public func add<S:Storage where S.ElementType:NumericType>
    (lhs:Tensor<S>, _ rhs:S.ElementType, result:Tensor<S>)
{
    elementwiseBinaryOp(lhs, rhs, result: result, op: { $0 + $1 })
}

/**
 Adds `lhs` to `rhs` and puts resulting value in `result`. If their shapes don't match, calls `broadcast`.
 
 - Parameter lhs: Scalar
 - Parameter rhs: Tensor
 - Parameter result: Tensor to place results of addition
 */
public func add<S:Storage where S.ElementType:NumericType>
    (lhs:S.ElementType, _ rhs:Tensor<S>, result:Tensor<S>)
{
    elementwiseBinaryOp(lhs, rhs, result: result, op: { $0 + $1 })
}


/**
 Adds `rhs` to `lhs` in-place. If their shapes don't match, calls `broadcast`.
 
 - Parameter lhs: Tensor
 - Parameter rhs: Scalar
 */
public func iadd<S:Storage where S.ElementType:NumericType>
    (lhs:Tensor<S>, _ rhs:Tensor<S>)
{
    if lhs.shape == rhs.shape {
        elementwiseBinaryOp(lhs, rhs, result: lhs, op: { $0 + $1 })
    } else {
        let (l, r) = broadcast(lhs, rhs)
        elementwiseBinaryOp(l, r, result: lhs, op: { $0 + $1 })
    }
}

/**
 Adds `rhs` to `lhs` in-place. If their shapes don't match, calls `broadcast`.
 
 - Parameter lhs: Tensor
 - Parameter rhs: Scalar
 */
public func iadd<S:Storage where S.ElementType:NumericType>
    (lhs:Tensor<S>, _ rhs:S.ElementType)
{
    elementwiseBinaryOp(lhs, rhs, result: lhs, op: { $0 + $1 })
}

/**
 Returns the addition of `lhs` to `rhs`. If their shapes don't match, calls broadcast.
 
 - Parameter lhs: Tensor
 - Parameter rhs: Tensor
 - Returns: Results of `lhs` + `rhs`
 */
public func +<S:Storage where S.ElementType:NumericType>
    (lhs:Tensor<S>, rhs:Tensor<S>) -> Tensor<S>
{
    let result = Tensor<S>(lhs.shape)
    add(lhs, rhs, result: result)
    
    return result
}

/**
 Returns the addition of `lhs` to `rhs`. If their shapes don't match, calls broadcast.
 
 - Parameter lhs: Tensor
 - Parameter rhs: Scalar
 - Returns: Results of `lhs` + `rhs`
 */
public func +<S:Storage where S.ElementType:NumericType>
    (lhs:Tensor<S>, rhs:S.ElementType) -> Tensor<S>
{
    let result = Tensor<S>(lhs.shape)
    add(lhs, rhs, result: result)
    
    return result
}

/**
 Returns the addition of `lhs` to `rhs`. If their shapes don't match, calls broadcast.
 
 - Parameter lhs: Tensor
 - Parameter rhs: Scalar
 - Returns: Results of `lhs` + `rhs`
 */
public func +<S:Storage where S.ElementType:NumericType>
    (lhs:S.ElementType, rhs:Tensor<S>) -> Tensor<S>
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
public func +=<S:Storage where S.ElementType:NumericType>
    (lhs:Tensor<S>, rhs:Tensor<S>)
{
    iadd(lhs, rhs)
}

/**
 Adds `rhs` to `lhs` in-place. If their shapes don't match, calls `broadcast`.
 
 - Parameter lhs: Tensor
 - Parameter rhs: Tensor
 */
public func +=<S:Storage where S.ElementType:NumericType>
    (lhs:Tensor<S>, rhs:S.ElementType)
{
    iadd(lhs, rhs)
}

//
// subtraction
//

public func sub<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:Tensor<StorageType>, _ rhs:Tensor<StorageType>, result:Tensor<StorageType>)
{
    if lhs.shape == rhs.shape {
        elementwiseBinaryOp(lhs, rhs, result: result, op: { return $0 - $1 })
    } else {
        let (l, r) = broadcast(lhs, rhs)
        elementwiseBinaryOp(l, r, result: result, op: { return $0 - $1 })
    }
    
}

public func sub<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:StorageType.ElementType, _ rhs:Tensor<StorageType>, result:Tensor<StorageType>)
{
    elementwiseBinaryOp(lhs, rhs, result: result, op: { return $0 - $1 })
}

public func isub<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:Tensor<StorageType>, _ rhs:Tensor<StorageType>)
{
    if lhs.shape == rhs.shape {
        elementwiseBinaryOp(lhs, rhs, result: lhs, op: { $0 - $1 })
    } else {
        let (l, r) = broadcast(lhs, rhs)
        elementwiseBinaryOp(l, r, result: lhs, op: { $0 - $1 })
    }
}

public func -<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:Tensor<StorageType>, rhs:Tensor<StorageType>) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(lhs.shape)
    sub(lhs, rhs, result: result)
    
    return result
}

public func -<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:StorageType.ElementType, rhs:Tensor<StorageType>) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(rhs.shape)
    sub(lhs, rhs, result: result)
    
    return result
}


public func -=<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:Tensor<StorageType>, rhs:Tensor<StorageType>)
{
    isub(lhs, rhs)
}


//
// elementwise division
//

public func div<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:Tensor<StorageType>, rhs:Tensor<StorageType>, result:Tensor<StorageType>)
{
    elementwiseBinaryOp(lhs, rhs, result: result, op: { return $0 / $1 })
}


public func div<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:Tensor<StorageType>, rhs:StorageType.ElementType, result:Tensor<StorageType>)
{
    elementwiseBinaryOp(lhs, rhs, result: result, op: { return $0 / $1 })
}


public func idiv<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:Tensor<StorageType>, _ rhs:Tensor<StorageType>)
{
    if lhs.shape == rhs.shape {
        elementwiseBinaryOp(lhs, rhs, result: lhs, op: { $0 / $1 })
    } else {
        let (l, r) = broadcast(lhs, rhs)
        elementwiseBinaryOp(l, r, result: lhs, op: { $0 / $1 })
    }
}

public func idiv<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:Tensor<StorageType>, _ rhs:StorageType.ElementType)
{
    elementwiseBinaryOp(lhs, rhs, result: lhs, op: { $0 / $1 })
}

public func /<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:Tensor<StorageType>, rhs:Tensor<StorageType>) -> Tensor<StorageType>
{
    let shape = max(lhs.shape, rhs.shape)
    let result = Tensor<StorageType>(shape)
    div(lhs, rhs: rhs, result: result)
    return result
}

public func /=<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:Tensor<StorageType>, rhs:Tensor<StorageType>)
{
    idiv(lhs, rhs)
}

public func /=<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:Tensor<StorageType>, rhs:StorageType.ElementType)
{
    idiv(lhs, rhs)
}

//
// element-wise multiplication
//

public func mul<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:Tensor<StorageType>, _ rhs:Tensor<StorageType>, result:Tensor<StorageType>)
{
    if lhs.shape == rhs.shape {
        elementwiseBinaryOp(lhs, rhs, result: result, op: { return $0 * $1 })
    } else {
        let (l, r) = broadcast(lhs, rhs)
        elementwiseBinaryOp(l, r, result: result, op: { return $0 * $1 })
    }
}

public func mul<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:Tensor<StorageType>, rhs:StorageType.ElementType, result:Tensor<StorageType>)
{
    elementwiseBinaryOp(lhs, rhs, result: result, op: { return $0 * $1 })
}

public func mul<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:StorageType.ElementType, _ rhs:Tensor<StorageType>, result:Tensor<StorageType>)
{
    elementwiseBinaryOp(lhs, rhs, result: result, op: { return $0 * $1 })
}

public func mul<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:Tensor<StorageType>, _ rhs:StorageType.ElementType, result:Tensor<StorageType>)
{
    elementwiseBinaryOp(lhs, rhs, result: result, op: { return $0 * $1 })
}

public func imul<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:Tensor<StorageType>, _ rhs:Tensor<StorageType>)
{
    if lhs.shape == rhs.shape {
        elementwiseBinaryOp(lhs, rhs, result: lhs, op: { $0 * $1 })
    } else {
        let (l, r) = broadcast(lhs, rhs)
        elementwiseBinaryOp(l, r, result: lhs, op: { $0 * $1 })
    }
}

public func imul<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:Tensor<StorageType>, _ rhs:StorageType.ElementType)
{
    elementwiseBinaryOp(lhs, rhs, result: lhs, op: { $0 * $1 })
}

public func *<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:Tensor<StorageType>, rhs:Tensor<StorageType>) -> Tensor<StorageType>
{
    let shape = max(lhs.shape, rhs.shape)
    let result = Tensor<StorageType>(shape)
    
    mul(lhs, rhs, result: result)
    
    return result
}

public func *<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:StorageType.ElementType, rhs:Tensor<StorageType>) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(rhs.shape)
    mul(lhs, rhs, result: result)

    return result
}

public func *<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:Tensor<StorageType>, rhs:StorageType.ElementType) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(lhs.shape)
    mul(lhs, rhs: rhs, result: result)

    return result
}

public func *=<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:Tensor<StorageType>, rhs:Tensor<StorageType>)
{
    imul(lhs, rhs)
}

public func *=<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:Tensor<StorageType>, rhs:StorageType.ElementType)
{
    imul(lhs, rhs)
}


//
// matrix multiplication
//

func matmul<S:Storage where S.ElementType:NumericType>
    (lhs:Tensor<S>, _ rhs:Tensor<S>, addTo result:Tensor<S>)
{
    precondition(lhs.shape.dims.count == 2)
    precondition(rhs.shape.dims.count == 2)
    precondition(result.shape == Extent(lhs.shape[0], rhs.shape[1]))
    
    // nxp * pxm (lhs.shape[1] == rhs.shape[0])
    for i in 0..<lhs.shape[0] {
        for j in 0..<rhs.shape[1] {
            for k in 0..<lhs.shape[1] {
                result[i, j] += lhs[i, k]*rhs[k, j]
            }
        }
    }
}

func matmul<S:Storage where S.ElementType:NumericType>
    (lhs:Tensor<S>, _ rhs:Tensor<S>, result:Tensor<S>)
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
public func dot<S:Storage where S.ElementType:FloatNumericType>
    (lhs:Tensor<S>, _ rhs:Tensor<S>, result:Tensor<S>)
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

public func dot<S:Storage where S.ElementType:FloatNumericType>
    (lhs:Tensor<S>, _ rhs:Tensor<S>, addTo result:Tensor<S>)
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
public func dot<S:Storage where S.ElementType:NumericType>
    (lhs:Tensor<S>, _ rhs:Tensor<S>) -> S.ElementType
{
    precondition(isVector(lhs.type) && isVector(rhs.type))
    precondition(lhs.shape.elements == rhs.shape.elements, "Number of elements must match")

    var result:S.ElementType = 0

    for (l, r) in Zip2Sequence(lhs.indices(), rhs.indices()) {
        result = result + lhs[l]*rhs[r]
    }

    return result
}

public func ⊙<S:Storage where S.ElementType:NumericType>
    (lhs:Tensor<S>, rhs:Tensor<S>) -> S.ElementType
{
    return dot(lhs, rhs)
}

public func ⊙<S:Storage where S.ElementType:FloatNumericType>
    (lhs:Tensor<S>, rhs:Tensor<S>) -> Tensor<S>
{
//    let result = Tensor<S>(shape: Extent(left.shape[0], 1))
    let result = Tensor<S>(Extent(lhs.shape[0], rhs.shape[1]))
    dot(lhs, rhs, result: result)
    return result
}

//
// outer product
//

public func outer<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:Tensor<StorageType>, _ rhs:Tensor<StorageType>, result:Tensor<StorageType>)
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

public func outer<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:Tensor<StorageType>, _ rhs:Tensor<StorageType>, addTo result:Tensor<StorageType>)
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

public func ⊗<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:Tensor<StorageType>, rhs:Tensor<StorageType>) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(Extent(lhs.shape.elements, rhs.shape.elements))
    outer(lhs, rhs, result: result)
    return result
}

public func abs<StorageType:Storage where StorageType.ElementType:NumericType>
    (tensor:Tensor<StorageType>) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(tensor.shape)
    for index in tensor.indices() {
        result[index] = abs(tensor[index])
    }
    
    return result
}

public func isClose<StorageType:Storage where StorageType.ElementType:NumericType>
    (lhs:Tensor<StorageType>, _ rhs:Tensor<StorageType>, eps: StorageType.ElementType) -> Bool
{
    let diff = lhs - rhs
    let adiff:Tensor<StorageType> = abs(diff)
    for i in adiff.indices() {
        if adiff[i] >= eps { return false }
    }
    return true
}


public func hist<StorageType:Storage where StorageType.ElementType == Double>
    (t:Tensor<StorageType>, bins:Int) -> Tensor<StorageType>
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
    tensor:Tensor<S>,
    axis:Int,
    result:Tensor<S>,
    op:(S.ElementType, S.ElementType) -> S.ElementType)
{
    precondition(axis < tensor.shape.count)
    
    let reduced:[(index:Int, element:Int)] = tensor.shape
        .enumerate()
        .filter { $0.index != axis }
    
    let indices = reduced.map { $0.index }
    
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
    
//    return result
}

func reduce<S:Storage>(tensor:Tensor<S>,
            axis:Int,
            op:(S.ElementType, S.ElementType) -> S.ElementType)
    -> Tensor<S>
{
    // calculate new shape
    let reduced:[(index:Int, element:Int)] = tensor.shape
        .enumerate()
        .filter { $0.index != axis }
    
    let newShape = reduced.map { $0.element }
    let result = Tensor<S>(Extent(newShape))

    reduce(tensor, axis: axis, result: result, op: op)
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

public func sum<S:Storage>
    (tensor:Tensor<S>, axis:Int, result:Tensor<S>)
{
    reduce(tensor, axis: axis, result: result, op: +)
}

public func sum<StorageType:Storage>(tensor:Tensor<StorageType>) -> StorageType.ElementType {
    return reduce(tensor, op: +)
}

/*
 hO=(h+2pH−kH)/sY+1
 w0=(w+2pW−kW)/sX+1
 */
public func conv2d<S:Storage>(input:Tensor<S>, kernel:Tensor<S>, stride:[Int]=[1, 1], padding:[Int]=[0, 0], paddingValue:S.ElementType=0) -> Tensor<S> {
    let centerX = kernel.shape[1] / 2
    let centerY = kernel.shape[0] / 2
    
    let rows = (input.shape[0]+2*padding[0]-kernel.shape[0])/stride[0]
    let cols = (input.shape[1]+2*padding[1]-kernel.shape[1])/stride[1]
    let outputShape = Extent(rows, cols)
    
    let out = Tensor<S>(outputShape)
    
    print("outputShape = \(outputShape)")
    
    for i in 0..<(outputShape[0]) {
        for j in 0..<(outputShape[1]) {
            for k in 0..<kernel.shape[0] {
                
                let kflipped = kernel.shape[0] - k - 1
                for l in 0..<kernel.shape[1] {
                    let lflipped = kernel.shape[1] - l - 1
                    
                    let m = (i + (k - centerY) - padding[0])*stride[0] + centerY
                    let n = (j + (l - centerX) - padding[1])*stride[1] + centerX
                    
                    let o = i+padding[0]/2
                    let p = j+padding[1]/2
                    
                    if (m >= 0 && m < input.shape[0] && n >= 0 && n < input.shape[1]) {
                        out[o, p] += input[m, n] * kernel[kflipped, lflipped]
                    } else {
                        out[o, p] += paddingValue * kernel[kflipped, lflipped]
                    }
                }
            }
        }
    }
    
    return out
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
    let result = Tensor<StorageType>(tensor.shape)
    pow(tensor, power, result: result)
    return result
}

public func pow<StorageType:Storage where StorageType.ElementType:FloatNumericType>
    (tensor:Tensor<StorageType>, _ power:StorageType.ElementType, result:Tensor<StorageType>)
{
    for i in result.indices() {
        result[i] = StorageType.ElementType.pow(tensor[i], power)
    }
}

func exp<StorageType:Storage where StorageType.ElementType:FloatNumericType>
    (tensor:Tensor<StorageType>) -> Tensor<StorageType>
{
    let result = Tensor<StorageType>(tensor.shape)
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

func norm<StorageType:Storage where StorageType.ElementType:FloatNumericType>
    (tensor:Tensor<StorageType>, axis:Int) -> Tensor<StorageType>
{
    let p = pow(tensor, StorageType.ElementType(2.0))
    let s = sum(p, axis: axis)
    return sqrt(s)
}

func sigmoid<S:Storage where S.ElementType:FloatNumericType>
    (input:Tensor<S>, output:Tensor<S>)
{
    precondition(input.shape == output.shape)
    for index in input.indices() {
        output[index] = S.ElementType(1.0) / (S.ElementType(1.0) + S.ElementType.exp(-input[index]))
//        print("s: \(input[index]), \(output[index])")
    }
}

func sigmoid<S:Storage where S.ElementType:FloatNumericType>
    (input:Tensor<S>) -> Tensor<S>
{
//    precondition(input.shape == output.shape)
    let output:Tensor<S> = zeros(input.shape)
    for index in input.indices() {
        output[index] = S.ElementType(1.0) / (S.ElementType(1.0) + S.ElementType.exp(-input[index]))
    }
    
    return output
}

func log<S:Storage where S.ElementType:FloatNumericType>
    (input:Tensor<S>, result:Tensor<S>)
{
    for (i1, i2) in Zip2Sequence(input.indices(), result.indices()) {
        result[i2] = S.ElementType.log(input[i1])
    }
}


