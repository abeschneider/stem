//
//  checkgradient.swift
//  stem
//
//  Created by Abe Schneider on 12/8/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation

func calcForwardGrad<S:Storage>
    (_ op:Op<S>, params:Tensor<S>, eps:Double) -> Tensor<S> where S.ElementType:FloatNumericType
{
    let jacobian:Tensor<S> = zeros(Extent(params.shape.elements, op.output.shape.elements))

    for i in 0..<params.shape.elements {
        let old_value:S.ElementType = params[i]
        
        // positive direction
        params[i] = old_value + S.ElementType(eps)
        op.reset()
        op.apply()
        let pvalue = copy(ravel(op.output))
        
        // negative direction
        params[i] = old_value - S.ElementType(eps)
        op.reset()
        op.apply()
        let nvalue = copy(ravel(op.output))
        
        // return to original value
        params[i] = old_value
        
        let diff:Tensor<S> = (pvalue - nvalue) * S.ElementType(1.0/(2.0*eps))
        jacobian[i, all] = ravel(diff)
    }
    
    return jacobian
}

func calcBackwardGrad<S:Storage>
    (_ forward:Op<S>, _ backward:Op<S>, gradParams:Tensor<S>) -> Tensor<S> where S.ElementType:FloatNumericType
{
    forward.reset()
    forward.apply()
    backward.apply()
    
    let gradOutput = ravel(backward.inputs["gradOutput"]!.output())
    let jacobian:Tensor<S> = zeros(Extent(gradParams.shape.elements, gradOutput.shape.elements))

    for i in 0..<gradOutput.shape.elements {
        fill(gradOutput, value: 0)
        gradOutput[i] = 1
        
        (backward as! GradientType).reset()
        backward.apply()
        var din = gradParams
        din = din.reshape(Extent(din.shape[0], 1))
        jacobian[all, i] = din
    }
    
    return jacobian
}

public func checkGradient<S:Storage>
    (_ op:Op<S>, grad:Op<S>, params:Tensor<S>, gradParams:Tensor<S>, eps:Double) -> S.ElementType where S.ElementType:FloatNumericType
{
    op.reset()
    op.apply()
    grad.apply()
    
    let fgrad = calcForwardGrad(op, params: ravel(params), eps: eps)
    let bgrad = calcBackwardGrad(op, grad, gradParams: ravel(gradParams))
    
    let error = Tensor<S>(zeros(fgrad.shape))
    sub(fgrad, bgrad, result: error)
    return max(abs(error))
}

// TODO: can we get rid of this version, or are both versions required?
public func checkGradient<S:Storage, OpT:Op<S>>
    (_ op:OpT, grad:Op<S>, input:Tensor<S>, eps:Double) -> S.ElementType where S.ElementType:FloatNumericType, OpT:Loss, OpT.StorageType.ElementType==S.ElementType
{
    op.apply()
    grad.apply()
    
    //    let result = Tensor<S>(Extent(gradParams.map { $0.shape.elements }.reduce(0, combine: +)))
    let result:Tensor<S> = zeros(grad.output.shape)

    // copy gradients, they will be overwritten from subsequent calls to `fn`
    let analytical_diff = copy(grad.output)
    
    for i in 0..<input.shape.elements {
        let old_value:S.ElementType = input[i]
        
        // positive direction
        input[i] = old_value + S.ElementType(eps)
        op.apply()

        let pvalue:OpT.StorageType.ElementType = op.value
        
        // negative direction
        input[i] = old_value - S.ElementType(eps)
        op.apply()
        let nvalue:OpT.StorageType.ElementType = op.value
        
        
        // return to original value
        input[i] = old_value
        
        let numerical_diff = (pvalue - nvalue) / S.ElementType(2.0*eps)
        
        // TODO: Look into alternate formulations (e.g. either norm of both, or max of denom.)
        result[i] = abs((numerical_diff - analytical_diff[i])/analytical_diff[i])
    }
    
    return max(abs(result))
}
