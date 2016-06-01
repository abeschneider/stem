//
//  checkgradient.swift
//  stem
//
//  Created by Abe Schneider on 12/8/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation

public func checkGradient<S:Storage where S.ElementType:FloatNumericType>
    (params params:Tensor<S>,
            gradParams:Tensor<S>,
            eps:Double,
            fn:() -> S.ElementType) -> Tensor<S>
{
//    let result = Tensor<S>(Extent(gradParams.map { $0.shape.elements }.reduce(0, combine: +)))
    let result:Tensor<S> = zeros(Extent(gradParams.shape))
    
    // calculate gradients for `analytical_diff`
    fn()
    
    // copy gradients, they will be overwritten from subsequent calls to `fn`
    let analytical_diff = copy(gradParams)
    
    for i in 0..<params.shape[0] {
        let old_value:S.ElementType = params[i]
        
        // positive direction
        params[i] = old_value + S.ElementType(eps)
        let pvalue = fn()
        
        // negative direction
        params[i] = old_value - S.ElementType(eps)
        let nvalue = fn()
        
        // return to original value
        params[i] = old_value
        
        let numerical_diff = (pvalue - nvalue) / S.ElementType(2.0*eps)
        
        // TODO: Look into alternate formulations (e.g. either norm of both, or max of denom.)
        result[i] = abs((numerical_diff - analytical_diff[i])/analytical_diff[i])
        let adiff:S.ElementType = analytical_diff[i]
    }
    
    return result
}

public func checkGradient<S:Storage where S.ElementType:FloatNumericType>
    (params params:[Tensor<S>],
     gradParams:[Tensor<S>],
     eps:Double,
     fn:() -> S.ElementType) -> Tensor<S>
{
    let result = Tensor<S>(Extent(gradParams.map { $0.shape.elements }.reduce(0, combine: +)))
    
    // calculate gradients for `analytical_diff`
    fn()
    
    // copy gradients, they will be overwritten from subsequent calls to `fn`
//    let analytical_diff = copy(gradParams)
    print(gradParams.map { copy($0) })
    let analytical_diff = concat(gradParams.map { copy($0).reshape(Extent($0.shape.elements)) }, axis: 0)
    print("adiff = \(analytical_diff)")
    
    var c = 0
    
    for param in params {
        let p = param.reshape(Extent(param.shape.elements))
//        print("p = \(p)")
        for i in 0..<p.shape[0] {
            let old_value:S.ElementType = p[i]
            
            // positive direction
            p[i] = old_value + S.ElementType(eps)
            let pvalue = fn()
            
            // negative direction
            p[i] = old_value - S.ElementType(eps)
            let nvalue = fn()
            
            // return to original value
            p[i] = old_value
            
            let numerical_diff = (pvalue - nvalue) / S.ElementType(2.0*eps)
            
            // TODO: Look into alternate formulations (e.g. either norm of both, or max of denom.)
            result[c] = abs((numerical_diff - analytical_diff[c])/analytical_diff[c])
            c += 1
        }
    }
    
    return result
}
