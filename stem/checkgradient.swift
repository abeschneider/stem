//
//  checkgradient.swift
//  stem
//
//  Created by Abe Schneider on 12/8/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation

func checkGradient<StorageType:Storage where StorageType.ElementType:FloatNumericType>( //) where StorageType.ElementType == Double>(
    input:Tensor<StorageType>,
    var params:StorageType,
    gradParams:StorageType,
    eps:Double,
    _ fn:(Tensor<StorageType>) -> StorageType.ElementType) -> Tensor<StorageType>
{
//    let result = Vector<StorageType>(rows: gradParams.size)
    let result = Tensor<StorageType>(Extent(gradParams.size))
    
    // calculate gradients
    fn(input)
    
    // copy gradients from the last operation -- they will
    // be overwritten from subsequent calls to `fn`
//    let analytical_diff = Vector<StorageType>(storage: gradParams, shape: Extent(gradParams.size))
    let analytical_diff = Tensor<StorageType>(storage: gradParams, shape: Extent(gradParams.size))
    
    for i in 0..<params.size {
        let old_value = params[i]
        
        // positive direction
        params[i] = old_value + StorageType.ElementType(eps)
        let pvalue = fn(input)
        
        // negative direction
        params[i] = old_value - StorageType.ElementType(eps)
        let nvalue = fn(input)
        
        // return to original value
        params[i] = old_value
        
        let numerical_diff = (pvalue - nvalue) / StorageType.ElementType(2.0*eps)
        result[i] = (numerical_diff - analytical_diff[i])/analytical_diff[i]
    }
    
    return result
}
