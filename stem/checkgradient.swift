//
//  checkgradient.swift
//  stem
//
//  Created by Abe Schneider on 12/8/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation

func checkGradient<StorageType:Storage where StorageType.ElementType == Double>(
    input:ColumnVector<StorageType>,
    var params:StorageType,
    gradParams:StorageType,
    eps:Double,
    _ fn:(ColumnVector<StorageType>) -> StorageType.ElementType) -> Vector<StorageType>
{
    let result = Vector<StorageType>(rows: gradParams.size)
    
    // make a copy of the paramters
//    var params = StorageType(storage: flattenedGradParams)
    
    // calculate gradients
    fn(input)
    
    // copy gradients from the last operation -- they will
    // be overwritten from subsequent calls to `fn`
    let analytical_diff = Vector<StorageType>(storage: gradParams, shape: Extent(gradParams.size))
    
//    for i in 0..<params.size { params[i] = 0 }
    
//    let numerical_diff = Vector<StorageType>(rows: gradParams.size)
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
        
//        numerical_diff[i] = (pvalue - nvalue) / StorageType.ElementType(2.0*eps)
        let numerical_diff = (pvalue - nvalue) / StorageType.ElementType(2.0*eps)
        result[i] = (numerical_diff - analytical_diff[i])/analytical_diff[i]
    }
    
//    let num = analytical_diff - numerical_diff
//    let result = norm(num) / norm(analytical_diff + numerical_diff)
    return result
}

//func check_gradient<StorageType:Storage,
//                    StorageType2:Storage, // FIXME: why is this needed? if this isn't here compiler complains about 'input'
//                    ModuleType where ModuleType:Module,
//                    ModuleType:GradientModule,
//                    StorageType.ElementType:NumericType>(
//    module module:ModuleType,
//    params flattenedParams:StorageType,
//    gradParams flattenedGradParams:StorageType,
//    input val:Vector<StorageType2>,
//    precision eps:Double,
//    _ fn:(Vector<StorageType2>) -> StorageType.ElementType) -> Vector<StorageType>
//{
//    let result = Vector<StorageType>(rows: flattenedGradParams.size)
//    
//    // make a copy of the paramters
//    var params = StorageType(storage: flattenedGradParams)
//    
//    // calculate gradients
//    fn(val)
//    
//    // copy gradients from the last operation -- they will
//    // be overwritten from subsequent calls
//    let analytical_diff = Vector<StorageType>(storage: flattenedGradParams, shape: Extent(flattenedGradParams.size))
//    
//    for i in 0..<flattenedParams.size {
//        let old_value = flattenedParams[i]
//        params[i] = flattenedParams[i] + StorageType.ElementType(eps)
//        let pvalue = fn(val)
//        
//        params[i] = old_value - StorageType.ElementType(eps)
//        let nvalue = fn(val)
//        params[i] = old_value
//        
//        let numerical_diff = (pvalue - nvalue) / StorageType.ElementType(2.0*eps)
//        result[i] = abs(numerical_diff - analytical_diff[i])
//    }
//    
//    return result
//}

/*
func grad_forward<StorageType:Storage, ModuleType where ModuleType:Module, ModuleType:GradientModule>(
    module:ModuleType,
    input:Vector<StorageType>,
    param:Vector<StorageType>,
    perturbation:StorageType.ElementType) -> Matrix<StorageType>
{
    let output = module.forward(input)
    let jacobian = Matrix<StorageType>(rows: param.shape.elements, cols: output.shape.elements)
    for i in 0..<param.shape.elements {
        let old_value = param[i]
        param[i] = old_value - perturbation
        let nvalue = module.forward(input)
        param[i] = old_value + perturbation
        let pvalue = module.forward(input)
        let diff = (pvalue - nvalue)/(2*perturbation)
        jacobian[i, 0..<param.shape.elements] = diff
    }
    
    return jacobian
}*/