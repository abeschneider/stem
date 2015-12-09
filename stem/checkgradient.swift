//
//  checkgradient.swift
//  stem
//
//  Created by Abe Schneider on 12/8/15.
//  Copyright © 2015 none. All rights reserved.
//

import Foundation

//std::list<real_t> check_gradient(std::function<real_t (const vector_t &)> fn,
//GradientModule &mod,
//const vector_t &input,
//real_t eps)

/*
func check_gradient<StorageType:Storage, ModuleType where ModuleType:Module, ModuleType:GradientModule>(
    fn:(Vector<StorageType>) -> StorageType.ElementType,
    module:ModuleType,
    var flattenedParams:StorageType,
    input:Vector<StorageType>,
    eps:StorageType.ElementType)
{
    // calculate gradients
    fn(input)
    
    // copy gradients from the last operation -- they will
    // be overwritten from subsequent calls
    
    
    
    for i in 0..<flattenedParams.shape[0] {
        let old_value = flattenedParams[i]
        flattenedParams[i] = flattenedParams[i] + eps
        let pvalue = fn(input)
        
        flattenedParams[i] = old_value-eps
        let nvalue = fn(input)
        flattenedParams[i] = old_value
        
        let numerical_diff = (pvalue - nvalue) / (2*eps)
//        let analytical_diff = grad_pos++
        // let diff = abs(numerical_diff - analytical_diff)
        // result.append(diff)
    }
}*/

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