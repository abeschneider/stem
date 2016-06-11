//
//  optimize.swift
//  stem
//
//  Created by Schneider, Abraham R. on 6/5/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation

protocol Optimizer {
    associatedtype StorageType:Storage
    func optimize() -> Op<StorageType>
}

public class GradientDescentOptimizer<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    var alpha:Symbol<S>
    var forward:Op<S>
    var backward:Op<S>
    var params:[Tensor<S>]
    var gradParams:[Tensor<S>]
    
    // NB: With current version of swift, constraints cannot be placed on
    // Op s.t. its differentiable. Until this is changed, this code can fail a
    // at runtime if non-differentiable op is used
    public init(_ op:Op<S>, alpha:Symbol<S>) {
        self.alpha = alpha
        forward = op
        backward = (op as! Differentiable).gradient() as! Op<S>
        
        params = forward.params()
        gradParams = backward.params()
        
        super.init(inputs: [alpha],
                   output: forward.output,
                   labels: ["alpha"])
    }
    
    public override func apply() {
        (backward as! GradientType).reset()
        
        forward.apply()
        backward.apply()
        
        // this is a place where having a TensorScalar class might be nice
        let a:S.ElementType = alpha.output[0]
        for (param, gradParam) in Zip2Sequence(params, gradParams) {
//            print("\(param), \(gradParam), \(param - a*gradParam)")
            param -= a*gradParam
        }
    }
}
