//
//  optimize.swift
//  stem
//
//  Created by Schneider, Abraham R. on 6/5/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation
import Tensor

protocol Optimizer {
    associatedtype StorageType:Storage
    func optimize() -> Op<StorageType>
}

open class GradientDescentOptimizer<S:Storage>: Op<S> where S.ElementType:FloatNumericType {
    var alpha:Constant<S>
    var forward:Op<S>
    var backward:Op<S>
    var params:[Tensor<S>]
    var gradParams:[Tensor<S>]
    
    // NB: With current version of swift, constraints cannot be placed on
    // Op s.t. its differentiable. Until this is changed, this code can fail a
    // at runtime if non-differentiable op is used
    public init(_ op:Op<S>, alpha:Constant<S>) {
        self.alpha = alpha
        forward = op
        backward = (op as! Differentiable).gradient() as! Op<S>
        
        params = forward.params()
        gradParams = backward.params()
        
//        super.init(inputs: [("alpha", InputType(alpha))],
//                   outputs: forward.outputs)
//        super.init()
        super.init(inputs: ["alpha"], outputs: [])
        connect(from: alpha, "output", to: self, "alpha")
        let forwardOutputs:[Tensor<S>] = forward.outputs["output"]!
        outputs["output"] = forwardOutputs
    }

    required public init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    open override func apply() {
        (backward as! GradientType).reset()
        
        forward.apply()
        backward.apply()
        
        // this is a place where having a TensorScalar class might be nice
        let a:S.ElementType = alpha.output[0]
        for (param, gradParam) in zip(params, gradParams) {
            param -= a*gradParam
        }
    }
}
