//
//  constant.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/12/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation

open class Constant<S:Storage>: Op<S> {
    public init(_ value:S.ElementType) {
        super.init(inputs: [], outputs: ["output"])
        outputs["output"] = [Tensor<S>([value])]
    }
    
    public init(_ value:Tensor<S>) {
        super.init(inputs: [], outputs: ["output"])
        outputs["output"] = [value]
    }
    
    public init(_ shape:Extent) {
        super.init(inputs: [], outputs: ["output"])
        outputs["output"] = [Tensor<S>(shape)]
    }
    
//    public init(data:DataSequence<S>) {
//        super.init(inputs: [], outputs: ["output"])
//        outputs["output"] = [data.next()!]
//    }
    
    // required for Copyable
    public required init(op:Op<S>, shared:Bool) {
        super.init(inputs: [], outputs: ["output"])
        outputs["output"] = [Tensor<S>(op.output.shape)]
    }
    
    open func set(_ input:Tensor<S>, copy makeCopy:Bool=true) {
        if makeCopy {
            copy(from: input, to: output)
        } else {
            output = input
        }
    }
    
    open override func apply() {}
}
