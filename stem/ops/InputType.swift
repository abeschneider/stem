//
//  InputType.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/13/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation

// contains selected output from a given op that
// provides input to target op
public struct InputType<S:Storage> {
    public var op:Op<S>?
    
    // outputs of op
    public var outputs:[Tensor<S>]
    
    public var index:Int?
    
    public init() {
        outputs = []
        index = 0
    }
    
    public init(_ op:Op<S>, _ label:String) {
        self.op = op
        outputs = op.outputs[label]!
        index = 0
    }
    
    public init(_ op:Op<S>) {
        self.op = op
        outputs = op.outputs["output"]!
        index = 0
    }
    
    public func output() -> Tensor<S> {
        return outputs[index!]
    }
    
    public func output() -> [Tensor<S>] {
        return outputs
    }
}
