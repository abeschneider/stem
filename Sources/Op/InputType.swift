//
//  InputType.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/13/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation
import Tensor

// contains selected output from a given op that
// provides input to target op
//public struct InputType<S:Storage> {
//    public var op:Op<S>?
//    
//    // outputs of op
//    public var _outputs:[Tensor<S>]
//    
//    public var index:Int?
//    public var label:String = "output"
//    
//    public init() {
//        _outputs = []
//        index = 0
//    }
//    
//    public init(_ op:Op<S>, _ label:String) {
//        self.op = op
//        self.label = label
//        _outputs = op.outputs[label]!
//        index = 0
//    }
//    
//    public init(_ op:Op<S>) {
//        self.op = op
//        _outputs = op.outputs["output"]!
//        index = 0
//    }
//    
//    public func output() -> Tensor<S> {
//        return _outputs[index!]
//    }
//    
//    public func outputs() -> [Tensor<S>] {
//        return _outputs
//    }
//}
