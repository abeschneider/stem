//
//  Connections.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/13/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation
import Tensor

// TODO: can InputType be replaced by Source?
// source for a connection
public struct Source<S:Storage> {
    public var op:Op<S>?
    public var label:String = "output"
    public var index:Int?
    
    public var output:Tensor<S> {
        return op!.outputs[label]![index!]
    }
    
    public var outputs:[Tensor<S>] {
        return op!.outputs[label]!
    }
    
    public init() {}
    
    public init(op:Op<S>, label:String="output", index:Int?=0) {
        self.op = op
        self.label = label
        self.index = index
    }
}

// target for a connection
public struct Target<S:Storage> {
    public var op:Op<S>
    public var label:String
    public var index:Int?
    
    
    public init(op:Op<S>, label:String="input", index:Int?=0) {
        self.op = op
        self.label = label
        self.index = index
    }
}

public func connect<S:Storage>(from:Op<S>, _ outputLabel:String="output", to:Op<S>, _ inputLabel:String="input") {
    to.setInput(inputLabel, to: from, outputLabel)
}

public func connect<S:Storage>(from:[Op<S>], _ outputLabel:String="output", to:Op<S>, _ inputLabel:String="input") {
    to.setInput(inputLabel, to: from, outputLabel)
}

public func connect<S:Storage>(_ source:Source<S>, _ target:Target<S>) {
    target.op.setInput(target.label, to: source.op!, source.label)
}
