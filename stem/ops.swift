//
//  ops.swift
//  stem
//
//  Created by Schneider, Abraham R. on 5/25/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation

func createUID() -> Int {
    return uniform()
}


public class Op<StorageType:Storage>: Hashable {
    var id:Int

    public var inputs:[Op<StorageType>] = []
    public var output:Tensor<StorageType>
    
    required public init(inputs:[Op<StorageType>], output:Tensor<StorageType>) {
        self.inputs = inputs
        self.output = output
        id = createUID()
    }
    
    public func apply() {
        assertionFailure()
    }
    
    public var hashValue: Int { return id }
}

public func ==<S:Storage>(lhs:Op<S>, rhs:Op<S>) -> Bool {
    return lhs.id == rhs.id
}

public protocol Gradient {
    associatedtype StorageType:Storage
    func reset()
    func update(alpha:StorageType.ElementType)
}

public protocol Loss {
    associatedtype StorageType:Storage
    
    var value:StorageType.ElementType { get }
}

public class Symbol<S:Storage>: Op<S> {
    public init(_ input:Tensor<S>) {
        super.init(inputs: [], output: input)
    }
    
    public init(_ shape:Extent) {
        super.init(inputs: [], output: Tensor<S>(shape))
    }
    
    public func set(input:Tensor<S>) {
        output = copy(input)
    }
    
    public override func apply() {}
}

public class Sigmoid<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    public init(input:Op<S>) {
        super.init(inputs: [input], output: Tensor<S>(input.output.shape))
    }
    
    public override func apply() {
        sigmoid(inputs[0].output, output: output)
    }
}

public class SigmoidGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    public typealias StorageType = S
    
    public var sigmoid:Sigmoid<S> { return inputs[0] as! Sigmoid<S> }
    
    public init(op:Sigmoid<S>, input:Op<S>, gradInput:Op<S>) {
        super.init(inputs: [op, input, gradInput], output: Tensor<S>(input.output.shape))
    }
    
    public override func apply() {
        mul(sigmoid.output, (1-sigmoid.output), result: output)
    }
    
    public func reset() {
        fill(output, value: 0)
    }
    
    public func update(alpha:S.ElementType) {}
}

public class Linear<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    public var weight:Tensor<S>
    public var bias:Tensor<S>
    
    public init(numInputs:Int, numOutputs:Int) {
        weight = uniform(Extent(numOutputs, numInputs))
        bias = zeros(Extent(numOutputs))
        super.init(inputs: [], output: zeros(Extent(numOutputs, numInputs)))
    }
    
    public init(input:Op<S>, numOutputs:Int) {
        let inputSize = input.output.shape[0]
        weight = uniform(Extent(numOutputs, inputSize))
        bias = zeros(Extent(numOutputs))
        super.init(inputs: [input], output: zeros(bias.shape))
    }
    
    public init(input:Op<S>, weight:Tensor<S>, bias:Tensor<S>) {
        self.weight = weight
        self.bias = bias
        super.init(inputs: [input], output: zeros(bias.shape))
    }
    
    public override func apply() {
        dot(weight, inputs[0].output, result: output)
        add(output, bias, result: output)
    }
}

public class LinearGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    public typealias StorageType = S
    public var weight:Tensor<S>
    public var bias:Tensor<S>
    
    public var linear:Linear<S> { return inputs[0] as! Linear<S> }
    public var input:Tensor<S> { return inputs[1].output }
    public var gradInput:Tensor<S> { return inputs[2].output }
    
    public required init(op:Linear<S>, input:Op<S>, gradInput:Op<S>, weight:Tensor<S>?=nil, bias:Tensor<S>?=nil) {
        if let w = weight {
            self.weight = w
        } else {
            self.weight = zeros(op.weight.shape)
        }
        
        if let b = bias {
            self.bias = b
        } else {
            self.bias = zeros(Extent(op.weight.shape[0]))
        }
        
        super.init(inputs: [op, input, gradInput], output: zeros(Extent(op.weight.shape[1])))
    }
    
    public override func apply() {
        outer(gradInput, input, result: weight)
        bias += gradInput
        dot(linear.weight.transpose(), gradInput, result: output)
    }
    
    public func reset() {
        fill(weight, value: 0)
        fill(bias, value: 0)
        fill(output, value: 0)
    }
    
    public func update(alpha:S.ElementType) {
        linear.weight -= (alpha*weight)
        linear.bias -= (alpha*bias)
    }
}

public class L2Loss<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Loss {
    public typealias StorageType = S
    
    public var value:S.ElementType = 0
    
    public var inputValue:Tensor<S> { return inputs[0].output }
    public var targetValue:Tensor<S> { return inputs[1].output }
    
    public init(value:Op<S>, target:Op<S>) {
        super.init(inputs: [value, target], output: Tensor<S>(value.output.shape))
    }
    
    public override func apply() {
        sub(inputValue, targetValue, result: output)
        pow(output, 2, result: output)
        value = sum(output)
    }
}

public class L2LossGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    public typealias StorageType = S
    
    public var inputValue:Tensor<S> { return inputs[1].output }
    public var targetValue:Tensor<S> { return inputs[2].output }
    
    public init(op:L2Loss<S>, input:Op<S>, target:Op<S>) {
        super.init(inputs: [op, input, target], output: Tensor<S>(op.output.shape))
    }
    
    public override func apply() {
        sub(inputValue, targetValue, result: output)
        output *= 2
    }
    
    public func reset() {
        fill(output, value: 0)
    }
    
    public func update(alpha:S.ElementType) {}
}


