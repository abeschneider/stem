//
//  nn.swift
//  stem
//
//  Created by Abe Schneider on 11/29/15.
//  Copyright © 2015 none. All rights reserved.
//

import Foundation

//class Module<StorageType:Storage> {
//    var input_shape:Extent
//    var output_shape:Extent
//    var output:Tensor<StorageType>
//    
//    init(input_shape:Extent, output_shape:Extent) {
//        self.input_shape = input_shape
//        self.output_shape = output_shape
//        
//        output = Tensor<StorageType>(shape: output_shape)
//    }
//}


protocol Op {
    associatedtype StorageType:Storage
    
    // general form -- specialized forms can appear in child classes
    func apply(input:Tensor<StorageType>) throws -> Tensor<StorageType>
}

protocol Shapeable {
    var shape:Extent { get }
}

protocol Gradient {
    associatedtype StorageType:Storage
    
    // general form -- specialized forms can appear in child classes
    func apply(input:Tensor<StorageType>, gradOutput:Tensor<StorageType>) throws -> Tensor<StorageType>
    
    func clear()
}

//protocol Criterion {
//    typealias StorageType:Storage
//    
//    func forward(input:Tensor<StorageType>) -> StorageType.ElementType
//    func backward(input:Tensor<StorageType>) -> Tensor<StorageType>
//}

class LinearOp<S:Storage>: Op, Shapeable {
    typealias StorageType = S
    
    var weight:Matrix<StorageType>
    var bias:RowVector<StorageType>
    
    var output:Tensor<StorageType>?
    
    var shape:Extent { return weight.shape }
    
    init(input_size:Int, output_size:Int) {
        weight = Matrix<StorageType>(rows: input_size, cols: output_size)
        bias = RowVector<StorageType>(cols: output_size)
    }
    
    init(weight:Matrix<StorageType>, bias:RowVector<StorageType>?=nil) {
        self.weight = weight
        if let b = bias {
            self.bias = b
        } else {
            self.bias = RowVector<StorageType>(cols: self.weight.shape[1])
        }
    }
    
    // handle case were we have to do static-dispatch
    func apply(input:Tensor<StorageType>) throws -> Tensor<StorageType> {
        // figure out which of the specialized types applies
        if input.shape.span == 1 {
            return try apply(ColumnVector<StorageType>(input))
        } else if input.shape.span == 2 {
            return try apply(Matrix<StorageType>(input))
        }
        
        assert(false)
    }
    
    func apply(input:ColumnVector<StorageType>) throws -> Vector<StorageType> {
        if input.shape.elements != weight.shape[0] {
            throw TensorError.SizeMismatch(lhs: input.shape, rhs: weight.shape)
        }
        
        // verify that output is the correct shape
        if output == nil || output!.shape[0] != weight.shape[1] {
            // if not, allocate new output storage
            output = RowVector<StorageType>(cols: weight.shape[1])
        }
        
        if let out = output {
            // out = weight'*input
            fill(out, value: 0)
            try dot(left: weight.transpose(), right: input, result: out)

            // out += bias
            iadd(left: out as! RowVector, right: bias)
        }
        
        return output! as! Vector
    }

    func apply(input:Matrix<StorageType>) throws -> Matrix<StorageType> {
        // verify that output is the correct shape
        if output == nil || output!.shape[0] != weight.shape[0]{
            // if not, allocate new output storage
            output = Matrix<StorageType>(rows: weight.shape[1], cols: weight.shape[0])
        }
        
        if let out = output {
            // out = weight'*input
            fill(out, value: 0)
            try dot(left: weight.transpose(), right: input, result: out)
            
            // out += bias
            iadd(left: out as! Matrix, right: bias)
        }
        
        return output! as! Matrix
    }
}

class LinearGradient<S:Storage>: Gradient, Shapeable {
    typealias StorageType = S
    
    var gradWeight:Matrix<StorageType>
    var gradBias:RowVector<StorageType>
    var gradInput:Tensor<StorageType>?
    
    var shape:Extent { return gradWeight.shape }
    
    init(input_size:Int, output_size:Int) {
        gradWeight = Matrix<StorageType>(rows: input_size, cols: output_size)
        gradBias = RowVector<StorageType>(cols: output_size)
    }
    
    init(weight:Matrix<StorageType>, bias:RowVector<StorageType>?=nil) {
        self.gradWeight = weight
        if let b = bias {
            self.gradBias = b
        } else {
            self.gradBias = RowVector<StorageType>(cols: self.gradWeight.shape[1])
        }
    }
    
    func apply(input:Tensor<StorageType>, gradOutput:Tensor<StorageType>) throws -> Tensor<StorageType> {
        // figure out which of the specialized types applies
        if input.shape.span == 1 {
            return try apply(Vector<StorageType>(input), gradOutput: ColumnVector<StorageType>(gradOutput))
        } else if input.shape.span == 2 {
            return try apply(Matrix<StorageType>(input), gradOutput: Matrix<StorageType>(gradOutput))
        }
        
        assert(false)
    }
    
    func apply(input:Vector<StorageType>, gradOutput:ColumnVector<StorageType>) throws -> Vector<StorageType>
    {
        if gradInput == nil || gradInput!.shape[0] != gradWeight.shape[0] {
            gradInput = ColumnVector<StorageType>(rows: gradWeight.shape[0])
        }
        
        // dW += grad_output*input'
        outer(left: input, right: gradOutput, result: gradWeight)
        
        // grad_bias += grad_output
        iadd(left: gradBias, right: gradOutput)
        
        // gradInput! = weight*gradOutput
        let tmp:Vector<StorageType> = try gradWeight⊙gradOutput
        iadd(left: self.gradInput! as! Vector<StorageType>, right: tmp)
        
        return self.gradInput! as! Vector
    }
    
    func clear() {
        fill(gradWeight, value: 0)
        fill(gradBias, value: 0)
        if gradInput != nil { fill(gradInput!, value: 0) }
    }
}

class SigmoidModule<S:Storage>: Op {
    typealias StorageType = S
    
    var output:Tensor<StorageType>?
    
    func apply(input:Tensor<StorageType>) -> Tensor<StorageType> {
        assert(false)
    }
    
    func apply(input:ColumnVector<StorageType>) -> Vector<StorageType> {
        // verify that output is the correct shape
        if output == nil || output!.shape[0] != input.shape[0] {
            // if not, allocate new output storage
            output = ColumnVector<StorageType>(rows: input.shape[0])
        }

        if let out = output {
            for i in input.storageIndices() {
                out.storage[i] = StorageType.ElementType(1.0) / (StorageType.ElementType(1.0) + StorageType.ElementType.exp(-input.storage[i]))
            }
        }
        
        return output! as! Vector<StorageType>
    }
}

class SigmoidGradient<S:Storage>: Gradient {
    typealias StorageType = S
    
    var gradInput:Tensor<StorageType>?
    
    func apply(input:Tensor<StorageType>, gradOutput:Tensor<StorageType>) -> Tensor<StorageType> {
        assert(false)
    }

    func apply(input:Vector<StorageType>,
        gradOutput:ColumnVector<StorageType>) -> Vector<StorageType>
    {
        // grad_input += grad_output % (1.0 - *output) % *output;
        if gradInput == nil || gradInput!.shape[0] != input.shape[0] {
            gradInput = ColumnVector<StorageType>(rows: input.shape[0])
        }
        
        if let gin = gradInput {
            // out = StorageType.ElementType(1.0) - out
            let tmp = StorageType.ElementType(1.0) - gin
            imul(left: tmp, right: gin)
            imul(left: tmp, right: gradOutput)
            gin += tmp
        }
        
        return gradInput! as! Vector<StorageType>
    }
    
    func clear() {
        if let g = gradInput {
            fill(g, value: 0)
        }
    }
}

//class L2Loss<S:Storage where S.ElementType:NumericType>: Criterion {
//    typealias StorageType = S
//    
//    var target:Tensor<StorageType>
//    
//    init(target:Tensor<StorageType>) {
//        self.target = target
//    }
//    
//    func apply(input:Tensor<StorageType>) -> StorageType.ElementType {
//        let d:Tensor<StorageType> = input - target
//        let p = pow(d, StorageType.ElementType(2.0))
//        let s = sum(p)
//        let result = StorageType.ElementType(0.5)*s
//        return result
//    }
//    
//    func backward(input:Tensor<StorageType>) -> Tensor<StorageType> {
//        return input - target
//    }
//}