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


protocol Module {
    typealias StorageType:Storage

    // general form -- specialized forms can appear in child classes
    func forward(input:Tensor<StorageType>) -> Tensor<StorageType>
}

protocol GradientModule {
    typealias StorageType:Storage
    
    // general form -- specialized forms can appear in child classes
    func backward(input:Tensor<StorageType>, gradOutput:Tensor<StorageType>) -> Tensor<StorageType>
    
    func clear()
}

protocol CriterionModule {
    typealias StorageType:Storage
    
    func forward(input:Tensor<StorageType>) -> StorageType.ElementType
    func backward(input:Tensor<StorageType>) -> Tensor<StorageType>
}

class LinearModule<S:Storage>: Module, GradientModule {
    typealias StorageType = S
    
    var weight:Matrix<StorageType>
    var bias:RowVector<StorageType>
    
    var gradWeight:Matrix<StorageType>
    var gradBias:RowVector<StorageType>
    
    var output:Tensor<StorageType>?
    var gradInput:Tensor<StorageType>?
    
    init(input_size:Int, output_size:Int) {
        weight = Matrix<StorageType>(rows: input_size, cols: output_size)
        bias = RowVector<StorageType>(cols: output_size)
        
        gradWeight = Matrix<StorageType>(rows: input_size, cols: output_size)
        gradBias = RowVector<StorageType>(cols: output_size)
    }
    
    init(weight:Matrix<StorageType>, bias:RowVector<StorageType>?=nil, gradWeight:Matrix<StorageType>?=nil, gradBias:RowVector<StorageType>?=nil) {
        self.weight = weight
        if let b = bias {
            self.bias = b
        } else {
            self.bias = RowVector<StorageType>(cols: self.weight.shape[1])
        }
        
        if let gw = gradWeight {
            self.gradWeight = gw
        } else {
            self.gradWeight = Matrix(rows: self.weight.shape[0], cols: self.weight.shape[1])
        }
        
        if let gb = gradBias {
            self.gradBias = gb
        } else {
            self.gradBias = RowVector(cols: self.bias.shape[0])
        }
    }
    
    func forward(input:Tensor<StorageType>) -> Tensor<StorageType> {
        assert(false)
    }
    
    func forward(input:ColumnVector<StorageType>) -> Vector<StorageType> {
        assert(input.shape.elements == weight.shape[0])
        
        // verify that output is the correct shape
        if output == nil || output!.shape[0] != weight.shape[1] {
            // if not, allocate new output storage
            output = RowVector<StorageType>(cols: weight.shape[1])
        }
        
        if let out = output {
            // out = weight'*input
            fill(out, value: 0)
            dot(left: weight.transpose(), right: input, result: out)

            // out += bias
            iadd(left: out as! RowVector, right: bias)
        }
        
        return output! as! Vector
    }

    func forward(input:Matrix<StorageType>) -> Matrix<StorageType> {
        // verify that output is the correct shape
        if output == nil || output!.shape[0] != weight.shape[0]{
            // if not, allocate new output storage
            output = Matrix<StorageType>(rows: weight.shape[1], cols: weight.shape[0])
        }
        
        if let out = output {
            // out = weight'*input
            fill(out, value: 0)
            dot(left: weight.transpose(), right: input, result: out)
            
            // out += bias
            iadd(left: out as! Matrix, right: bias)
        }
        
        return output! as! Matrix
    }
    
    func backward(input:Tensor<StorageType>, gradOutput:Tensor<StorageType>) -> Tensor<StorageType> {
        assert(false)
    }
    
    // create a version of backward for Matrix
    func backward(input:Vector<StorageType>, gradOutput:ColumnVector<StorageType>) -> Vector<StorageType> {
        if gradInput == nil || gradInput!.shape[0] != weight.shape[0] {
            gradInput = ColumnVector<StorageType>(rows: weight.shape[0])
        }

        gradInput! = weight*gradOutput
        
        // dW += grad_output*input'
        outer(left: input, right: gradOutput, result: gradWeight)
        
        // grad_bias += grad_output
        iadd(left: gradBias, right: gradOutput)
        
        return gradInput! as! Vector
    }
    
    func clear() {
        fill(gradWeight, value: 0)
        fill(gradBias, value: 0)
    }
}

class L2Loss<S:Storage where S.ElementType:NumericType>: CriterionModule {
    typealias StorageType = S
    
    var target:Tensor<StorageType>
    
    init(target:Tensor<StorageType>) {
        self.target = target
    }
    
    func forward(input:Tensor<StorageType>) -> StorageType.ElementType {
        let d:Tensor<StorageType> = input - target
        let p = pow(d, StorageType.ElementType(2.0))
        let s = sum(p)
        let result = StorageType.ElementType(0.5)*s
        return result
    }
    
    func backward(input:Tensor<StorageType>) -> Tensor<StorageType> {
        return input - target
    }
}