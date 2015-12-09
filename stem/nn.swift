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

/*struct MultiTensorIndex<StorageType:Storage>: GeneratorType {
    typealias Element = (StorageView<StorageType>, Int)
    
    var views:[StorageView<StorageType>]
    var currentView:Int
    var indices:GeneratorSequence<StorageViewIndex<StorageType>>
    
    init(views:StorageView<StorageType>...) {
        self.views = views
        currentView = 0
        indices = views[currentView].storageIndices()
    }
    
    mutating func next() -> Element? {
        var offset:Int?
        repeat {
            offset = indices.next()
            if offset == nil {
                if ++currentView > views.count { return nil }
                indices = views[++currentView].storageIndices()
            }
        } while offset == nil
        
        if offset == nil { return nil }
        return (views[currentView], offset!)
    }
}*/


/*
protocol Module {
    typealias StorageType:Storage

    // general form -- specialized forms can appear in child classes
    func forward(input:Tensor<StorageType>) throws -> Tensor<StorageType>
}

protocol GradientModule {
    typealias StorageType:Storage
    
    // general form -- specialized forms can appear in child classes
    func backward(input:Tensor<StorageType>, grad_output:Tensor<StorageType>) throws -> Tensor<StorageType>
    
    func clear()
}

class LinearModule<S:Storage>: Module, GradientModule {
    typealias StorageType = S
    
    var weight:Matrix<StorageType>
    var bias:RowVector<StorageType>
    
    var grad_weight:Matrix<StorageType>
    var grad_bias:RowVector<StorageType>
    
    var output:Tensor<StorageType>?
    var grad_input:Tensor<StorageType>?
    
    init(input_size:Int, output_size:Int) {
        weight = Matrix<StorageType>(rows: input_size, cols: output_size)
        bias = RowVector<StorageType>(rows: output_size)
        
        grad_weight = Matrix<StorageType>(rows: input_size, cols: output_size)
        grad_bias = RowVector<StorageType>(rows: output_size)
    }
    
    init(weight:Matrix<StorageType>, bias:RowVector<StorageType>?=nil, gradWeight:Matrix<StorageType>?=nil, gradBias:RowVector<StorageType>?=nil) {
        self.weight = weight
        if let b = bias {
            self.bias = b
        } else {
            self.bias = RowVector<StorageType>(rows: self.weight.shape[1])
        }
        
        if let gw = gradWeight {
            grad_weight = gw
        } else {
            grad_weight = Matrix(rows: self.weight.shape[0], cols: self.weight.shape[1])
        }
        
        if let gb = gradBias {
            grad_bias = gb
        } else {
            grad_bias = RowVector(rows: self.bias.shape[0])
        }
    }
    
    func forward(input:Tensor<StorageType>) throws -> Tensor<StorageType> {
        throw IllegalOperation()
    }
    
    func forward(input:Vector<StorageType>) -> Vector<StorageType> {
        // verify that output is the correct shape
        if output == nil || output!.shape[0] != weight.shape[1] {
            // if not, allocate new output storage
            output = Vector<StorageType>(rows: weight.shape[1])
        }
        
        if let out = output {
            // output = W'*input
            print("shapes: \(weight.transpose().shape), \(input.shape), \(out.shape)")
            dot(left: weight.transpose(), right: input, result: out)
            
            // output += bias
            add(left: out, right: bias, result: out)
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
            // output = W'*input
            dot(left: weight.transpose(), right: input, result: out)
            
            // output += bias
            add(left: out as! Matrix, right: bias, result: out as! Matrix)
        }
        
        return output! as! Matrix
    }
    
    func backward(input:Tensor<StorageType>, grad_output:Tensor<StorageType>) throws -> Tensor<StorageType> {
        throw IllegalOperation()
    }
    
    // create a version of backward for Matrix
    func backward(input:Vector<StorageType>, grad_output:Vector<StorageType>) -> Vector<StorageType> {
        if grad_input == nil || grad_input!.shape[0] != weight.shape[1] {
            grad_input = Vector<StorageType>(rows: weight.shape[0])
        }

        // grad_input += W*grad_output
        dot(left: weight, right: grad_output, result: grad_input!)
        
        // dW += grad_output*input'
        outer(left: input, right: grad_output, result: grad_weight)
        
        // bias += grad_output
        add(left: grad_bias, right: grad_output, result: grad_bias)
        
        return grad_input! as! Vector
    }
    
    func clear() {
        fill(grad_weight, value: 0)
        fill(grad_bias, value: 0)
    }
    
//    func backward(input:Matrix<StorageType>, grad_output:Matrix<StorageType>) -> Matrix<StorageType> {
//    }
}
*/