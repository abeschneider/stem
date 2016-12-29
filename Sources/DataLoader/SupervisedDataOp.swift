//
//  SupervisedDataOp.swift
//  stem
//
//  Created by Abraham Schneider on 12/28/16.
//
//

import Foundation
import Tensor
import Op

open class SupervisedDataOp<S:Storage, DataType:SupervisedData>: Op<S> where DataType.StorageType==S {
    var data:DataType
    var process:(Tensor<S>, Tensor<S>) -> (Tensor<S>, Tensor<S>)
    
    open var label:Tensor<S> {
        get { return outputs["label"]![0] }
        set { outputs["label"] = [newValue] }
    }
    
    public init(data:DataType) {
        self.data = data
        
        // default to do nothing else
        self.process = {(d:Tensor<S>, l:Tensor<S>) -> (Tensor<S>, Tensor<S>) in
            return (d, l)
        }
        
        super.init(inputs: [], outputs: ["output", "label"])
        outputs["output"] = Tensor<S>(data.imageSize)
        outputs["label"] = Tensor<S>(data.labelSize)
    }
    
    public required init(op: Op<S>, shared: Bool) {
        fatalError("init(op:shared:) has not been implemented")
    }
    
    public func setProcess(process:@escaping (Tensor<S>, Tensor<S>) -> (Tensor<S>, Tensor<S>)) {
        self.process = process
    }
    
    open override func apply() {
        let tuple = data.next()!
        let sample:Tensor<S> = tuple.0
        let sampleLabel:Tensor<S> = tuple.1
        
        output.resize(sample.shape)
        label.resize(sampleLabel.shape)
        
        copy(from: sample, to: output)
        copy(from: sampleLabel, to: label)        
    }
}
