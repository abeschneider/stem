//
//  main.swift
//  stem
//
//  Created by Abraham Schneider on 11/30/16.
//
//

import Foundation
import Tensor
import Op
import DataLoader

typealias F = NativeStorage<Float>

//func makeConvLayer(numFilters:Int, filterSize:Extent) -> Op<F> {
//    let conv = Conv2dOp<F>(numFilters: numFilters, filterSize: filterSize)
//    let relu = ReLUOp<F>()
//    let pool = PoolingOp<F>(poolingSize: Extent(2, 2), stride: Extent(2, 2), evalFn: max)
//
//    return SequentialOp<F>(conv, relu, pool)
//}

func createNetwork<DataType>(data:DataType) -> SequentialOp<F> where DataType:SupervisedData, DataType.StorageType==F {
    let model = SequentialOp<F>()
    
    // layer 1
    let input = SupervisedDataOp<F, DataType>(data: data)
    model.append(input)
    model.append(Conv2dOp(numFilters: 1, filterSize: Extent(3, 3)))
    model.append(ReLUOp())
    model.append(PoolingOp(poolingSize: Extent(2, 2), stride: Extent(2, 2), evalFn: max))
    // output size is now (14, 14)
    
    // convert image to vector for dense layers
    model.append(FlattenOp())
    
    // reduce dimensions down to 10 (0..<10 digits)
    model.append(LinearOp(outputSize: 10))
    
    // make this a logistic regression problem
    model.append(LogSoftMaxOp())
    
    // loss function using labeled data from input as the target
    model.append(CrossEntropyLoss<F>(target: Source(op: input, label: "label")))
    
    return model
}

func train(model:SequentialOp<F>, params:[Tensor<F>], modelGrad:Op<F>, gradParams:[Tensor<F>], batchSize:Int) -> F.ElementType {
    // forward (includes loss)
    model.apply()
    
    // backward
    modelGrad.apply()
    
    // average across batch
    var loss = (model[model.count-1] as! CrossEntropyLoss<F>).value
    loss /= Float(batchSize)
    for gradParam in gradParams {
        gradParam /= Float(batchSize)
    }
    
    // update parameters
    for (param, gradParam) in zip(params, gradParams) {
        param -= 10e-1*gradParam
    }

    // zero out computed gradients
    modelGrad.reset()
    
    
    // return error from loss
    return (model[model.count-1] as! CrossEntropyLoss<F>).value
}


func main() {
    let data = MNISTData(imageData: URL(string: "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")!,
                         labelData: URL(string: "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")!,
                         batchSize: 5)
    
    let (input, label) = data.next()!
    print(input.shape)
    print(label.shape)
    
    let model:SequentialOp<F> = createNetwork(data: data)
    let modelGrad = model.gradient() as! CollectionGradient<F>
    
    print(model)
    print(modelGrad)
    
    let params = model.params()
    let gradParams = modelGrad.params()

    let batchSize = data.batchSize
    let size = data.count / batchSize
    for epoch in 0..<100 {
        print("epoch: \(epoch)")
        for _ in 0..<size {
            var loss:Float = 0.0
            for _ in 0..<batchSize {
                loss += train(model: model, params: params, modelGrad: modelGrad, gradParams: gradParams, batchSize: batchSize)
            }
        
            loss /= Float(batchSize)
            print("loss = \(loss)")
        }
    }
}

main()
