//
//  opTests.swift
//  stem
//
//  Created by Schneider, Abraham R. on 5/25/16.
//  Copyright © 2016 none. All rights reserved.
//

import XCTest
import stem


class opTests: XCTestCase {
    typealias S = NativeStorage<Double>

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    func testLinearOp() {
        let num_inputs = 10
        let num_outputs = 10

        // provides a flat view of all parameters to make gradient testing simple
        let storage = S(size: num_inputs*num_outputs + num_outputs)
        let gradStorage = S(size: num_inputs*num_outputs + num_outputs)

        let w = Tensor<S>(Extent(num_outputs, num_inputs), storage: storage)
        let b = Tensor<S>(Extent(num_outputs), storage: storage, offset: num_inputs*num_outputs)

        let gw = Tensor<S>(Extent(num_outputs, num_inputs), storage: gradStorage)
        let gb = Tensor<S>(Extent(num_outputs), storage: gradStorage, offset: num_inputs*num_outputs)

        let params = Tensor<S>(Extent(num_inputs*num_outputs + num_outputs), storage: storage)
        let gradParams = Tensor<S>(Extent(num_inputs*num_outputs + num_outputs), storage: gradStorage)
        
        let input = Symbol<S>(uniform(Extent(num_inputs)))
        let target = Symbol<S>(uniform(Extent(num_outputs)))
        
        let linear = Linear<S>(input: input, weight: w, bias: b)
        let loss = L2Loss<S>(value: linear,  target: target)
        let lossgrad = L2LossGrad<S>(op: loss, input: linear, target: target)
        let lineargrad = LinearGrad<S>(op: linear, input: input, gradInput: loss, weight: gw, bias: gb)
        
//        _testOp(linear, opgrad: lineargrad, params: params, gradParams: gradParams, loss: loss, lossgrad: lossgrad, eps: 10e-6)
        params.uniform()
        
        let eps = 10e-6
        let opgrad = lineargrad
        let op = linear
        let result = checkGradient(params: params, gradParams: gradParams, eps: eps)
        {
            lossgrad.reset()
            opgrad.reset()
            
            // forward
            op.apply()
            loss.apply()
            
            // backward
            lossgrad.apply()
            opgrad.apply()
            
            // return error
            return loss.value
        }
        
        for i in result.indices() {
            XCTAssertLessThanOrEqual(result[i], eps)
        }
    }
    
    func testSigmoidOp() {
        let num_inputs = 10
        let num_outputs = 10
        
        // provides a flat view of all parameters to make gradient testing simple
        let storage = S(size: num_inputs*num_outputs + num_outputs)
        let gradStorage = S(size: num_inputs*num_outputs + num_outputs)
        
        let w = Tensor<S>(Extent(num_outputs, num_inputs), storage: storage)
        let b = Tensor<S>(Extent(num_outputs), storage: storage, offset: num_inputs*num_outputs)
        
        let gw = Tensor<S>(Extent(num_outputs, num_inputs), storage: gradStorage)
        let gb = Tensor<S>(Extent(num_outputs), storage: gradStorage, offset: num_inputs*num_outputs)
        
        let params = Tensor<S>(Extent(num_inputs*num_outputs + num_outputs), storage: storage)
        let gradParams = Tensor<S>(Extent(num_inputs*num_outputs + num_outputs), storage: gradStorage)
        
        let input = Symbol<S>(uniform(Extent(num_inputs)))
        let target = Symbol<S>(uniform(Extent(num_outputs)))
        
        let linear = Linear<S>(input: input, weight: w, bias: b)
        
        
        let sigmoid = Sigmoid<S>(input: linear)
        let lineargrad = LinearGrad<S>(op: linear, input: input, gradInput: sigmoid, weight: gw, bias: gb)
        
        let loss = L2Loss<S>(value: sigmoid,  target: target)
        let sigmoidgrad = SigmoidGrad<S>(op: sigmoid, input: linear, gradInput: loss)
        let lossgrad = L2LossGrad<S>(op: loss, input: sigmoid, target: target)
        
        
        //        _testOp(linear, opgrad: lineargrad, params: params, gradParams: gradParams, loss: loss, lossgrad: lossgrad, eps: 10e-6)
        params.uniform()
        
        let eps = 10e-6
        let opgrad = lineargrad
        let op = linear
        let result = checkGradient(params: params, gradParams: gradParams, eps: eps)
        {
            lossgrad.reset()
            opgrad.reset()
            sigmoidgrad.reset()
            
            // forward
            op.apply()
            sigmoid.apply()
            loss.apply()
            
            // backward
            lossgrad.apply()
            sigmoid.apply()
            opgrad.apply()
            
            // return error
            return loss.value
        }
        
        for i in result.indices() {
            XCTAssertLessThanOrEqual(result[i], eps)
        }
    }


//    func _testOp<B:Op<D>, L:Op<D>, LG:Op<D>, D:Storage where B:Gradient, L:Loss, LG:protocol<Gradient, Loss>>
//        (op:Op<D>,
//         opgrad:B,
//         params:Tensor<D>,
//         gradParams:Tensor<D>,
//         loss:L,
//         lossgrad:LG,
//         eps:Double)
//    {
//        params.uniform()
//        
//        let result = checkGradient(params: params, gradParams: gradParams, eps: eps)
//        {
//            lossgrad.reset()
//            opgrad.reset()
//            
//            // forward
//            op.apply()
//            loss.apply()
//            
//            // backward
//            lossgrad.apply()
//            opgrad.apply()
//            
//            // return error
//            return loss.value
//        }
//        
//        for i in result.indices() {
//            XCTAssertLessThanOrEqual(result[i], eps)
//        }
//    }

}
