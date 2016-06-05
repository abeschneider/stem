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
        let eps:Double = 10e-6
        let input = Symbol<S>(uniform(Extent(10)))
        
        let gradOutput = Symbol<S>(zeros(Extent(5)))
        
        let linear = Linear<S>(inputSize: 10, outputSize: 5)
        linear.bias.uniform()
        linear.setInput("input", to: input)

        let linearGrad = linear.gradient() as! LinearGrad<S>
        linearGrad.setInput("gradOutput", to: gradOutput)
        
        // test gradient wrt the input
        let inputError = checkGradient(linear, grad: linearGrad, params: input.output!, gradParams: linearGrad.output!, eps: eps)
        XCTAssertLessThan(inputError, eps)
        
        // test gradient wrt the parameters
        let weightError = checkGradient(linear, grad: linearGrad, params: linear.weight, gradParams: linearGrad.weight, eps: eps)
        XCTAssertLessThan(weightError, eps)

        let biasError = checkGradient(linear, grad: linearGrad, params: linear.bias, gradParams: linearGrad.bias, eps: eps)
        XCTAssertLessThan(biasError, eps)
    }
    
    func testLinearOp2() {
        let eps:Double = 10e-6
        let input = Symbol<S>(uniform(Extent(10, 3)))
        
        let gradOutput = Symbol<S>(zeros(Extent(5, 3)))
        
        let linear = Linear<S>(inputSize: 10, outputSize: 5)
        linear.bias.uniform()
        linear.setInput("input", to: input)

        let linearGrad = linear.gradient() as! LinearGrad<S>
        linearGrad.setInput("gradOutput", to: gradOutput)
        
        // TODO: figure out why this is required needed (hint: causes resize)
        linearGrad.apply()
        
        // test gradient wrt the input
        let inputError = checkGradient(linear, grad: linearGrad, params: input.output!, gradParams: linearGrad.output!, eps: eps)
        XCTAssertLessThan(inputError, eps)
    }
    
    func testSigmoid() {
        let eps:Double = 10e-6
        let input = Symbol<S>(uniform(Extent(10)))
        let gradOutput = Symbol<S>(zeros(Extent(10)))
        
        let sigmoid = Sigmoid<S>(size: 10)
        sigmoid.setInput("input", to: input)
        
        let sigmoidGrad = sigmoid.gradient() as! SigmoidGrad<S>
        sigmoidGrad.setInput("gradOutput", to: gradOutput)
        
        // test gradient wrt the input
        let inputError = checkGradient(sigmoid, grad: sigmoidGrad, params: input.output!, gradParams: sigmoidGrad.output!, eps: eps)
        XCTAssertLessThan(inputError, eps)
    }
    
    func testLoss() {
        let eps:Double = 10e-6
        let input = Symbol<S>(uniform(Extent(10)))
        let target = Symbol<S>(uniform(Extent(10)))
        
        let loss = L2Loss<S>(target: target)
        loss.setInput("input", to: input)
        
        let lossGrad = loss.gradient() as! L2LossGrad<S>

        // test gradient wrt to the input
        let inputError = checkGradient(loss, grad: lossGrad, input: input.output!, eps: eps)
        XCTAssertLessThan(inputError, eps)
    }
    
    func testSequence() {
        let eps:Double = 10e-6
        let input = Symbol<S>(uniform(Extent(10)))
        let gradOutput = Symbol<S>(zeros(Extent(5)))


        let seq = Sequence<S>(
            input,
            Linear<S>(inputSize: 10, outputSize: 5),
            Sigmoid<S>(size: 5)
        )

        let seqGrad = seq.gradient() as! SequenceGradient<S>
        seqGrad.setInput("gradOutput", to: gradOutput)
        seqGrad.ops[0].setInput("gradOutput", to: gradOutput)
        
        // test gradient wrt to the input
        let inputError = checkGradient(seq, grad: seqGrad, params: input.output!, gradParams: seqGrad.output!, eps: eps)
        XCTAssertLessThan(inputError, eps)
    }
    
    func testLogOp() {
        let eps = 10e-6
        let input = Symbol<S>(uniform(Extent(10)))
        let gradOutput = Symbol<S>(zeros(Extent(10)))
        
        let log = Log<S>(size: 10)
        log.setInput("input", to: input)
        

        let logGrad = log.gradient() as! LogGrad<S>
        logGrad.setInput("gradOutput", to: gradOutput)
        
        // test gradient wrt to the input
        let inputError = checkGradient(log, grad: logGrad, params: input.output!, gradParams: logGrad.output!, eps: eps)
        XCTAssertLessThan(inputError, eps)
    }
}
