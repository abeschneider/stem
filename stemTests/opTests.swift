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
        let input = Symbol<S>(uniform(Extent(10)))
        let target = Symbol<S>(uniform(Extent(5)))
        
        let linear = Linear<S>(inputSize: 10, outputSize: 5)
        linear.setInput("input", to: input)
        
        
        let loss = L2Loss(target: target)
        loss.setInput("input", to: linear)
        
        let lossGrad = loss.gradient()
        let linearGrad = linear.gradient()
        linearGrad.setInput("gradOutput", to: lossGrad)
        
        
        let eps = 10e-6
        let result = checkGradient(params: input.output!,
                                   gradParams: (linearGrad as! Op<S>).output!,
                                   eps: eps)
        {
            linearGrad.reset()
            lossGrad.reset()
            
            linear.apply()
            loss.apply()
            
            lossGrad.apply()
            linearGrad.apply()
            
            return loss.value
        }
        
        XCTAssert(isClose(result, zeros(Extent(result.shape)), eps: eps))
    }
    
    func testSigmoidOp() {
        let input = Symbol<S>(uniform(Extent(5)))
        let target = Symbol<S>(uniform(Extent(5)))
        
        let sigmoid = Sigmoid<S>(size: 5)
        sigmoid.setInput("input", to: input)
        
        let loss = L2Loss(target: target)
        loss.setInput("input", to: sigmoid)

        let lossGrad = loss.gradient()
        let sigmoidGrad = sigmoid.gradient()
        sigmoidGrad.setInput("gradOutput", to: lossGrad)
        
        
        let eps = 10e-6
        let result = checkGradient(params: input.output!,
                                   gradParams: (sigmoidGrad as! Op<S>).output!,
                                   eps: eps)
        {
            sigmoidGrad.reset()
            lossGrad.reset()
            
            sigmoid.apply()
            loss.apply()
            
            lossGrad.apply()
            sigmoidGrad.apply()
            
            return loss.value
        }
        
        XCTAssert(isClose(result, zeros(Extent(result.shape)), eps: eps))
    }

    func testLogOp() {
        let input = Symbol<S>(uniform(Extent(5)))
        let target = Symbol<S>(uniform(Extent(5)))
        
        let log = Log<S>(size: 5)
        log.setInput("input", to: input)
        
        let loss = L2Loss(target: target)
        loss.setInput("input", to: log)

        let lossGrad = loss.gradient()
        let logGrad = log.gradient()
        logGrad.setInput("gradOutput", to: lossGrad)
        
        let eps = 10e-6
        let result = checkGradient(params: input.output!,
                                   gradParams: (logGrad as! Op<S>).output!,
                                   eps: eps)
        {
            logGrad.reset()
            lossGrad.reset()
            
            log.apply()
            loss.apply()
            
            lossGrad.apply()
            logGrad.apply()
            
            return loss.value
        }
        
        XCTAssert(isClose(result, zeros(Extent(result.shape)), eps: eps))
    }

    func testSequenceOp() {
        let input = Symbol<S>(uniform(Extent(10)))
        let target = Symbol<S>(uniform(Extent(5)))
        
        let seq = Sequence<S>(
            input,
            Linear<S>(inputSize: 10, outputSize: 5),
            Sigmoid<S>(size: 5),
            L2Loss<S>(target: target)
        )
        
        let seqGrad = seq.gradient() as! SequenceGradient<S>
        
        let eps = 10e-6
        let result = checkGradient(params: input.output!,
                                   gradParams: seqGrad.output!,
                                   eps: eps)
        {
            seqGrad.reset()
            seq.apply()
            seqGrad.apply()
            
            let loss = seq.ops.last! as! L2Loss<S>
            return loss.value
        }
        
        XCTAssert(isClose(result, zeros(Extent(result.shape)), eps: eps))
    }    
}
