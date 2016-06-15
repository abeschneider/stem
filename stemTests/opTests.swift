//
//  opTests.swift
//  stem
//
//  Created by Schneider, Abraham R. on 5/25/16.
//  Copyright © 2016 none. All rights reserved.
//

import XCTest
import stem

// NB: For some reason unit tests don't see the previous version, so redeclared
// for testing purposes
public func copyOp<T:protocol<OpType, Copyable>>(op:T, shared:Bool) -> T {
    //    return op.dynamicType.init(op: op as! Op<T.StorageType>, shared: shared)
    return copy(op, shared: shared)
}

class opTests: XCTestCase {
    typealias D = NativeStorage<Double>

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testLinearForwardVector() {
        let w = Tensor<D>([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
        let b:Tensor<D> = zeros(Extent(4))
        
        let input = Symbol<D>(Tensor<D>([1, 2, 3]))
        let linear = Linear<D>(weight: w, bias: b)
        linear.setInput("input", to: input)
        linear.apply()
        
        let expected = Tensor<D>([1, 2, 3, 0])
        XCTAssert(isClose(linear.output, expected, eps: 10e-4), "Not close")
    }
    
    func testLinearForwardMatrix() {
        let w = Tensor<D>([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
        let b:Tensor<D> = zeros(Extent(4))
        
        let input = Symbol<D>(Tensor<D>([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        let linear = Linear<D>(weight: w, bias: b)
        linear.setInput("input", to: input)
        linear.apply()
        
        let expected = Tensor<D>([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]])
        XCTAssert(isClose(linear.output, expected, eps: 10e-4), "Not close")
    }
    
    func testLinearOpCopy() {
        let linear = Linear<D>(inputSize: 10, outputSize: 5)
        let linear2 = copyOp(linear, shared: true)
        let linear3 = copyOp(linear, shared: false)

        let w = ravel(linear.weight)
        let w2 = ravel(linear2.weight)
        let w3 = ravel(linear3.weight)
        for i in 0..<w.shape.elements {
            w[i] = 0
            XCTAssertEqualWithAccuracy(w2[i], 0, accuracy: 1e-6)
            XCTAssertNotEqualWithAccuracy(w3[i], 0, 1e-6)
        }
        
        let b = ravel(linear.bias)
        let b2 = ravel(linear2.bias)
        let b3 = ravel(linear3.bias)
        for i in 0..<b.shape.elements {
            b[i] = 1
            XCTAssertEqualWithAccuracy(b2[i], 1, accuracy: 1e-6)
            XCTAssertNotEqualWithAccuracy(b3[i], 1, 1e-6)
        }
    }
    
    func testLinearOpGradient() {
        let eps:Double = 10e-8
        let input = Symbol<D>(uniform(Extent(10)))
        
        let gradOutput = Symbol<D>(zeros(Extent(5)))
        
        let linear = Linear<D>(outputSize: 5)
        linear.bias.uniform()
        linear.setInput("input", to: input)

        let linearGrad = linear.gradient() as! LinearGrad<D>
        linearGrad.setInput("gradOutput", to: gradOutput)
//        print(linearGrad.inputs)
        
        // test gradient wrt the input
        let inputError = checkGradient(linear, grad: linearGrad, params: input.output, gradParams: linearGrad.output, eps: eps)
        XCTAssertLessThan(inputError, eps)
        
        // test gradient wrt the parameters
        let weightError = checkGradient(linear, grad: linearGrad, params: linear.weight, gradParams: linearGrad.weight, eps: eps)
        XCTAssertLessThan(weightError, eps)

        let biasError = checkGradient(linear, grad: linearGrad, params: linear.bias, gradParams: linearGrad.bias, eps: eps)
        XCTAssertLessThan(biasError, eps)
    }
    
    func testLinearOpGradient2() {
        let eps:Double = 10e-6
        let input = Symbol<D>(uniform(Extent(10, 3)))
        
        let gradOutput = Symbol<D>(zeros(Extent(5, 3)))
        
        let linear = Linear<D>(outputSize: 5)
        linear.bias.uniform()
        linear.setInput("input", to: input)

        let linearGrad = linear.gradient() as! LinearGrad<D>
        linearGrad.setInput("gradOutput", to: gradOutput)
        
        // test gradient wrt the input
        let inputError = checkGradient(linear, grad: linearGrad, params: input.output, gradParams: linearGrad.output, eps: eps)
        XCTAssertLessThan(inputError, eps)
        
        // test gradient wrt to the parameters
        let weightError = checkGradient(linear, grad: linearGrad, params: linear.weight, gradParams: linearGrad.weight, eps: eps)
        XCTAssertLessThan(weightError, eps)
        
        let biasError = checkGradient(linear, grad: linearGrad, params: linear.bias, gradParams: linearGrad.bias, eps: eps)
        XCTAssertLessThan(biasError, eps)
    }
    
    func testSigmoidGradient() {
        let eps:Double = 10e-6
        let input = Symbol<D>(uniform(Extent(10)))
        let gradOutput = Symbol<D>(zeros(Extent(10)))
        
        let sigmoid = Sigmoid<D>()
        sigmoid.setInput("input", to: input)
        
        let sigmoidGrad = sigmoid.gradient() as! SigmoidGrad<D>
        sigmoidGrad.setInput("gradOutput", to: gradOutput)
        
        // test gradient wrt the input
        let inputError = checkGradient(sigmoid, grad: sigmoidGrad, params: input.output, gradParams: sigmoidGrad.output, eps: eps)
        XCTAssertLessThan(inputError, eps)
    }
    
    func testLossGradient() {
        let eps:Double = 10e-6
        let input = Symbol<D>(uniform(Extent(10)))
        let target = Symbol<D>(uniform(Extent(10)))
        
        let loss = L2Loss<D>(target: target)
        loss.setInput("input", to: input)
        
        let lossGrad = loss.gradient() as! L2LossGrad<D>

        // test gradient wrt to the input
        let inputError = checkGradient(loss, grad: lossGrad, input: input.output, eps: eps)
        XCTAssertLessThan(inputError, eps)
    }
    
    func testSequenceConnected() {
        let eps = 10e-6
        
        let input = Symbol<D>(uniform(Extent(5)))
        let gradOutput = Symbol<D>(uniform(Extent(5)))
        let seq = Sequence<D>(
            input,
            Linear<D>(outputSize: 5),
            Sigmoid<D>()
        )
        
        let seq2 = Sequence<D>(
            Linear<D>(outputSize: 5),
            Sigmoid<D>()
        )
        
        let seq3 = Sequence<D>(
            seq,
            seq2
        )
        
        let seqGrad3 = seq3.gradient() as! SequenceGradient<D>
        seqGrad3.setInput("gradOutput", to: gradOutput)
        
        let inputError = checkGradient(seq3, grad: seqGrad3, params: input.output, gradParams: seqGrad3.output, eps: eps)
        XCTAssertLessThan(inputError, eps)
    }
    
    func testSequenceOpCopy() {
        let seq = Sequence<D>(
            Symbol<D>(Extent(10)),
            Linear<D>(outputSize: 5),
            Sigmoid<D>()
        )
  
        let seq2 = copyOp(seq, shared: false)
        seq2.apply()
    }
    
    func testSequenceGradient() {
        let eps:Double = 10e-6
        let input = Symbol<D>(uniform(Extent(10)))
        let gradOutput = Symbol<D>(zeros(Extent(5)))


        let seq = Sequence<D>(
            input,
            Linear<D>(outputSize: 5),
            Sigmoid<D>()
        )

        let seqGrad = seq.gradient() as! SequenceGradient<D>
        seqGrad.setInput("gradOutput", to: gradOutput)
        
        // test gradient wrt to the input
        let inputError = checkGradient(seq, grad: seqGrad, params: input.output, gradParams: seqGrad.output, eps: eps)
        XCTAssertLessThan(inputError, eps)
    }
    
    func testLogOpGradient() {
        let eps = 10e-6
        let input = Symbol<D>(uniform(Extent(10)))
        let gradOutput = Symbol<D>(zeros(Extent(10)))
        
        let log = Log<D>()
        log.setInput("input", to: input)
        

        let logGrad = log.gradient() as! LogGrad<D>
        logGrad.setInput("gradOutput", to: gradOutput)
        
        // test gradient wrt to the input
        let inputError = checkGradient(log, grad: logGrad, params: input.output, gradParams: logGrad.output, eps: eps)
        print(inputError)
        XCTAssertLessThan(inputError, eps)
    }
    
    func testUnrollGradient() {
        let eps = 10e-6
        let layer = Sequence<D>(Linear<D>(outputSize: 5), Sigmoid<D>())
        
        let input = Symbol<D>(uniform(Extent(5)))
        let gradOutput = Symbol<D>(zeros(Extent(5)))
        
        let seq = unroll([layer.ops[0], layer.ops[1]], count: 3)
        seq.setInput("input", to: input)
        
        let seqGrad = seq.gradient() as! SequenceGradient<D>
        seqGrad.setInput("gradOutput", to: gradOutput)

        // test gradient wrt to the input
        let inputError = checkGradient(seq, grad: seqGrad, params: input.output, gradParams: seqGrad.output, eps: eps)
        XCTAssertLessThan(inputError, eps)
    }
}
