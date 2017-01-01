//
//  opTests.swift
//  stem
//
//  Created by Schneider, Abraham R. on 5/25/16.
//  Copyright © 2016 none. All rights reserved.
//

import XCTest
@testable import Tensor
@testable import Op

class opTests: XCTestCase {
    typealias D = NativeStorage<Double>
    typealias I = NativeStorage<Int>
    
    // Why is this required to be redefined?
    func copy<T:OpType & Copyable>(op:T, shared:Bool) -> T {
        return type(of: op).init(op: op as! Op<T.StorageType>, shared: shared)
    }

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
        
        let input = ConstantOp<D>(Tensor<D>([1, 2, 3]))
        let linear = LinearOp<D>(weight: w, bias: b)
        connect(from: input, to: linear)
        linear.apply()
        
        let expected = Tensor<D>([1, 2, 3, 0])
        XCTAssert(isClose(linear.output, expected, eps: 10e-4), "Not close")
    }
    
    func testLinearForwardMatrix() {
        let w = Tensor<D>([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
        let b:Tensor<D> = zeros(Extent(4))
        
        let input = ConstantOp<D>(Tensor<D>([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        let linear = LinearOp<D>(weight: w, bias: b)
        connect(from: input, to: linear)
        linear.apply()
        
        let expected = Tensor<D>([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]])
        XCTAssert(isClose(linear.output, expected, eps: 10e-4), "Not close")
    }
    
//    func testLinearOpCopy() {
//        let linear = LinearOp<D>(inputSize: 10, outputSize: 5)
//        let linear2 = copy(op: linear, shared: true)
//        let linear3 = copy(op: linear, shared: false)
//
//        let w = ravel(linear.weight)
//        let w2 = ravel(linear2.weight)
//        let w3 = ravel(linear3.weight)
//        for i in 0..<w.shape.elements {
//            w[i] = 0
//            XCTAssertEqualWithAccuracy(w2[i], 0, accuracy: 1e-6)
//            XCTAssertNotEqualWithAccuracy(w3[i], 0, 1e-6)
//        }
//        
//        let b = ravel(linear.bias)
//        let b2 = ravel(linear2.bias)
//        let b3 = ravel(linear3.bias)
//        for i in 0..<b.shape.elements {
//            b[i] = 1
//            XCTAssertEqualWithAccuracy(b2[i], 1, accuracy: 1e-6)
//            XCTAssertNotEqualWithAccuracy(b3[i], 1, 1e-6)
//        }
//    }
    
    func testLinearOpGradient() {
        let eps:Double = 10e-6
        let input = ConstantOp<D>(uniform(Extent(10)))
        
        let gradOutput = ConstantOp<D>(zeros(Extent(5)))
        
        let linear = LinearOp<D>(outputSize: 5)
        linear.bias.uniform()
        connect(from: input, to: linear)

        let linearGrad = linear.gradient() as! LinearGrad<D>
        connect(from: gradOutput, to: linearGrad, "gradOutput")
        
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
        let input = ConstantOp<D>(uniform(Extent(10, 3)))
        
        let gradOutput = ConstantOp<D>(zeros(Extent(5, 3)))
        
        let linear = LinearOp<D>(outputSize: 5)
        linear.bias.uniform()
        connect(from: input, to: linear)

        let linearGrad = linear.gradient() as! LinearGrad<D>
        connect(from: gradOutput, to: linearGrad, "gradOutput")
        
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
        let input = ConstantOp<D>(uniform(Extent(10)))
        let gradOutput = ConstantOp<D>(zeros(Extent(10)))
        
        let sigmoid = SigmoidOp<D>()
        connect(from: input, to: sigmoid)
        
        let sigmoidGrad = sigmoid.gradient() as! SigmoidGrad<D>
        connect(from: gradOutput, to: sigmoidGrad, "gradOutput")
        
        // test gradient wrt the input
        let inputError = checkGradient(sigmoid, grad: sigmoidGrad, params: input.output, gradParams: sigmoidGrad.output, eps: eps)
        XCTAssertLessThan(inputError, eps)
    }
    
    func testTanhGradient() {
        let eps:Double = 10e-8
        let input = ConstantOp<D>(uniform(Extent(10)))
        let gradOutput = ConstantOp<D>(zeros(Extent(10)))
        
        let tanh = TanhOp<D>()
        connect(from: input, to: tanh)
        
        let tanhGrad = tanh.gradient() as! TanhGrad<D>
        connect(from: gradOutput, to: tanhGrad, "gradOutput")
        
        calcForwardGrad(tanh, params: input.output, eps: eps)
        
        // test gradient wrt the input
        let inputError = checkGradient(tanh, grad: tanhGrad, params: input.output, gradParams: tanhGrad.output, eps: eps)
        XCTAssertLessThan(inputError, eps)
    }
    
    func testLossGradient() {
        let eps:Double = 10e-6
        let input = ConstantOp<D>(uniform(Extent(10)))
        let target = ConstantOp<D>(uniform(Extent(10)))
        
        let loss = L2Loss<D>()
        connect(from: input, to: loss, "input")
        connect(from: target, to: loss, "target")
        
        let lossGrad = loss.gradient() as! L2LossGrad<D>
        connect(from: loss, to: lossGrad, "op")
        connect(from: input, to: lossGrad, "input")
        connect(from: target, to: lossGrad, "target")

        // test gradient wrt to the input
        let inputError = checkGradient(loss, grad: lossGrad, input: input.output, eps: 10e-4)

        XCTAssertLessThan(inputError, eps)
    }
    
    func testConcatOp() {
        let inputValues:[Tensor<D>] = [uniform(Extent(5)), uniform(Extent(5)), uniform(Extent(5))]
        let inputs = inputValues.map { ConstantOp<D>($0) }
        let concatOp = ConcatOp<D>(inputs)
        
        concatOp.apply()
        let expected = concat(inputValues)

        XCTAssert(isClose(concatOp.output, expected, eps: 10e-3))
    }
    
    func testConcatGradient() {
        let eps = 10e-6
        
        // make checking gradient easier by storing all inputs
        // as a single parameter
        let input:Tensor<D> = uniform(Extent(15))
        let inputs = [ConstantOp<D>(input[0..<5]), ConstantOp<D>(input[5..<10]), ConstantOp<D>(input[10..<15])]
        let gradOutput = ConstantOp<D>(zeros(Extent(15)))
        
        let concat = ConcatOp<D>()
        connect(from: inputs, to: concat)
        
        let concatGrad = concat.gradient() as! ConcatGrad<D>
        connect(from: gradOutput, to: concatGrad, "gradOutput")
        
        // make checking gradient easier by storing all gradient output
        // as a single parameter
        let output:Tensor<D> = zeros(Extent(15))
        concatGrad.outputs["output"] = [output[0..<5], output[5..<10], output[10..<15]]

        // test gradient wrt to the input
        let inputError = checkGradient(concat, grad: concatGrad, params: input, gradParams: output, eps: eps)
        XCTAssertLessThan(inputError, eps)
    }
    
    func testViewOp() {
        let input = ConstantOp<D>(Extent(5, 5))
        let view = ViewOp(input: input, ranges: [1..<4, 1..<4])
        fill(view.output, value: 1)
        
        view.apply()
        
        let expected = Tensor<D>([
            [0.0,	0.0,	0.0,	0.0,	0.0],
            [0.0,	1.0,	1.0,	1.0,	0.0],
            [0.0,	1.0,	1.0,	1.0,	0.0],
            [0.0,	1.0,	1.0,	1.0,	0.0],
            [0.0,	0.0,	0.0,	0.0,	0.0]])
        
        XCTAssert(isClose(input.output, expected, eps: 10e-6))
    }
    
    func testViewGradient() {
        let eps = 10e-6

        let inputValue:Tensor<D> = uniform(Extent(5, 5))
        let input = ConstantOp<D>(inputValue)
        let gradOutput = ConstantOp<D>(zeros(Extent(3, 2)))
        
        
        let view = ViewOp<D>(input: input, ranges: [1..<4, 2..<4])
        let inputView = ravel(inputValue[1..<4, 2..<4])
        
        let viewGrad = view.gradient() as! ViewGrad<D>
        connect(from: gradOutput, to: viewGrad, "gradOutput")
        
        // test gradient wrt to the input
        let inputError = checkGradient(view, grad: viewGrad, params: inputView, gradParams: viewGrad.output, eps: eps)

        XCTAssertLessThan(inputError, eps)
    }
    
    func testAddOp() {
        let v1 = ConstantOp<D>(2*ones(Extent(5)))
        let v2 = ConstantOp<D>(3*ones(Extent(5)))
        let v3 = ConstantOp<D>(5*ones(Extent(5)))

        let addOp = AddOp<D>(v1, v2, v3)
        addOp.apply()
        
        let expected = Tensor<D>([10, 10, 10, 10, 10])
        XCTAssert(isClose(addOp.output, expected, eps:10e-4))
    }
    
    func testAddOpGradient() {
        let eps = 10e-6
        let values:Tensor<D> = Tensor<D>([5, 5, 5, 5, 5, 2, 2, 2, 2, 2])
        
        let v1 = ConstantOp<D>(values[0..<5])
        let v2 = ConstantOp<D>(values[5..<10])

        let addOp = AddOp<D>(v1, v2)
        
        let addOpGrad = addOp.gradient() as! AddOpGrad<D>
        
        let gradOutput = ConstantOp<D>(zeros(Extent(5)))
        connect(from: gradOutput, to: addOpGrad, "gradOutput")
        
        let gradInput:Tensor<D> = zeros(Extent(10))
        addOpGrad.outputs["output"] = [gradInput[0..<5], gradInput[5..<10]]

        let inputError = checkGradient(addOp, grad: addOpGrad, params: values, gradParams: gradInput, eps: eps)
        XCTAssertLessThan(inputError, eps)
    }
    
    func testMulOp() {
        let v1 = ConstantOp<D>(2*ones(Extent(5)))
        let v2 = ConstantOp<D>(3*ones(Extent(5)))
        let v3 = ConstantOp<D>(5*ones(Extent(5)))
        
        let mulOp = MulOp<D>(v1, v2, v3)
        mulOp.apply()
        
        let expected = Tensor<D>([30, 30, 30, 30, 30])
        XCTAssert(isClose(mulOp.output, expected, eps:10e-4))
    }
    
    func testMulOpGradient() {
        let eps = 10e-6
        let values:Tensor<D> = Tensor<D>([5, 5, 5, 5, 5, 2, 2, 2, 2, 2])
        
        let v1 = ConstantOp<D>(values[0..<5])
        let v2 = ConstantOp<D>(values[5..<10])
        
        let mulOp = MulOp<D>(v1, v2)
        
        let mulOpGrad = mulOp.gradient() as! MulOpGrad<D>
        
        let gradOutput = ConstantOp<D>(zeros(Extent(5)))
        connect(from: gradOutput, to: mulOpGrad, "gradOutput")
        
        let gradInput:Tensor<D> = zeros(Extent(10))
        mulOpGrad.outputs["output"] = [gradInput[0..<5], gradInput[5..<10]]
        
        let inputError = checkGradient(mulOp, grad: mulOpGrad, params: values, gradParams: gradInput, eps: eps)
        XCTAssertLessThan(inputError, eps)
    }
    
    func testConvOp1() {
        let input = ConstantOp(Tensor<D>([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        
        let convOp = Conv2dOp<D>(input: input, numFilters: 1, filterSize: Extent(3, 3))
        convOp.kernels[0, all, all] = Tensor<D>([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        convOp.apply()
        
        let expected = Tensor<D>([[-13, -20, -17], [-18, -24, -18], [13, 20, 17]])
        XCTAssert(isClose(convOp.output, expected, eps:10e-4))
    }
    
    func testConvOp2() {
        let input = ConstantOp(Tensor<D>([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        
        let convOp = Conv2dOp<D>(input: input, numFilters: 2, filterSize: Extent(3, 3))
        convOp.kernels[0, all, all] = Tensor<D>([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        convOp.kernels[1, all, all] = Tensor<D>([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        convOp.apply()
        
        let expected = 2*Tensor<D>([[-13, -20, -17], [-18, -24, -18], [13, 20, 17]])
        XCTAssert(isClose(convOp.output, expected, eps:10e-4))
    }
    
    // test basic functionality
    func testConvOpGradient1() {
        let eps = 10e-6
        let values = Tensor<D>([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        let input = ConstantOp(values)
        
        let convOp = Conv2dOp<D>(input: input, numFilters: 1, filterSize: Extent(3, 3))
        connect(from: input, to: convOp)
        
        let convOpGrad = convOp.gradient() as! Conv2dGrad<D>
        
        let gradOutput = ConstantOp<D>(zeros(Extent(3, 3)))
        connect(from: gradOutput, to: convOpGrad, "gradOutput")
        
        let inputError = checkGradient(convOp, grad: convOpGrad, params: input.output, gradParams: convOpGrad.output, eps: eps)
        XCTAssertLessThan(inputError, eps)
    }
    
    // test with more than one filter
    func testConvOpGradient2() {
        let eps = 10e-6
        let values = Tensor<D>([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        let input = ConstantOp(values)
        
        let convOp = Conv2dOp<D>(input: input, numFilters: 2, filterSize: Extent(3, 3))
        connect(from: input, to: convOp)

        let convOpGrad = convOp.gradient() as! Conv2dGrad<D>

        let gradOutput = ConstantOp<D>(zeros(Extent(3, 3)))
        connect(from: gradOutput, to: convOpGrad, "gradOutput")
        
        let inputError = checkGradient(convOp, grad: convOpGrad, params: values, gradParams: convOpGrad.output, eps: eps)
        XCTAssertLessThan(inputError, eps)
    }
    
    // test with an image size larger than the kernel
    func testConvOpGradient3() {
        let eps = 10e-6
        let values = Tensor<D>([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]])
        let input = ConstantOp(values)
        
        let convOp = Conv2dOp<D>(input: input, numFilters: 1, filterSize: Extent(3, 3))
        connect(from: input, to: convOp)
        
        let convOpGrad = convOp.gradient() as! Conv2dGrad<D>
        
        let gradOutput = ConstantOp<D>(zeros(Extent(5, 5)))
        connect(from: gradOutput, to: convOpGrad, "gradOutput")
        
        // test gradient wrt the input
        let inputError = checkGradient(convOp, grad: convOpGrad, params: values, gradParams: convOpGrad.output, eps: eps)
        XCTAssertLessThan(inputError, eps)
        
        // test gradient wrt the parameters
        let kernelError = checkGradient(convOp, grad: convOpGrad, params: convOp.kernels[0, all, all], gradParams: convOpGrad.kernels[0, all, all], eps: eps)
        XCTAssertLessThan(kernelError, eps)
//
//        // test gradient wrt the parameters
//        let weightError = checkGradient(linear, grad: linearGrad, params: linear.weight, gradParams: linearGrad.weight, eps: eps)
//        XCTAssertLessThan(weightError, eps)
    }
    
    func testMaxPoolingOp() {
        let poolingOp = PoolingOp<D>(poolingSize: Extent(2, 2), stride: Extent(2, 2), evalFn: max)
        
        let data:Tensor<D> = zeros(Extent(10, 10))
        for i in 0..<10 {
            for j in 0..<10 {
                data[i, j] = Double(i)*Double(j)
            }
        }
        
        let input = ConstantOp(data)
        connect(from: input, to: poolingOp)
        poolingOp.apply()
        
        let expected = Tensor<D>([[1, 3, 5, 7, 9], [3, 9, 15, 21, 27], [5, 15, 25, 35, 45], [7, 21, 35, 49, 63], [9, 27, 45, 63, 81]])
        XCTAssert(isClose(poolingOp._output, expected, eps:10e-4))
    }
    
    func testMaxPoolingGrad() {
        let eps = 10e-6
        let poolingOp = PoolingOp<D>(poolingSize: Extent(2, 2), stride: Extent(2, 2), evalFn: max)
        
        let data:Tensor<D> = zeros(Extent(10, 10))
        for i in 0..<10 {
            for j in 0..<10 {
                data[i, j] = Double(i)*Double(j)
            }
        }
        
        let input = ConstantOp(data)
        connect(from: input, to: poolingOp)
        
        XCTAssertEqual(poolingOp.output.shape, Extent(1, 5, 5))

        let poolingGrad = poolingOp.gradient() as! PoolingGrad<D>
        
        let gradOutput = ConstantOp<D>(zeros(Extent(5, 5)))
        connect(from: gradOutput, to: poolingGrad, "gradOutput")
        
        let inputError = checkGradient(poolingOp, grad: poolingGrad, params: data, gradParams: poolingGrad.output, eps: eps)
        XCTAssertLessThan(inputError, eps)
    }
    
    func testCollection() {
        let eps = 10e-6
        let input = ConstantOp<D>(uniform(Extent(5)))
        let gradOutput = ConstantOp<D>(zeros(Extent(5)))

        let linear = LinearOp<D>(outputSize: 5)
        let sigmoid = SigmoidOp<D>()
        
        connect(from: input, to: linear)
        connect(from: linear, to: sigmoid)
        
        let c = CollectionOp<D>(ops: [input, linear, sigmoid],
                                inputs: [input],
                                outputs: [sigmoid],
                                ordering: SequentialOrdering<D>())
        
        let cgrad = c.gradient() as! CollectionGradient<D>
        connect(from: gradOutput, to: cgrad, "gradOutput")
        
        // test gradient wrt to the input
        let inputError = checkGradient(c, grad: cgrad, params: input.output, gradParams: cgrad.output, eps: eps)
        XCTAssertLessThan(inputError, eps)
    }
    
    func testSequentialOp() {
        let eps = 10e-6
        let input = ConstantOp<D>(uniform(Extent(5)))
        let gradOutput = ConstantOp<D>(zeros(Extent(5)))
        
        let linear = LinearOp<D>(outputSize: 5)
        let sigmoid = SigmoidOp<D>()
        
        let c = SequentialOp<D>(input, linear, sigmoid)
        
        let cgrad = c.gradient() as! CollectionGradient<D>
        connect(from: gradOutput, to: cgrad, "gradOutput")
        
        // test gradient wrt to the input
        let inputError = checkGradient(c, grad: cgrad, params: input.output, gradParams: cgrad.output, eps: eps)
        XCTAssertLessThan(inputError, eps)
    }
    
    func testSequentialOp2() {
        let eps = 10e-6
        let input = ConstantOp<D>(uniform(Extent(5)))
        let gradOutput = ConstantOp<D>(zeros(Extent(5)))
        
        let linear = LinearOp<D>(outputSize: 5)
        let linear2 = LinearOp<D>(outputSize: 5)
        let sigmoid = SigmoidOp<D>()
        let sigmoid2 = SigmoidOp<D>()
        
        let c = SequentialOp<D>(input, linear, sigmoid)
        let c2 = SequentialOp<D>(linear2, sigmoid2)
        let c3 = SequentialOp<D>(c, c2)
        
        let c3grad = c3.gradient() as! CollectionGradient<D>
        connect(from: gradOutput, to: c3grad, "gradOutput")
        
        // test gradient wrt to the input
        let inputError = checkGradient(c3, grad: c3grad, params: input.output, gradParams: c3grad.output, eps: eps)
        XCTAssertLessThan(inputError, eps)
    }
    
    func testRNN1() {
        let eps = 10e-6
        
        let size = 10
        let input = ConstantOp<D>(uniform(Extent(size)))
        let gradOutput = ConstantOp<D>(zeros(Extent(size)))
        
        let concat = ConcatOp<D>()
        let linear = LinearOp<D>(outputSize: size)
        let sigmoid = SigmoidOp<D>(size: size)
        let prevOutput = ConstantOp<D>(sigmoid.output)

        connect(from: [input, prevOutput], to: concat)
        connect(from: concat, to: linear)
        connect(from: linear, to: sigmoid)
        
        let rnn = CollectionOp<D>(ops: [input, prevOutput, concat, linear, sigmoid],
                                  inputs: [],
                                  outputs: [sigmoid],
                                  ordering: SequentialOrdering())
        
        // manually build backwards graph
        let concatGrad = concat.gradient() as! ConcatGrad<D>
        let linearGrad = linear.gradient() as! LinearGrad<D>
        let sigmoidGrad = sigmoid.gradient() as! SigmoidGrad<D>
        
        connect(from: sigmoidGrad, to: linearGrad, "gradOutput")
        connect(from: linearGrad, to: concatGrad, "gradOutput")
        
        let rnnGrad = CollectionGradient<D>(ops: [sigmoidGrad, linearGrad, concatGrad],
                                            inputs: [sigmoidGrad],
                                            outputs: [concatGrad],
                                            ordering: SequentialOrdering<D>())
        
        connect(from: gradOutput, to: rnnGrad, "gradOutput")

        let inputError = checkGradient(rnn, grad: rnnGrad, params: input.output, gradParams: rnnGrad.output, eps: eps)
        XCTAssertLessThan(inputError, eps)
    }
    
    func testRNN2() {
        let eps = 10e-6
        
        let size = 10
        let input = ConstantOp<D>(uniform(Extent(size)))
        let gradOutput = ConstantOp<D>(zeros(Extent(size)))
        
        let concat = ConcatOp<D>()
        let linear = LinearOp<D>(outputSize: size)
        let sigmoid = SigmoidOp<D>(size: size)
        let prevOutput = ConstantOp<D>(sigmoid.output)
        
        connect(from: [input, prevOutput], to: concat)
        connect(from: concat, to: linear)
        connect(from: linear, to: sigmoid)
        
        let rnn = CollectionOp<D>(ops: [input, prevOutput, concat, linear, sigmoid],
                                  inputs: [],
                                  outputs: [sigmoid],
                                  ordering: SequentialOrdering())
        
        let rnnGrad = rnn.gradient() as! CollectionGradient<D>
        connect(from: gradOutput, to: rnnGrad, "gradOutput")
        
        let inputError = checkGradient(rnn, grad: rnnGrad, params: input.output, gradParams: rnnGrad.output, eps: eps)
        XCTAssertLessThan(inputError, eps)
    }
    
    // using Identity like Torch
    /*func testRNNAlt() {
        let eps = 10e-6
        
        // input, h_{t-1}
        let size = 5
        let input = IdentityOp<D>()
        let prevOutput = IdentityOp<D>()
        let concat = ConcatOp<D>([input, prevOutput])
        let linear = LinearOp<D>(outputSize: size)
        let sigmoid = SigmoidOp<D>(size: size)
        
        connect(from: concat, to: linear)
        connect(from: linear, to: sigmoid)
        connect(from: sigmoid, to: prevOutput)
    }*/
    
    
/*
forward:
     [input[0], 0s] -> concat() -> linear() -> sigmoid() -> prevOutput
     [input[1], prevOutput] -> concat() -> linear() -> sigmoid() -> prevOutput
     ...
     
backward:
     error -> sigmoidGrad() -> linearGrad() -> concatGrad() -> [gradInput[n], gradPrevOutput]
     gradPrevOutput -> sigmoidGrad() -> linearGrad() -> concatGrad() -> [gradInput[n-1], gradPrevOutput]
     ...
     
     
-- unrolled version
     
forward:
     sigmoid(linear(concat(input[1], sigmoid(linear(concat([input[0], 0s]))))))
     c0 = concat(input[0], 0s)
     l0 = linear(c0)
     s0 = sigmoid(l0)
     c1 = concat(input[1], s0)
     l1 = linear(c1)
     s1 = sigmoid(l1)
     
backward:
     sg1 = sigmoidGrad(gradInput)
     lg1 = linearGrad(sg1)
     input[1], cg1 = concatGrad(lg1)
     sg0 = sigmoidGrad(cg1)
     lg0 = linearGrad(sg0)
     input[0], cg0 = concatGrad(lg0)
*/
    func testRNN3() {
        let eps = 10e-3
        
        let size = 5
        let input = ConstantOp<D>(uniform(Extent(size)))
//        let gradOutput = ConstantOp<D>(zeros(Extent(size)))
//        let gradOutput = ConstantOp<D>(Tensor<D>([0.5, 0.5, 0.5, 0.5, 0.5]))
        
        let concat = ConcatOp<D>()
        let linear = LinearOp<D>(inputSize: 2*size, outputSize: size)
        let sigmoid = SigmoidOp<D>(size: size)
        
        // is a Variable in order to allow connections to have an effect
        let prevOutput = VariableOp<D>(Extent(size))
//        let prevOutput = ConstantOp<D>(zeros(Extent(size)))
        
        connect(from: [input, prevOutput], to: concat)
        connect(from: concat, to: linear)
        connect(from: linear, to: sigmoid)
        connect(from: sigmoid, to: prevOutput)
        
        let rnn = CollectionOp<D>(ops: [input, prevOutput, concat, linear, sigmoid],
                                  inputs: [],
                                  outputs: [sigmoid],
                                  ordering: RepeatedSequentialOrdering<D>(count: 2))
        
//        rnn.apply()
        
//        let rnnGrad = rnn.gradient() as! CollectionGradient<D>
        
        // manually build backwards graph
        //let prevOutputGrad = prevOutput.gradient() as! VariableGradient<D>
        // for now try not using Gradient version (should be the same?)
//        let prevOutputGrad = VariableOp<D>(Extent(size))
        let prevOutputGrad = prevOutput.gradient() as! VariableGrad<D>
        let concatGrad = concat.gradient() as! ConcatGrad<D>
        let linearGrad = linear.gradient() as! LinearGrad<D>
        let sigmoidGrad = sigmoid.gradient() as! SigmoidGrad<D>

        connect(from: prevOutputGrad, to: sigmoidGrad, "gradOutput")
        connect(from: sigmoidGrad, to: linearGrad, "gradOutput")
        connect(from: linearGrad, to: concatGrad, "gradOutput")
        
        // currently this has no effect (regardless of index or if this is commented out)
        connect(Source(op: concatGrad, label: "output", index: 0),
                Target(op: prevOutputGrad, label: "gradOutput"))
        
        // want:
        // collection[gradOutput] -> sigmoid[gradOutput]
        // ...
        //prevGradOutput -> sigmoid[gradOutput]

        // ops: [prevOutputGrad, sigmoidGrad, linearGrad, concatGrad],
        // outputs: [concatGrad],
        let gradOutput = ConstantOp<D>(Extent(size))
        let rnnGrad = CollectionGradient<D>(ops: [sigmoidGrad, linearGrad, concatGrad, prevOutputGrad],
                                            inputs: [],
                                            outputs: [concatGrad],
                                            ordering: RepeatedSequentialOrdering<D>(count: 2))
        
//        rnnGrad.apply()

////        connect(from: gradOutput, to: rnnGrad, "gradOutput")
////        print(rnnGrad)
//        
////        rnn.apply()        
//        let out:[Tensor<D>] = concatGrad.outputs["output"]!
//        fill(out[0], value: Double(0.5))
////        rnnGrad.apply()
//
        connect(from: gradOutput, to: rnnGrad, "gradOutput")
        let inputError = checkGradient(rnn, grad: rnnGrad, params: input.output, gradParams: gradOutput.output, eps: eps)
//        let weightError = checkGradient(rnn, grad: rnnGrad, params: linear.weight, gradParams: linearGrad.weight, eps: eps)
////        XCTAssertLessThan(inputError, eps)
    }
    
//    func testLSTM() {
//        let size = 10
//        // [(input (+) out-) -> T -> tanh, (input (+) out-) -> T -> sigmoid] -> mul
//        // [mul, (input (+) out)] -> T -> sigmoid] -> sum
//        // sum -> state -> out
//        
//        let input = ConstantOp<D>(zeros(Extent(size)))
//        let output = ConstantOp<D>(zeros(Extent(size)))
//        let prevOutput = ConstantOp<D>(zeros(Extent(size)))
//        
//        let inputTransform = Sequence<D>(
//            ConcatOp<D>(input, prevOutput),
//            LinearOp<D>(outputSize: size),
//            SigmoidOp<D>()
//        )
//        
//        let inputGate = Sequence<D>(
//            ConcatOp<D>(input, output),
//            LinearOp<D>(outputSize: size),
//            TanhOp<D>()
//        )
//        
//        let inputValue = MulOp(inputTransform, inputGate)
//        let cell = inputValue // + forgetValue
//        
//        // forward
//        let data:[Tensor<D>] = []
//        var outputs:[Tensor<D>] = []
//        for (t, x) in data.enumerate() {
//            input.set(x)
//            cell.apply()
//            outputs.append(cell.output)
//        }
//        
//        // backwards
//        let grad = unroll([cell], count: 5)
//        for out in outputs {
//            
//        }
////        copy(from: output.output, to: prevOutput.output)
//
////        let forgetGate = Sequence<D>(
////            ConcatOp<D>(input, output),
////            LinearOp<D>(outputSize: size),
////            SigmoidOp<D>()
////        )
//        
//        
//        // test gating for inputValue
//        
//        
////        let forgetValue = MulOp(state, forgetGate)
////        let stateInput = AddOp(inputValue, forgetValue)
//    }
    
    func testRNN4() {
        let eps = 10e-3
        
        let size = 5
        
        let initial = ConstantOp<D>(zeros(Extent(size)))
        let input = ConstantOp<D>(uniform(Extent(size)))
        let concat = ConcatOp<D>()
        let linear = LinearOp<D>(inputSize: 2*size, outputSize: size)
        let sigmoid = SigmoidOp<D>(size: size)
        
        // is a Variable in order to allow connections to have an effect
        let prevOutput = VariableOp<D>(Extent(size))
        //        let prevOutput = ConstantOp<D>(zeros(Extent(size)))
        
        connect(from: [input, prevOutput], to: concat)
        connect(from: concat, to: linear)
        connect(from: linear, to: sigmoid)
        connect(from: sigmoid, to: prevOutput)
        
        let rnn = RecurrentCollectionOp<D>(ops: [input, prevOutput, concat, linear, sigmoid],
                                  inputs: [initial],
                                  outputs: [sigmoid],
                                  recurrentVars: [prevOutput],
                                  ordering: RepeatedSequentialOrdering<D>(count: 2))
        
        
        let prevOutputGrad = prevOutput.gradient() as! VariableGrad<D>
        let concatGrad = concat.gradient() as! ConcatGrad<D>
        let linearGrad = linear.gradient() as! LinearGrad<D>
        let sigmoidGrad = sigmoid.gradient() as! SigmoidGrad<D>
        
        connect(from: prevOutputGrad, to: sigmoidGrad, "gradOutput")
        connect(from: sigmoidGrad, to: linearGrad, "gradOutput")
        connect(from: linearGrad, to: concatGrad, "gradOutput")
        
        // currently this has no effect (regardless of index or if this is commented out)
        connect(Source(op: concatGrad, label: "output", index: 0),
                Target(op: prevOutputGrad, label: "gradOutput"))

        let gradOutput = ConstantOp<D>(zeros(Extent(size)))
        let rnnGrad = RecurrentCollectionGrad<D>(ops: [sigmoidGrad, linearGrad, concatGrad, prevOutputGrad],
                                                 inputs: [],
                                                 outputs: [concatGrad],
                                                 recurrentVars: [prevOutputGrad],
                                                 ordering: RepeatedSequentialOrdering<D>(count: 2))
        
        connect(from: gradOutput, to: rnnGrad, "gradOutput")
        
//        rnn.apply()
//        rnnGrad.apply()
        let inputError = checkGradient(rnn, grad: rnnGrad, params: input.output, gradParams: gradOutput.output, eps: eps)
    }
}
