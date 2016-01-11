//
//  nnTests.swift
//  stem
//
//  Created by Abe Schneider on 12/3/15.
//  Copyright © 2015 none. All rights reserved.
//

import XCTest
@testable import stem

class nnTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testGraph() {
        typealias S = NativeStorage<Double>
//        let w = Matrix<NativeStorage<Double>>([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], copyTransposed: true)
        let vec = Vector<S>([1, 2, 3])
        let v = Variable<S>(variable: vec)
        let l = Linear<S>(numInputs: 3, numOutputs: 3)
        
//        try! connect(from: AnyGraphModule(v), to: AnyGraphModule(l))
        try! connect(from: v, to: l)
        print(v.toModule)
    }

    /*func testLinearForwardVector() {
        let w = Matrix<NativeStorage<Double>>([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], copyTransposed: true)
        XCTAssertEqual(w.shape, Extent(3, 4))
        
        let linear = LinearModule<NativeStorage<Double>>(weight: w)
        let input = ColumnVector<NativeStorage<Double>>([1, 2, 3])
        let output = linear.forward(input)

        let expected = Vector<NativeStorage<Double>>([1, 2, 3, 0])
        XCTAssert(isClose(output, expected, eps: 10e-4), "Not close")

        let w2 = Matrix<CBlasStorage<Double>>([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], copyTransposed: true)
        let linear2 = LinearModule<CBlasStorage<Double>>(weight: w2)
        let input2 = ColumnVector<CBlasStorage<Double>>([1, 2, 3])
        let output2 = linear2.forward(input2)
        
        let expected2 = Vector<CBlasStorage<Double>>([1, 2, 3, 0])
        XCTAssert(isClose(output2, expected2, eps: 10e-4), "Not close")
    }
    
    func testLinearForwardMatrix() {
        let w = Matrix<NativeStorage<Double>>([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], copyTransposed: true)
        let linear = LinearModule<NativeStorage<Double>>(weight: w)
        let input = Matrix<NativeStorage<Double>>([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        let output = linear.forward(input)
        
        let expected = Matrix<NativeStorage<Double>>([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]])
        XCTAssert(isClose(output, expected, eps: 10e-4), "Not close")
        
        let w2 = Matrix<CBlasStorage<Double>>([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], copyTransposed: true)
        let linear2 = LinearModule<CBlasStorage<Double>>(weight: w2)
        let input2 = Matrix<CBlasStorage<Double>>([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        let output2 = linear2.forward(input2)
        
        let expected2 = Matrix<CBlasStorage<Double>>([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]])
        XCTAssert(isClose(output2, expected2, eps: 10e-4), "Not close")
    }
    
    func testLinearBackward() {
        let w = Matrix<NativeStorage<Double>>([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], copyTransposed: true)
        let linear = LinearGradientModule<NativeStorage<Double>>(weight: w)
        let input = ColumnVector<NativeStorage<Double>>([1, 2, 3])
        
        linear.forward(input)
        let grad_output = ColumnVector<NativeStorage<Double>>([1, 2, 3, 4])
        let grad_input = linear.backward(input, gradOutput: grad_output)
        
        let expected = Vector<NativeStorage<Double>>([1, 2, 3])
        XCTAssert(isClose(grad_input, expected, eps: 10e-4), "Not close")

        let w2 = Matrix<CBlasStorage<Double>>([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], copyTransposed: true)
        let linear2 = LinearGradientModule<CBlasStorage<Double>>(weight: w2)
        let input2 = ColumnVector<CBlasStorage<Double>>([1, 2, 3])
        linear2.forward(input2)
        
        let grad_output2 = ColumnVector<CBlasStorage<Double>>([1, 2, 3, 4])
        let grad_input2 = linear2.backward(input2, gradOutput: grad_output2)
        
        let expected2 = Vector<CBlasStorage<Double>>([1, 2, 3])
        XCTAssert(isClose(grad_input2, expected2, eps: 10e-4), "Not close")
    }
    
    func testSharedStorage() {
        typealias S = NativeStorage<Double>
        
        let num_inputs = 10
        let num_outputs = 5
        
        // provides a flat view of all parameters to make gradient testing simple
        let storage = S(size: num_inputs*num_outputs + num_outputs)
        let weight = Matrix<S>(storage: storage,
                               shape: Extent(num_inputs, num_outputs),
                               offset: 0)
        
        let bias = RowVector<S>(storage: storage,
                                shape: Extent(num_outputs),
                                offset: num_inputs*num_outputs)
        
        // alter weight values only
        for i in weight.storageIndices() {
            weight.storage[i] = 1.0
        }
        
        // check bias was unaffected
        for i in bias.storageIndices() {
            XCTAssertEqual(bias.storage[i], 0.0)
        }
        
        // alter bias values only
        for i in bias.storageIndices() {
            bias.storage[i] = 2.0
        }
        
        // check weight was unaffected
        for i in weight.storageIndices() {
            XCTAssertEqual(weight.storage[i], 1.0)
        }
    }
    
    func testModule<M:Module, S:Storage where M:Gradient, M.StorageType==S, S.ElementType==Double>(
        rng:RandomNumberGenerator,
        module:M,
        storage:S,
        gradStorage:M.StorageType,
        numInputs:Int,
        numOutputs:Int
    )
    {
        let target = Vector<S>(rows: numOutputs)
        target.uniform(rng)
        
        let loss = L2Loss(target: target)
        
        let input = ColumnVector<S>(rows: numInputs)
        input.uniform(rng)
        
        let eps = 10e-6
        let result = checkGradient(input, params: storage, gradParams: gradStorage, eps: eps) {
            module.clear()
            
            // FIXME: For some reason the wrong forward/backward are getting select (Tensor).
            // This works, but is not the desired dispatch. If the code is moved into a single
            // test case, it works as expected. Possible compiler bug?
            
            // calculate error
            let output = ColumnVector<S>(module.forward($0))
            let error = loss.forward(output)
            
            // calculate gradient (for analytical gradient)
            let grad = loss.backward(output)
            module.backward($0, gradOutput: ColumnVector(grad))
            
            return error
        }
        
        for i in result.storageIndices() {
            XCTAssertLessThanOrEqual(result.storage[i], eps)
        }
    }
    
    func testNativeLinearGradient() {
        typealias S = NativeStorage<Double>
        
        let rng = RandomNumberGenerator()
        
        let num_inputs = 20
        let num_outputs = 10
        
        // provides a flat view of all parameters to make gradient testing simple
        let storage = S(size: num_inputs*num_outputs + num_outputs)
        let gradStorage = S(size: num_inputs*num_outputs + num_outputs)
        
        let weight = Matrix<S>(storage: storage,
                               shape: Extent(num_inputs, num_outputs),
                               offset: 0)
        weight.uniform(rng)

        let bias = RowVector<S>(storage: storage,
                                shape: Extent(num_outputs),
                                offset: num_inputs*num_outputs)

        let gradWeight = Matrix<S>(storage: gradStorage,
                                   shape: Extent(num_inputs, num_outputs),
                                   offset: 0)

        let gradBias = RowVector<S>(storage: gradStorage,
                                    shape: Extent(num_outputs),
                                    offset: num_inputs*num_outputs)

        // need to provide a method to point to external gradient storage as well
        let linear = LinearGradientModule<S>(weight: weight, bias: bias, gradWeight: gradWeight, gradBias: gradBias)
        testModule(rng, module: linear, storage: storage, gradStorage: gradStorage, numInputs: linear.shape[0], numOutputs: linear.shape[1])
    }

    func testSigmoidGradient() {
        typealias S = NativeStorage<Double>
        
        let rng = RandomNumberGenerator()
        
        let num_inputs = 20
        let num_outputs = 10

    }*/
    
    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measureBlock {
            // Put the code you want to measure the time of here.
        }
    }

}
