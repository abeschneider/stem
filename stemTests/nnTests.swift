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

    func testLinearForwardVector() {
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
        let linear = LinearModule<NativeStorage<Double>>(weight: w)
        let input = ColumnVector<NativeStorage<Double>>([1, 2, 3])
        
        linear.forward(input)
        let grad_output = ColumnVector<NativeStorage<Double>>([1, 1, 1, 1])
        let grad_input = linear.backward(input, grad_output: grad_output)
  
        print(grad_input)
        let expected = Vector<NativeStorage<Double>>([1, 1, 1])
        XCTAssert(isClose(grad_input, expected, eps: 10e-4), "Not close")

        let w2 = Matrix<CBlasStorage<Double>>([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], copyTransposed: true)
        let linear2 = LinearModule<CBlasStorage<Double>>(weight: w2)
        let input2 = ColumnVector<CBlasStorage<Double>>([1, 2, 3])
        linear2.forward(input2)
        
        let grad_output2 = ColumnVector<CBlasStorage<Double>>([1, 1, 1, 1])
        let grad_input2 = linear2.backward(input2, grad_output: grad_output2)
        
        let expected2 = Vector<CBlasStorage<Double>>([1, 1, 1])
        XCTAssert(isClose(grad_input2, expected2, eps: 10e-4), "Not close")
    }
    
    func testNativeLinearGradient() {
        typealias S = NativeStorage<Double>
        
        let num_inputs = 100
        let num_outputs = 50
        
        // provides a flat view of all parameters to make gradient testing simple
        let storage = S(size: num_inputs*num_outputs + num_outputs)
        let grad_storage = S(size: num_inputs*num_outputs + num_outputs)
        
        var pos = 0
        let weight = Matrix<S>(storage: storage, shape: Extent(num_inputs, num_outputs), offset: pos)

        pos += num_inputs*num_outputs
        let bias = RowVector<S>(storage: storage, shape: Extent(num_outputs), offset: pos)

        pos += 0
        let grad_weight = Matrix<S>(storage: grad_storage, shape: Extent(num_inputs, num_outputs), offset: pos)

        pos += num_inputs*num_outputs
        let grad_bias = RowVector<S>(storage: grad_storage, shape: Extent(num_outputs), offset: pos)

        // need to provide a method to point to external gradient storage as well
        let linear = LinearModule<S>(weight: weight, bias: bias, gradWeight: grad_weight, gradBias: grad_bias)


        let input = ColumnVector<S>(rows: num_inputs)
//        check_gradient(linear, storage, grad_storage, input, 10e-4) { (module, input) -> Vector<S> in
//            let output = module.forward(input)
//            // cost function
//        }
        
//        let output = linear.forward(input)
//        let grad_output = ColumnVector<S>(rows: num_outputs)
//        let grad_input = linear.backward(input, grad_output: grad_output)
    }

    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measureBlock {
            // Put the code you want to measure the time of here.
        }
    }

}
