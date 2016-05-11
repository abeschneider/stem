//
//  graphTests.swift
//  stem
//
//  Created by Schneider, Abraham R. on 4/17/16.
//  Copyright © 2016 none. All rights reserved.
//

import XCTest
import stem

class graphTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    func testExample() {
//        typealias D = NativeStorage<Double>
//        
//        let W = Symbol<D>(value: Matrix<D>(rows: 3, cols: 3))
//        let bias = Symbol<D>(value: Vector<D>(rows: 3))
//        let input = Symbol<D>(value: Vector<D>(cols: 3))
//        
//        // TODO: Provide method to construct that doesn't require input. This
//        // will allow convenience methods of construction like:
//        //  sequence(Linear(W, b), Sigmoid()),
//        // which would create the topology along with the traversal.
//        
//        let linear = Linear<D>(input: input, weight: W, bias: bias)
//        let sigmoid = Sigmoid<D>(input: linear.outputs[0])
//        
//        let seq = SequentialTraversal<D>()
//        seq.add(linear)
//        seq.add(sigmoid)
//        seq.update()
    }
    
    func testExamples2() {
        typealias D = NativeStorage<Double>
        let v1 = Tensor<D>(Extent(1, 3))
        let v2 = Tensor<D>(Extent(3, 1))
        let result = Tensor<D>(Extent(1, 1))
        dot(v1, v2, result: result)
    }
}
