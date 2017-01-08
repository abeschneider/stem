//
//  optimTests.swift
//  stem
//
//  Created by Abraham Schneider on 1/7/17.
//
//

import XCTest
import Foundation
@testable import Tensor
@testable import Op

typealias F = NativeStorage<Float>
typealias D = NativeStorage<Double>

class optimTests: XCTestCase {
    func testSimple() {
        let alpha = 10e-5
        let input = ConstantOp<D>(Tensor(Extent(28, 28)))
        let target = ConstantOp<D>(Tensor(Extent(28, 28)))
        
        // use random input and targets (should not be dependent
        // on initial condition)
        input.output.uniform()
        target.output.uniform()

        let model = SequentialOp<D>()
        model.append(input)
        model.append(Conv2dOp(numFilters: 1, filterSize: Extent(3, 3)))
        
        let loss = L2Loss<D>()
        connect(from: target, "output", to: loss, "target")
        model.append(loss)
        
        let modelGrad = model.gradient() as! Op<D>
        
        // Get a list of all the parameters. This allows us to
        // update based on gradParams.
        let params = model.params()
        let gradParams = modelGrad.params()
        
        // calculate initial loss
        model.apply()
        modelGrad.apply()
        let initialLoss = loss.value
        
        for _ in 0..<10 {
            model.apply()
            modelGrad.apply()
            
            // update our params from the gradient
            for (param, gradParam) in zip(params, gradParams) {
                param -= (alpha*gradParam)
            }
        
            // reset gradients
            modelGrad.reset()
        }
        
        // Given random initialization, we don't know the exact
        // values, but we should expect this to be lower than before.
        XCTAssertLessThan(loss.value, initialLoss)
    }
}
