//
//  opSpeedTests.swift
//  stem
//
//  Created by Abraham Schneider on 1/9/17.
//
//

import Foundation
import XCTest
@testable import Tensor
@testable import Op

class opSpeedTests: XCTestCase {
    typealias DN = NativeStorage<Double>
    typealias DB = CBlasStorage<Double>
    typealias I = NativeStorage<Int>
    
    func testNativeMatMulPerformance() {
        let a:Tensor<DN> = uniform(Extent(50, 50))
        let b:Tensor<DN> = uniform(Extent(50, 50))
        let result = Tensor<DN>(Extent(50, 50))
        
        self.measure {
            dot(a, b, result: result)
        }
    }

    func testBLASMatMulPerformance() {
        let a:Tensor<DB> = uniform(Extent(50, 50))
        let b:Tensor<DB> = uniform(Extent(50, 50))
        let result = Tensor<DB>(Extent(50, 50))
        self.measure {
            dot(a, b, result: result)
        }
    }
}
