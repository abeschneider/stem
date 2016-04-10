//
//  randomTests.swift
//  stem
//
//  Created by Abe Schneider on 12/10/15.
//  Copyright © 2015 none. All rights reserved.
//

import XCTest
@testable import stem

class randomTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

//    func testRandomInt() {
//        let rng = RandomNumberGenerator()
//        let tensor = Tensor<NativeStorage<Int>>(shape: Extent(100000))
//        
//        // fill tensor with random numbers from a uniform distribution between 0 and 10
//        tensor.uniform(rng, first: 0, last: 10)
//        
//        let h = hist(tensor, bins: 10)
//        let m = max(h)
//        
//        // reinterpret as a float so we can perform division (there should be a
//        // cleaner method of doing this; likely through providing division rules via NumericType)
//        let h_float = Tensor<NativeStorage<Double>>(storage: h.storage.transform({ Double($0) }), shape: h.shape)
//        h_float /= Double(m)
//        
//        // with enough samples, everything should approach 1.0
//        for i in h_float.storageIndices() {
//            XCTAssertGreaterThan(h_float.storage[i], 0.9)
//        }
//    }
    
    func testRandomDouble() {
        let rng = RandomNumberGenerator()
        let size = 10000
        let bins = 10
        
        // must be within 10% (increase size for higher precision)
        let margin = 0.05
        let expected = Double(size)/Double(bins)
        let expected_margin = expected - expected*margin
        
        let tensor = Tensor<NativeStorage<Double>>(shape: Extent(size))
        tensor.uniform(rng)
        let h = hist(tensor, bins: bins)
        for i in h.indices() {
            XCTAssertGreaterThan(h[i], expected_margin)
        }
    }

    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measureBlock {
            // Put the code you want to measure the time of here.
        }
    }

}
