//
//  serializationTests.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/24/16.
//
//

import XCTest
@testable import Tensor

class serializationTests: XCTestCase {
    func testSimple() {
        let a = Tensor<NativeStorage<Float>>(Extent(5, 5))
        
        for i in 0..<5 {
            for j in 0..<5 {
                a[i, j] = Float(i*j)
            }
        }
        
        let bytes:[UInt8] = serialize(tensor: a)
        let b:Tensor<NativeStorage<Float>>? = deserialize(data: bytes)
        
        XCTAssertTrue(isClose(a, b!, eps: 10e-8))
    }
    
    func testView() {
        let a = Tensor<NativeStorage<Float>>(Extent(10, 10))
        
        for i in 0..<10 {
            for j in 0..<10 {
                a[i, j] = Float(i*j)
            }
        }
        
        let b = a[0..<5, 0..<5]
        let bytes:[UInt8] = serialize(tensor: b)
        let c:Tensor<NativeStorage<Float>>? = deserialize(data: bytes)
        
        XCTAssertTrue(isClose(b, c!, eps: 10e-8))
    }
}
