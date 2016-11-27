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
    
    func testDictionary() {
        let a = Tensor<NativeStorage<Float>>(Extent(5, 5))
        fill(a, value: 1)
        let b = Tensor<NativeStorage<Float>>(Extent(3, 3))
        fill(b, value: 2)
        
        let dict:[String:Tensor<NativeStorage<Float>>] = ["a":a, "b":b]
        let data:Data? = serialize(tensors: dict)
        
        let dict2:[String:Tensor<NativeStorage<Float>>] = deserialize(data: data!)
        
        XCTAssertTrue(isClose(a, dict2["a"]!, eps: 10e-8))
        XCTAssertTrue(isClose(b, dict2["b"]!, eps: 10e-8))
    }
}
