//
//  ordereddictionaryTests.swift
//  stem
//
//  Created by Schneider, Abraham R. on 6/12/16.
//  Copyright © 2016 none. All rights reserved.
//

import XCTest
@testable import stem

class ordereddictionaryTests: XCTestCase {
    typealias F = NativeStorage<Float>
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testCreateDictionary() {
        _ = OrderedDictionary<F>()
    }
    
    func testAddItem() {
        var dict = OrderedDictionary<F>()
        let op1 = Linear<F>(inputSize: 5, outputSize: 10)
        let op2 = Linear<F>(inputSize: 5, outputSize: 5)
        dict["l1"] = op1
        dict["l2"] = op2
        
        
        XCTAssert(dict.keys[0] == "l1")
//        XCTAssert(dict.orderedValues[0] == op1)
        XCTAssert(dict["l1"] == op1)
        
        XCTAssert(dict.keys[1] == "l2")
//        XCTAssert(dict.orderedValues[1]! == op2)
        XCTAssert(dict["l2"] == op2)
    }
    
    func testChangeItem() {
        var dict = OrderedDictionary<F>()
        let op1 = Linear<F>(inputSize: 5, outputSize: 10)
        let op2 = Linear<F>(inputSize: 5, outputSize: 5)
        let op3 = Linear<F>(inputSize: 5, outputSize: 20)
        let op4 = Linear<F>(inputSize: 5, outputSize: 3)
        
        dict["l1"] = op1
        dict["l2"] = op2
        dict["l1"] = op3
        dict["l2"] = op4
        
        XCTAssert(dict.keys[0] == "l1")
//        XCTAssert(dict.orderedValues[0] == op3)
        XCTAssert(dict["l1"] == op3)
        
        XCTAssert(dict.keys[1] == "l2")
//        XCTAssert(dict.orderedValues[1] == op4)
        XCTAssert(dict["l2"] == op4)
    }
}
