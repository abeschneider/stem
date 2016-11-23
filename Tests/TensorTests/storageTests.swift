//
//  storageTests.swift
//  stem
//
//  Created by Schneider, Abraham R. on 3/3/16.
//  Copyright © 2016 none. All rights reserved.
//

import XCTest
@testable import Tensor


class storageTests: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    
    func testStorageCreate() {
        _ = NativeStorage<Double>(size: 9)
    }
    
    func testCBlasStorageCreate() {
        _ = CBlasStorage<Double>(size: 9)
    }
    
    func testStorageIndexing1() {
        let storage = NativeStorage<Double>(size: 5)
        for i in 0..<5 {
            storage[i] = Double(i)
        }
        
        for i in 0..<5 {
            XCTAssertEqual(storage.array[i], Double(i))
        }
    }
    
    func testCBlasStorageIndexing1() {
        let storage = CBlasStorage<Double>(size: 5)
        for i in 0..<5 {
            storage[i] = Double(i)
        }
        
        for i in 0..<5 {
            XCTAssertEqual(storage.array[i], Double(i))
        }
    }
}
