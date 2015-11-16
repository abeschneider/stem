//
//  stemTests.swift
//  stemTests
//
//  Created by Abe Schneider on 11/10/15.
//  Copyright © 2015 none. All rights reserved.
//

import XCTest
@testable import stem

class stemTests: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testStorageCreate() {
        _ = CBlasStorage<Double>(shape: Extent(3, 3))
    }
    
    func testStorageIndex1() {
        let storage = CBlasStorage<Double>(shape: Extent(5))
        for i in 0..<storage.shape[0] {
            storage[i] = Double(i)
        }
        
        for i in 0..<storage.shape[0] {
            XCTAssertEqual(storage.array[i], Double(i))
        }
    }

    func testView1() {
        let storage = CBlasStorage<Double>(shape: Extent(10))
        let view = StorageRowView(storage: storage, view: 1...4)
        
        for i in 0..<view.shape[0] {
            view[i] = Double(i+10)
        }
        
        let expected = [0.0, 10.0, 11.0, 12.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i in 0..<storage.shape[0] {
            XCTAssertEqual(storage[i], expected[i])
        }
    }
    
    func testView2() {
        let storage = CBlasStorage<Double>(shape: Extent(10, 10))
        let view = StorageRowView(storage: storage, view: 2..<8, 2..<8)
        
        for i in 0..<100 {
            storage[i] = Double(i)
        }
        
        let expected = CBlasStorage<Double>(array: (0..<100).map { Double($0) }, shape: Extent(10, 10))
        let expectedView = StorageRowView(storage: expected)

        for i in 0..<view.shape[0] {
            for j in 0..<view.shape[1] {
                XCTAssertEqual(view[i, j], expectedView[i+2, j+2])
            }
        }
    }
    
    func testView3() {
        let storage = CBlasStorage<Double>(shape: Extent(10, 10))
        let view = StorageColumnView(storage: storage, view: 2..<8, 2..<8)
        
        for i in 0..<100 {
            storage[i] = Double(i)
        }
        
        let expected = CBlasStorage<Double>(array: (0..<100).map { Double($0) }, shape: Extent(10, 10))
        let expectedView = StorageRowView(storage: expected)
        
        for i in 0..<view.shape[0] {
            for j in 0..<view.shape[1] {
                XCTAssertEqual(view[i, j], expectedView[j+2, i+2])
            }
        }
    }
    
    
    func testColumnIndex1() {
        let storage = CBlasStorage<Double>(shape: Extent(10))
        let view = StorageRowView(storage: storage)
        let indices = view.storageIndices()
        
        for (i, index) in GeneratorSequence(indices).enumerate() {
            XCTAssertEqual(i, index)
        }
    }
    
    func testColumnIndex2() {
        let storage = CBlasStorage<Double>(shape: Extent(10, 10))
        let view = StorageColumnView(storage: storage, view: 5..<10, 5..<10)
        let indices = view.storageIndices()
        
        let expected:[Int] = [  55, 65, 75, 85, 95,
                                56, 66, 76, 86, 96,
                                57, 67, 77, 87, 97,
                                58, 68, 78, 88, 98,
                                59, 69, 79, 89, 99]
        
        var i:Int = 0
        for _ in 5..<10 {
            for _ in 5..<10 {
                XCTAssertEqual(indices.next()!, expected[i++])
            }
        }
    }
    
    func testColumnIndex3() {
        let storage = CBlasStorage<Double>(shape: Extent(5, 10))
        let view1 = StorageColumnView(storage: storage, view: 0..<5, 0..<10)
        let view2 = StorageColumnView(storage: storage, view: 0..<5, 1..<2)
        
        // alter only the values in the view
        for (i, index) in view2.storageIndices().enumerate() {
            storage[index] = Double(i+5)
        }
        
        let expected:[[Double]] = [[0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 8, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 9, 0, 0, 0, 0, 0, 0, 0, 0]]
        
        for i in 0..<view1.shape[0] {
            for j in 0..<view1.shape[1] {
                XCTAssertEqual(view1[i, j], expected[i][j])
            }
        }
    }
    
    func testCreateTensor1() {
        let v = CDTensor(array: [1, 2, 3, 4], shape: Extent(4))
        XCTAssertEqual(v.view.storage.shape.dims(), 1)
        XCTAssertEqual(v.view.storage.shape[0], 4)
        XCTAssertEqual(v.view.shape.dims(), 1)
        XCTAssertEqual(v.view.shape[0], 4)
        
        for i in 0..<4 {
            XCTAssertEqual(v[i], Double(i+1))
        }
    }
    
    func testVectorToString() {
        let v = CDVector([1, 2, 3, 4, 5])
        
        let expected = "[1.000,\t2.000,\t3.000,\t4.000,\t5.000]"
        XCTAssertEqual(String(v), expected)
    }
    
    func testNMatrixToString() {
        let m = DMatrix([[1, 2, 3, 4], [5, 6, 7, 8]])
        
        // layout should be row-major
        let expected = "[[1.000,\t2.000,\t3.000,\t4.000]\n [5.000,\t6.000,\t7.000,\t8.000]]"
        XCTAssertEqual(String(m), expected)
    }
    
    func testCMatrixToString() {
        let m = CDMatrix([[1, 2, 3, 4], [5, 6, 7, 8]])
        
        // layout should be column-major
        let expected = "[[1.000,\t3.000,\t5.000,\t7.000]\n [2.000,\t4.000,\t6.000,\t8.000]]"

        XCTAssertEqual(String(m), expected)
    }
    
    func testNVectorAdd() {
        let m1 = DVector([1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = DVector([8, 7, 6, 5, 4, 3, 2, 1])
        let result = DVector(rows: m1.shape[0])
        
        add(left: m1, right: m2, result: result)
        
        let expected = DVector([9, 9, 9, 9, 9, 9, 9, 9])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testCVectorAdd() {
        let m1 = CDVector([1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = CDVector([8, 7, 6, 5, 4, 3, 2, 1])
        let result = CDVector(rows: m1.shape[0])
        
        add(left: m1, right: m2, alpha: 1.0, result: result)
        
        let expected = CDVector([9, 9, 9, 9, 9, 9, 9, 9])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testNVectorAdd2() {
        let m1 = DVector([1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = DVector([8, 7, 6, 5, 4, 3, 2, 1])
        
        let result = m1 + m2
        
        let expected = DVector([9, 9, 9, 9, 9, 9, 9, 9])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testCDVectorAdd2() {
        let m1 = CDVector([1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = CDVector([8, 7, 6, 5, 4, 3, 2, 1])
        
        let result = m1 + m2
        
        let expected = CDVector([9, 9, 9, 9, 9, 9, 9, 9])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }

    // test a column-major vector
    func testNVectorMatrixAdd1() {
        let m1 = DMatrix([[1, 2, 3, 4], [5, 6, 7, 8]])
        let m2 = DMatrix([[8, 7, 6, 5], [4, 3, 2, 1]])
        let result = DVector(rows: m1.shape[1])
        
        add(left: m1[0,0..<4], right: m2[0,0..<4], result: result)
        
        let expected = DVector([9, 9, 9, 9])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    // test a row-major vector
    func testCVectorMatrixAdd2() {
        let m = CDMatrix([[1, 2, 3, 4], [5, 6, 7, 8]])

        let v1 = CDVector(m[0..<2, 1])
        let v2 = CDVector(m[0..<2, 2])
        let result = CDVector(rows: v1.shape[0])
        
        add(left: v1, right: v2, alpha: 1.0, result: result)
        
        let expected = CDVector([8, 10])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testNVectorDotProduct() {
        let m = DMatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        let v = DVector([1, 2, 3])
        
        let result = DVector(rows: 3)
        dot(left: m, right: v, result: result)
        
        let expected = DVector([1, 2, 3])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testCVectorDotProduct() {
        let m = CDMatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        let v = CDVector([1, 2, 3])
        
        let result = CDVector(rows: 3)
        dot(left: m, right: v, result: result)
        
        let expected = CDVector([1, 2, 3])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }

    func testBenchmarkCBlas() {
        let v1 = CDVector(rows: 1000000)
        
        self.measureBlock {
            let _ = v1 + v1
        }
    }

    func testBenchmarkNative() {
        let v1 = DVector(rows: 10000)
        
        self.measureBlock {
            let _ = v1 + v1
        }
    }

    //    func testPerformanceExample() {
//        // This is an example of a performance test case.
//        self.measureBlock {
//            // Put the code you want to measure the time of here.
//        }
//    }
}
