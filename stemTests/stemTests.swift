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
        let storage = NativeStorage<Double>(shape: Extent(5))
        for i in 0..<storage.shape[0] {
            storage[i] = Double(i)
        }
        
        for i in 0..<storage.shape[0] {
            XCTAssertEqual(storage.array[i], Double(i))
        }
    }

    func testView1() {
        let storage = NativeStorage<Double>(shape: Extent(10))
        var view = StorageView(storage: storage, window: [1...4])
        
        for i in 0..<view.shape[0] {
            view[i] = Double(i+10)
        }
        
        let expected = [0.0, 10.0, 11.0, 12.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i in 0..<storage.shape[0] {
            XCTAssertEqual(storage[i], expected[i])
        }
    }
    
    func testView2() {
        let array = (0..<20).map { Double($0) }
        let storage = NativeStorage<Double>(array: array, shape: Extent(4, 5))
        let view = StorageView(storage: storage, window: [1..<3, 2..<4])
        let expected:[Double] = [7, 8, 12, 13]

        var k = 0;
        for i in 0..<view.shape[0] {
            for j in 0..<view.shape[1] {
                XCTAssertEqual(view[i, j], expected[k++])
            }
        }
    }
    
    func testView2C() {
        let array = (0..<20).map { Double($0) }
        let storage = CBlasStorage<Double>(shape: Extent(4, 5))
        var view1 = StorageView(storage: storage, window: [0..<4, 0..<5])
        
        var k = 0;
        for i in 0..<view1.shape[0] {
            for j in 0..<view1.shape[1] {
                view1[i, j] = array[k++]
            }
        }
        
        let view2 = StorageView(storage: storage, window: [1..<3, 2..<4])
        let expected:[Double] = [7, 8, 12, 13]
        
        k = 0
        for i in 0..<view2.shape[0] {
            for j in 0..<view2.shape[1] {
                XCTAssertEqual(view2[i, j], expected[k++])
            }
        }
    }

    
    // TODO: currently testing nothing new .. either delete, or update
    func testView3() {
        let array = (0..<100).map { Double($0) }
        let storage = NativeStorage<Double>(array: array, shape: Extent(10, 10))
        let view = StorageView(storage: storage, window: [0..<10, 0..<10])
        
        var k = 0
        for i in 0..<view.shape[0] {
            for j in 0..<view.shape[1] {
                XCTAssertEqual(view[i, j], array[k++])
            }
        }
    }
    
    
    func testViewIndex1() {
        let storage = CBlasStorage<Double>(shape: Extent(10))
        let view = StorageView(storage: storage)
        let indices = view.storageIndices()
        
        for (i, index) in GeneratorSequence(indices).enumerate() {
            XCTAssertEqual(i, index)
        }
    }
    
    func testViewIndex2() {
        let storage = NativeStorage<Double>(shape: Extent(10, 10))
        let view = StorageView(storage: storage, window: [5..<10, 5..<10])
        let indices = view.storageIndices()
        
        let expected:[Int] = [  55, 56, 57, 58, 59,
                                65, 66, 67, 68, 69,
                                75, 76, 77, 78, 79,
                                85, 86, 87, 88, 89,
                                95, 96, 97, 98, 99]
        
        var i:Int = 0
        for j in 0..<view.shape[0] {
            for k in 0..<view.shape[1] {
                let idx = indices.next()!
                print("index = \(idx)")
                XCTAssertEqual(idx, expected[i++])
                XCTAssertEqual(storage[idx], view[j, k])
            }
        }
    }
    
    func testViewIndex3() {
        let storage = CBlasStorage<Double>(shape: Extent(10, 10))
        let view = StorageView(storage: storage, window: [5..<10, 5..<10])
        let indices = view.storageIndices()
        
        let expected:[Int] = [  55, 65, 75, 85, 95,
                                56, 66, 76, 86, 96,
                                57, 67, 77, 87, 97,
                                58, 68, 78, 88, 98,
                                59, 69, 79, 89, 99]
        
        var i:Int = 0
        for j in 0..<5 {
            for k in 0..<5 {
                let idx = indices.next()!
                XCTAssertEqual(idx, expected[i++])
                XCTAssertEqual(storage[idx], view[j, k])
            }
        }
    }
    
    func testViewIndex4() {
        let storage = NativeStorage<Double>(shape: Extent(5, 10))
        let view1 = StorageView(storage: storage, window: [0..<5, 0..<10])
        let view2 = StorageView(storage: storage, window: [0..<5, 1..<3])
        let view3 = StorageView(storage: storage, window: [1..<3, 0..<10])
        
        // alter only the values in the view
        for (i, index) in view2.storageIndices().enumerate() {
            storage[index] = Double(i+5)
        }
        
        for index in view3.storageIndices() {
            storage[index] = 6.0
        }
        
        let expected:[[Double]] = [[0, 5, 6, 0, 0, 0, 0, 0, 0, 0],
                                   [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                                   [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                                   [0, 11, 12, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 13, 14, 0, 0, 0, 0, 0, 0, 0]]
        
        for i in 0..<view1.shape[0] {
            for j in 0..<view1.shape[1] {
                XCTAssertEqual(view1[i, j], expected[i][j])
            }
        }
    }
    
    func testStorageTranspose() {
        let a:[Double] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        let storage1 = NativeStorage<Double>(array: a, shape: Extent(2, 5))
        
        let view1 = StorageView(storage: storage1, window: [0..<2, 0..<5])
        
        let window = Array(view1.window.reverse())
        let dimIndex = Array(view1.dimIndex.reverse())
        let view2 = StorageView(storage: view1.storage, window: window, dimIndex: dimIndex)
        
        XCTAssertEqual(view2.shape, Extent(5, 2))
        
        let expected:[Double] = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
        
        var k = 0
        for i in 0..<view2.shape[0] {
            for j in 0..<view2.shape[1] {
                XCTAssertEqual(view2[i, j], expected[k++])
            }
        }
    }
    
    func testCStorageTranspose() {
        let a:[Double] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        let storage1 = CBlasStorage<Double>(shape: Extent(2, 5))
        
        var view1 = StorageView(storage: storage1, window: [0..<2, 0..<5], dimIndex: [1, 0])
        
        var k = 0
        for i in 0..<view1.shape[0] {
            for j in 0..<view1.shape[1] {
                view1[i, j] = a[k++]
            }
        }
        
        let window = Array(view1.window.reverse())
        let dimIndex = Array(view1.dimIndex.reverse())
        let view2 = StorageView(storage: view1.storage, window: window, dimIndex: dimIndex)
        
        XCTAssertEqual(view2.shape, Extent(5, 2))
        
        let expected:[Double] = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
        
        k = 0
        for i in 0..<view2.shape[0] {
            for j in 0..<view2.shape[1] {
                XCTAssertEqual(view2[i, j], expected[k++])
            }
        }
    }
    
    func testIndexTranspose() {
        let array:[Double] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        let storage = NativeStorage<Double>(array: array, shape: Extent(2, 5))
        let view1 = StorageView(storage: storage, window: [0..<2, 0..<5])
        
        let window = Array(view1.window.reverse())
        let dimIndex = Array(view1.dimIndex.reverse())
        let view2 = StorageView(storage: view1.storage, window: window, dimIndex: dimIndex)
        
        let index = view2.storageIndices()
        
        let expected = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
        
        var k = 0
        for _ in 0..<view2.shape[0] {
            for _ in 0..<view2.shape[1] {
                XCTAssertEqual(index.next()!, expected[k++])
            }
        }
    }
    
    func testCreateTensor1() {
        let v = Tensor<NativeStorage<Double>>(array: [1, 2, 3, 4], shape: Extent(4))
        XCTAssertEqual(v.view.storage.shape.dims(), 1)
        XCTAssertEqual(v.view.storage.shape[0], 4)
        XCTAssertEqual(v.view.shape.dims(), 1)
        XCTAssertEqual(v.view.shape[0], 4)
        
        for i in 0..<4 {
            XCTAssertEqual(v[i], Double(i+1))
        }
    }
    
    func testCreateTensor2() {
        let array = (0..<20).map { Double($0) }
        let v = Tensor<NativeStorage<Double>>(array: array, shape: Extent(2, 5))
        XCTAssertEqual(v.shape, Extent(2, 5))
        
        var k = 0
        for i in 0..<v.shape[0] {
            for j in 0..<v.shape[1] {
                XCTAssertEqual(v[i, j], array[k++])
            }
        }
    }
    
    func testCreateNativeVector() {
        let array:[Double] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        let v = Vector<NativeStorage<Double>>(array)
        
        for i in 0..<v.shape[0] {
            XCTAssertEqual(v[i], array[i])
        }
    }
    
    func testCreateNativeMatrix() {
        let array:[[Double]] = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        let m = Matrix<NativeStorage<Double>>(array)
        
        // storage should not affect access
        for i in 0..<m.shape[0] {
            for j in 0..<m.shape[1] {
                XCTAssertEqual(m[i, j], array[i][j])
            }
        }
        
        // however storage should have entries in column-major format
        let expected:[Double] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        for i in 0..<expected.count {
            XCTAssertEqual(m.view.storage.array[i], expected[i])
        }
    }
    
    func testCreateCBlasMatrix() {
        let a:[[Double]] = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        let m = Matrix<CBlasStorage<Double>>(a)
        
        print(m.view.storage.array)
        
        // storage should not affect access
        for i in 0..<2 {
            for j in 0..<5 {
                XCTAssertEqual(m[i, j], a[i][j])
            }
        }
        
        // however storage should have entries in column-major format
        let expected:[Double] = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
        for i in 0..<expected.count {
            XCTAssertEqual(m.view.storage.array[i], expected[i])
        }
    }
    
    func testVectorToString() {
        let v = Vector<CBlasStorage<Double>>([1, 2, 3, 4, 5])
        
        let expected = "[1.000,\t2.000,\t3.000,\t4.000,\t5.000]"
        XCTAssertEqual(String(v), expected)
    }
    
    func testMatrixToString() {
        let m = Matrix<NativeStorage<Double>>([[1, 2, 3, 4], [5, 6, 7, 8]])
        
        // layout should be row-major
        let expected = "[[1.000,\t2.000,\t3.000,\t4.000]\n [5.000,\t6.000,\t7.000,\t8.000]]"
        XCTAssertEqual(String(m), expected)
    }

    
    func testVectorAdd() {
        let m1 = Vector<NativeStorage<Double>>([1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = Vector<NativeStorage<Double>>([8, 7, 6, 5, 4, 3, 2, 1])
        let result = Vector<NativeStorage<Double>>(rows: m1.shape[0])
        
        add(left: m1, right: m2, result: result)
        
        let expected = Vector<NativeStorage<Double>>([9, 9, 9, 9, 9, 9, 9, 9])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorAdd2() {
        let m1 = Vector<NativeStorage<Double>>([1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = Vector<NativeStorage<Double>>([8, 7, 6, 5, 4, 3, 2, 1])
        
        let result = m1 + m2
        
        let expected = Vector<NativeStorage<Double>>([9, 9, 9, 9, 9, 9, 9, 9])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorMatrixAdd1() {
        let m1 = Matrix<NativeStorage<Double>>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let m2 = Matrix<NativeStorage<Double>>([[8, 7, 6, 5], [4, 3, 2, 1]])
        let result = Matrix<NativeStorage<Double>>(rows: m1.shape[0], cols: m1.shape[1])
        
        add(left: m1, right: m2, result: result)

        let expected = Matrix<NativeStorage<Double>>([[9, 9, 9, 9], [9, 9, 9, 9]])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorDotProduct() {
        let m = Matrix<NativeStorage<Double>>([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        let v = Vector<NativeStorage<Double>>([1, 2, 3])
        
        let result = Vector<NativeStorage<Double>>(rows: 3)
        dot(left: m, right: v, result: result)
        
        let expected = Vector<NativeStorage<Double>>([1, 2, 3])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }

    func testCBlasVectorAdd1() {
        let m1 = Vector<CBlasStorage<Double>>([1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = Vector<CBlasStorage<Double>>([8, 7, 6, 5, 4, 3, 2, 1])
        let result = Vector<CBlasStorage<Double>>(rows: m1.shape[0])
        
        add(left: m1, right: m2, alpha: 1.0, result: result)
        
        let expected = Vector<CBlasStorage<Double>>([9, 9, 9, 9, 9, 9, 9, 9])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testCBlasVectorAdd2() {
        let m1 = Vector<CBlasStorage<Double>>([1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = Vector<CBlasStorage<Double>>([8, 7, 6, 5, 4, 3, 2, 1])
        
        let result = m1 + m2
        
        let expected = Vector<CBlasStorage<Double>>([9, 9, 9, 9, 9, 9, 9, 9])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testCBlasVectorMatrixAdd1() {
        let v1 = Vector<CBlasStorage<Double>>([5, 4])
        let v2 = Vector<CBlasStorage<Double>>([4, 5])
        let result = Vector<CBlasStorage<Double>>(rows: v1.shape[0])
        
        add(left: v1, right: v2, alpha: 1.0, result: result)

        let expected = Vector<CBlasStorage<Double>>([9, 9])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }

//    func testCBlasVectorMatrixAdd2() {
//        let v1 = Vector<CBlasStorage<Double>>([5, 4])
//        let v2 = Vector<CBlasStorage<Double>>([4, 5])
//        let result = Vector<CBlasStorage<Double>>(rows: v1.shape[0])
//        
//        add(left: v1.transpose(), right: v2.transpose(), alpha: 1.0, result: result)
//        
//        let expected = Vector<CBlasStorage<Double>>([9, 9])
//        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
//    }

    func testCVectorDotProduct() {
        let m = Matrix<CBlasStorage<Double>>([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        let v = Vector<CBlasStorage<Double>>([1, 2, 3])
        
        let result = Vector<CBlasStorage<Double>>(rows: 3)
        dot(left: m, right: v, result: result)
        
        let expected = Vector<CBlasStorage<Double>>([1, 2, 3])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    /*
    
    
    

    

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
//    }*/
}
