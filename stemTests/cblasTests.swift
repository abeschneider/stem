//
//  cblasTests.swift
//  stem
//
//  Created by Schneider, Abraham R. on 5/7/16.
//  Copyright © 2016 none. All rights reserved.
//

import XCTest
@testable import stem

typealias CD = CBlasStorage<Double>
typealias CF = CBlasStorage<Float>
typealias CI = CBlasStorage<Int>

class cblasTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    func asColumnMajor(array:[Double], rows:Int, cols:Int) -> [Double] {
        let m = Tensor<D>(array: array, shape: Extent(rows, cols))
        var result = Array<Double>(count: m.shape.elements, repeatedValue: Double(0))
        
        var k = 0
        for i in 0..<m.shape[1] {
            for j in 0..<m.shape[0] {
                result[k] = m[j, i]
                k += 1
            }
        }
        
        return result
    }

    func testCBlasTensorCreate() {
        // NB: storage for CBlas follows a column-major format
        let array:[Double] = asColumnMajor((0..<10).map { Double($0) }, rows: 2, cols: 5)
        let t = Tensor<CBlasStorage<Double>>(array: array, shape: Extent(2, 5))
        
        XCTAssertEqual(t.shape, Extent(2, 5))
        
        let expected = (0..<10).map { Double($0) }
        
        // traverse the tensor in column major to
        // test expected values
        var k = 0
        for index in t.indices(.ColumnMajor) {
            XCTAssertEqual(t[index], expected[k])
            k += 1
        }
    }
    
    func testCBlasTensorView1() {
        let array = asColumnMajor((0..<100).map { Double($0) }, rows: 10, cols: 10)
        let t1 = Tensor<CD>(array: array, shape: Extent(10, 10))
        let t2 = t1[1..<3, 1..<3]
        
        let expected:[Double] = [11, 12, 21, 22]
        
        var k = 0
        for i in t2.indices(.ColumnMajor) {
            XCTAssertEqual(t2[i], expected[k])
            k += 1
        }
    }
    
    func testCBlasTensorView2() {
        //        let tensor1 = Tensor<CD>(shape: Extent(3, 5))
        let t1:Tensor<CD> = Tensor<CD>(shape: Extent(3, 5))
        
        // top row
        let t2 = t1[0, 0..<5]
        for i in t2.indices() {
            t2[i] = 1
        }
        
        // second column
        let t3 = t1[0..<3, 1]
        for i in t3.indices() {
            t3[i] = 2
        }
        
        // lower right area
        let t4 = t1[1..<3, 2..<5]
        for i in t4.indices() {
            t4[i] = 3
        }
        
        let expected:[Double] = [1, 2, 1, 1, 1,
                                 0, 2, 3, 3, 3,
                                 0, 2, 3, 3, 3]
        
        var k = 0
        for i in t1.indices(.ColumnMajor) {
            XCTAssertEqual(t1[i], expected[k])
            k += 1
        }
    }

    func testCBlasTensorTranspose() {
        let array = asColumnMajor((0..<10).map { Double($0) }, rows: 2, cols: 5)
        let tensor1 = Tensor<CD>(array: array, shape: Extent(2, 5))
        let tensor2 = tensor1.transpose()
        
        // verify dimensions are correct
        XCTAssertEqual(tensor2.shape, Extent(5, 2))
        
        let expected:[Double] = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
        var k = 0
        
        for i in tensor2.indices(.ColumnMajor) {
            XCTAssertEqual(tensor2[i], expected[k])
            k += 1
        }
    }
    
    func testCBlasTensorReshape() {
        // TODO: check this is correct
        let array = asColumnMajor((0..<20).map { Double($0) }, rows: 4, cols: 5)
        let tensor1 = Tensor<CD>(array: array, shape: Extent(2, 10))
        let tensor2 = tensor1.reshape(Extent(4, 5))
        
        // verify dimensions are correct
        XCTAssertEqual(tensor2.shape, Extent(4, 5))
        
        let expected = (0..<20).map { Double($0) }
        
        // verify contents are still valid
        var k = 0
        for i in tensor2.indices(.ColumnMajor) {
            XCTAssertEqual(tensor2[i], expected[k])
            k += 1
        }
    }
    
    func testCBlasVectorAdd() {
        let m1 = Tensor<CD>(colvector: [1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = Tensor<CD>(colvector: [8, 7, 6, 5, 4, 3, 2, 1])
        let result = Tensor<CD>(shape: Extent(m1.shape[0]))
        
        add(left: m1, right: m2, result: result)
        
        let expected = Tensor<CD>(colvector: [9, 9, 9, 9, 9, 9, 9, 9])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testCBlasVectorAdd2() {
        let m1 = Tensor<CD>(colvector: [1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = Tensor<CD>(colvector: [8, 7, 6, 5, 4, 3, 2, 1])
        
        let result = m1 + m2
        
        let expected = Tensor<CD>(colvector: [9, 9, 9, 9, 9, 9, 9, 9])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }

    func testCBlasMatrixAdd1() {
        let m1 = Tensor<CD>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let m2 = Tensor<CD>([[8, 7, 6, 5], [4, 3, 2, 1]])
        let result = Tensor<CD>(shape: m1.shape)
        
        add(left: m1, right: m2, result: result)
        
        let expected = Tensor<CD>([[9, 9, 9, 9], [9, 9, 9, 9]])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }

    func testCBlasMatrixVectorAdd1() {
        let m = Tensor<CD>([[1, 2, 3, 4], [5, 6, 7, 8]])

        let v1 = Tensor<CD>(rowvector: [1, 1, 1, 1])
        let v2 = Tensor<CD>(colvector: [0.5, 0.5])
        let result = Tensor<CD>(shape: Extent(m.shape))

        add(left: m, right: v1, result: result)
        iadd(left: result, right: v2)

        let expected = Tensor<CD>([[2.5, 3.5, 4.5, 5.5], [6.5, 7.5, 8.5, 9.5]])
        
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testCBlasVectorSub() {
        let m1 = Tensor<CD>(colvector: [1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = Tensor<CD>(colvector: [8, 7, 6, 5, 4, 3, 2, 1])
        let result = Tensor<CD>(shape: m1.shape)
        
        sub(left: m1, right: m2, result: result)
        
        let expected = Tensor<CD>(colvector: [-7, -5, -3, -1, 1, 3, 5, 7])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }

    func testCBlasVectorSub2() {
        let m1 = Tensor<CD>(colvector: [1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = Tensor<CD>(colvector: [8, 7, 6, 5, 4, 3, 2, 1])
        
        let result = m1 - m2
        
        let expected = Tensor<CD>(colvector: [-7, -5, -3, -1, 1, 3, 5, 7])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testBroadcast0() {
        let v1 = Tensor<CD>(rowvector: [1, 2, 3]) // shape: (1, 3)
        let s1 = calculateBroadcastStride(v1, shape: Extent(3, 3))
        XCTAssertEqual(s1, [0, 1])

        let m1 = Tensor<CD>([[1, 2, 3]]) // shape: (1, 3)
        let s3 = calculateBroadcastStride(m1, shape: Extent(3, 3))
        XCTAssertEqual(s3, [0, 1])
        
        let v2 = Tensor<CD>(colvector: [1, 2, 3]) // shape: (3, 1)
        let s2 = calculateBroadcastStride(v2, shape: Extent(3, 3))
        XCTAssertEqual(s2, [1, 0])
        
        let m2 = Tensor<CD>([[1], [2], [3]]) // shape: (3, 1)
        let s4 = calculateBroadcastStride(m2, shape: Extent(3, 3))
        XCTAssertEqual(s4, [1, 0])

        let m3 = Tensor<CD>([[1, 2, 3], [4, 5, 6]]) // shape: (2, 3)
        let s5 = calculateBroadcastStride(m3, shape: Extent(2, 2, 3))
        XCTAssertEqual(s5, [0, 1, 2])
    }
    
    func testBroadcast1() {
        let v = Tensor<CD>(rowvector: [1, 2, 3])
        let b = broadcast(v, shape: Extent(3, 3))
        
        let expected = [1.0, 2.0, 3.0,
                        1.0, 2.0, 3.0,
                        1.0, 2.0, 3.0]
  
        print("\(b)")
        var k = 0
        for i in b.indices(.ColumnMajor) {
            XCTAssertEqual(b[i], expected[k])
            k += 1
        }
    }
    
    func testBroadcast2() {
        let v = Tensor<CD>(colvector: [1, 2, 3])
        let b = broadcast(v, shape: Extent(3, 3))
        
        let expected = [1.0, 1.0, 1.0,
                        2.0, 2.0, 2.0,
                        3.0, 3.0, 3.0]
        
        var k = 0
        for i in b.indices(.ColumnMajor) {
            XCTAssertEqual(b[i], expected[k])
            k += 1
        }
    }
    
    func testBroadcast3() {
        let m = Tensor<CD>([[1, 2, 3], [4, 5, 6]])
        let b = broadcast(m, shape: Extent(2, 2, 3))
        
        let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        
        for (k, i) in b.indices(.ColumnMajor).enumerate() {
            XCTAssertEqual(b[i], expected[k])
        }
    }

    func testBenchmarkCBlas1() {
        let v1 = Tensor<CD>(shape: Extent(100000))
        
        self.measureBlock {
            let _ = v1 + v1
        }
    }
    
    func testBenchmarkNative() {
        let v1 = Tensor<D>(shape: Extent(100000))
        
        self.measureBlock {
            let _ = v1 + v1
        }
    }
}
