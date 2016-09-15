//
//  stemTests.swift
//  stemTests
//
//  Created by Abe Schneider on 11/10/15.
//  Copyright © 2015 none. All rights reserved.
//

import XCTest
@testable import stem

typealias D = NativeStorage<Double>
typealias F = NativeStorage<Float>
typealias I = NativeStorage<Int>

class stemTests: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testTensorCreate() {
        let array:[Double] = [0, 1, 2, 3, 4,
                              5, 6, 7, 8, 9]
        let t = Tensor<D>(array: array, shape: Extent(2, 5))
        
        XCTAssertEqual(t.shape, Extent(2, 5))
        
        var k = 0
        for index in t.indices() {
            XCTAssertEqual(t[index], array[k])
            k += 1
        }
    }
    
    func testTensorView1() {
        let array = (0..<100).map { Double($0) }
        let t1 = Tensor<D>(array: array, shape: Extent(10, 10))
        let t2 = t1[1..<3, 1..<3]
        
        let expected:[Double] = [11, 12, 21, 22]
        
        var k = 0
        for i in t2.indices() {
            XCTAssertEqual(t2[i], expected[k])
            k += 1
        }
    }
    
    func testTensorView2() {
        let t1 = Tensor<D>(Extent(3, 5))

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
        
        let expected:[Double] = [
            1, 2, 1, 1, 1,
            0, 2, 3, 3, 3,
            0, 2, 3, 3, 3]
        
        var k = 0
        for i in t1.indices() {
            XCTAssertEqual(t1[i], expected[k])
            k += 1
        }
    }
    
    func testTensorTranspose() {
        let array = (0..<10).map { Double($0) }
        let t1 = Tensor<D>(array: array, shape: Extent(2, 5))
        let t2 = t1.transpose()
        
        // verify dimensions are correct
        XCTAssertEqual(t2.shape, Extent(5, 2))
        
        let expected:[Double] = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
        var k = 0
        
        for i in t2.indices() {
            XCTAssertEqual(t2[i], expected[k])
            k += 1
        }
        
        // as a sanity check, try transposing back
        let t3 = t2.transpose()
        XCTAssertEqual(t3.shape, Extent(2, 5))
        
        k = 0
        for i in t3.indices() {
            XCTAssertEqual(t3[i], array[k])
            k += 1
        }
    }
    
    func testTensorReshape() {
        let array = (0..<20).map { Double($0) }
        let tensor1 = Tensor<D>(array: array, shape: Extent(2, 10))
        let tensor2 = tensor1.reshape(Extent(4, 5))
        
        // verify dimensions are correct
        XCTAssertEqual(tensor2.shape, Extent(4, 5))

        // verify contents are still valid
        var k = 0
        for i in tensor2.indices() {
            XCTAssertEqual(tensor2[i], array[k])
            k += 1
        }
    }
    
    func testTensorExternalStorage() {
        let array = (0..<100).map { Double($0) }
        let tensor1 = Tensor<D>(array: array, shape: Extent(5, 10), offset: 0)
        let tensor2 = Tensor<D>(array: array, shape: Extent(5, 10), offset: 50)
        
        var k = 0
        for i in tensor1.indices() {
            XCTAssertEqual(tensor1[i], array[k])
            k += 1
        }
        
        for i in tensor2.indices() {
            XCTAssertEqual(tensor2[i], array[k])
            k += 1
        }
    }
    
    func testIndexGenerator1() {
        let shape = Extent(3, 2)
        
        let expected = [
            [0, 0],
            [1, 0],
            [2, 0],
            [0, 1],
            [1, 1],
            [2, 1]]
        
        for (i, index) in IteratorSequence(IndexGenerator(shape)).enumerated() {
            XCTAssertEqual(index, expected[i])
        }
        
        for (i, index) in IteratorSequence(IndexGenerator(shape, dimIndex: [0, 1])).enumerated() {
            XCTAssertEqual(index, expected[i])
        }
        
        for (i, index) in IteratorSequence(IndexGenerator(shape, order: .rowMajor)).enumerated() {
            XCTAssertEqual(index, expected[i])
        }
    }
    
    func testIndexGenerator2() {
        let shape = Extent(3, 2)
        
        let expected = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 0],
            [2, 1]
        ]
        
        for (i, index) in IteratorSequence(IndexGenerator(shape, dimIndex: [1, 0])).enumerated() {
            XCTAssertEqual(index, expected[i])
        }
        
        for (i, index) in IteratorSequence(IndexGenerator(shape, order: .columnMajor)).enumerated() {
            XCTAssertEqual(index, expected[i])
        }
    }
    
    func testStorageIndex1() {
        let array = (0..<20).map { Double($0) }
        let tensor = Tensor<D>(array: array, shape: Extent(2, 10))
        let indices = tensor.indices()
        
        for (i, index) in indices.enumerated() {
            let offset = tensor.calculateOffset(index)
            XCTAssertEqual(i, offset)
        }
    }
    
    func testStorageIndex2() {
        let array = (0..<100).map { Double($0) }
        let tensor1 = Tensor<D>(array: array, shape: Extent(10, 10))
        let tensor2 = tensor1[5..<10, 5..<10]
        var indices = tensor2.indices()

        let expected:[Int] = [
            55, 56, 57, 58, 59,
            65, 66, 67, 68, 69,
            75, 76, 77, 78, 79,
            85, 86, 87, 88, 89,
            95, 96, 97, 98, 99
        ]

        var i:Int = 0
        for j in 0..<tensor2.shape[0] {
            for k in 0..<tensor2.shape[1] {
                let idx = indices.next()!
                let offset = tensor2.calculateOffset(idx)
                XCTAssertEqual(offset, expected[i])
                XCTAssertEqual(tensor2[idx], tensor2[j, k])
                i += 1
            }
        }
    }
    
    func testStorageIndex3() {
        let array = (0..<15).map { Double($0) }
        let tensor = Tensor<D>(array: array, shape: Extent(3, 5))
        let subtensor = tensor[0, 0..<5]
        XCTAssertEqual(subtensor.shape.dims, [1, 5])
    }
    
    func testStorageIndex4() {
        let t = Tensor<D>(Extent(2, 3))
        let sub1 = t[0, all]
        let sub2 = t[all, 0]
        
        XCTAssertEqual(sub1.shape[1], 3)
        XCTAssertEqual(sub2.shape[0], 2)
    }
    
    func testStorageIndex5() {
        let m = Tensor<D>(Extent(2, 3))
        let sub1 = m[0, all]
        let sub2 = m[all, 0]
        
        XCTAssertEqual(sub1.shape[1], 3)
        XCTAssertEqual(sub2.shape[0], 2)
    }
    
    func testStorageIndex6() {
        let cube = Tensor<D>(Extent(3, 4, 5))
        let expected = (0..<cube.shape.elements).map { $0 }
        
        for (i, index) in IteratorSequence(IndexGenerator(cube.shape, order:.columnMajor)).enumerated() {
            let offset = cube.calculateOffset(index)
            XCTAssertEqual(offset, expected[i])
        }
    }
    
    func testCreateVector() {
        let array = (0..<20).map { Double($0) }
        let v = Tensor<D>(array)
        
        for i in 0..<v.shape[0] {
            XCTAssertEqual(v[i], array[i])
        }
    }
    
    func testCreateIntegerVector() {
        let array = (0..<20).map { Int($0) }
        let v = Tensor<I>(array)
        
        for i in 0..<v.shape[0] {
            XCTAssertEqual(v[i], array[i])
        }
    }
    
    func testCreateVectorFromMatrix() {
        let array = (0..<15).map { Double($0) }
        let t = Tensor<D>(array: array, shape: Extent(3, 5))
        
        let vector1 = t[0, 0..<5]
        let expected1:[Double] = [0, 1, 2, 3, 4]

        var k = 0
        for i in vector1.indices() {
            XCTAssertEqual(vector1[i], expected1[k])
            k += 1
        }

        let vector2 = t[0..<3, 1]
        let expected2:[Double] = [1, 6, 11]

        k = 0
        for i in vector2.indices() {
            XCTAssertEqual(vector2[i], expected2[k])
            k += 1
        }
        
        // verify transposition works on a vector created from a slice
        let vector3 = vector2.transpose()
        
        k = 0
        for i in vector3.indices() {
            // NB: expected value should not change from vector2
            XCTAssertEqual(vector3[i], expected2[k])
            k += 1
        }
    }
    
    func testCreateMatrix() {
        let array:[[Double]] = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        let m = Tensor<D>(array)
        
        XCTAssertEqual(m.shape, Extent(3, 4))

        // verify indexing works correctly
        for i in 0..<m.shape[0] {
            for j in 0..<m.shape[1] {
                XCTAssertEqual(m[i, j], array[i][j])
            }
        }
    }
    
    func testCreateMatrixFromMatrix() {
        let array:[[Double]] = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        let matrix1 = Tensor<D>(array)
        let matrix2 = matrix1[1..<3, 2..<4]
        
        let expected2:[[Double]] = [[6, 7],
                                    [10, 11]]
        
        for index in matrix2.indices() {
            XCTAssertEqual(matrix2[index], expected2[index[0]][index[1]])
        }
        
        let matrix3 = matrix2.transpose()
        let expected3:[[Double]] = [[6, 10],
                                    [7, 11]]

        for index in matrix3.indices() {
            XCTAssertEqual(matrix3[index], expected3[index[0]][index[1]])
        }
    }
    
    func testMatrixTranspose() {
        let array:[[Double]] = [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9]
        ]
        let matrix1 = Tensor<D>(array)
        let matrix2 = matrix1.transpose()
        
        // verify dimensions are correct
        XCTAssertEqual(matrix2.shape, Extent(5, 2))
        
        let expected:[Double] = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
        var k = 0
        
        for i in matrix2.indices() {
            XCTAssertEqual(matrix2[i], expected[k])
            k += 1
        }
        
        // as a sanity check, try transposing back
        let matrix3 = matrix2.transpose()
        XCTAssertEqual(matrix3.shape, Extent(2, 5))
        
        let expected2 = (0..<10).map { Double($0) }
        
        k = 0
        for i in matrix3.indices() {
            XCTAssertEqual(matrix3[i], expected2[k])
            k += 1
        }
    }
    
    func testMatrixTransposeOnReshape() {
        let array:[[Double]] = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11]
        ]
        
        let m = Tensor<D>(array)
        let ms = m.reshape(Extent(2,6))
        let mt = ms.transpose()
        
        XCTAssertEqual(mt.shape, Extent(6, 2))

        let expected:[Double] = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]

        var k = 0
        for i in 0..<mt.shape[0] {
            for j in 0..<mt.shape[1] {
                XCTAssertEqual(mt[i, j], expected[k])
                k += 1
            }
        }
    }
    
    func testVectorIndexing1() {
        let v = Tensor<D>([1, 2, 3, 4])
        let expected:[Double] = [1, 2, 3, 4]
        
        for i in 0..<v.shape[0] {
            XCTAssertEqual(v[i], expected[i])
        }
    }
    
    func testVectorIndexing2() {
        let v = Tensor<D>([1, 2, 3, 4, 5, 6, 7, 8])
        let expected = Tensor<D>([3, 4, 5, 6])
        
        let v2 = v[2..<6]
        XCTAssert(isClose(v2, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorIndexing3() {
        let v = Tensor<D>([1, 2, 3, 4, 5, 6, 7, 8])
        
        v[2..<6] = Tensor<D>([0, 0, 0, 0])
        
        let expected = Tensor<D>([1, 2, 0, 0, 0, 0, 7, 8])
        XCTAssert(isClose(v, expected, eps: 10e-4), "Not close")
    }
    
    func testMatrixIndexing1() {
        let m = Tensor<D>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let expected:[[Double]] = [[1, 2, 3, 4], [5, 6, 7, 8]]
        
        for i in 0..<m.shape[0] {
            for j in 0..<m.shape[1] {
                XCTAssertEqual(m[i, j], expected[i][j])
            }
        }
    }
    
    func testMatrixIndexing2() {
        let m = Tensor<D>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let expected = Tensor<D>([[2, 3], [6, 7]])
        
        let m2 = m[0..<2, 1..<3]
        XCTAssert(isClose(m2, expected, eps: 10e-4), "Not close")
    }
    
    func testMatrixIndexing3() {
        let m = Tensor<D>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let expected = Tensor<D>([[1, 0, 0, 4], [5, 0, 0, 8]])
        
        m[0..<2, 1..<3] = Tensor<D>([[0, 0], [0, 0]])
        
        XCTAssert(isClose(m, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorToString() {
        let v = Tensor<D>(rowvector: [1, 2, 3, 4, 5])
        
        let expected = "[[1.0,\t2.0,\t3.0,\t4.0,\t5.0]]"
        XCTAssertEqual(String(describing: v), expected)
    }
    
    func testMatrixToString() {
        let m = Tensor<D>([[1, 2, 3, 4], [5, 6, 7, 8]])
        
        // layout should be row-major
        let expected = "[[1.0,\t2.0,\t3.0,\t4.0]\n [5.0,\t6.0,\t7.0,\t8.0]]"
        XCTAssertEqual(String(describing: m), expected)
    }

    func testVectorAdd() {
        let m1 = Tensor<D>([1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = Tensor<D>([8, 7, 6, 5, 4, 3, 2, 1])
        let result = Tensor<D>(Extent(m1.shape[0]))
        
        add(m1, m2, result: result)
        
        let expected = Tensor<D>([9, 9, 9, 9, 9, 9, 9, 9])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorAdd2() {
        let m1 = Tensor<D>([1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = Tensor<D>([8, 7, 6, 5, 4, 3, 2, 1])
        
        let result = m1 + m2
        
        let expected = Tensor<D>([9, 9, 9, 9, 9, 9, 9, 9])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorAdd3() {
        let m1 = Tensor<D>([1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = Tensor<D>([8, 7, 6, 5, 4, 3, 2, 1])
        
        let result = m1 + m2 + m1 + m2
        
        let expected = Tensor<D>([18, 18, 18, 18, 18, 18, 18, 18])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testMatrixAdd1() {
        let m1 = Tensor<D>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let m2 = Tensor<D>([[8, 7, 6, 5], [4, 3, 2, 1]])
        let result = Tensor<D>(m1.shape)
        
        add(m1, m2, result: result)
        
        let expected = Tensor<D>([[9, 9, 9, 9], [9, 9, 9, 9]])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testMatrixVectorAdd1() {
        let m1 = Tensor<D>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let v1 = Tensor<D>(rowvector: [1, 1, 1, 1])
        let v2 = Tensor<D>(colvector: [0.5, 0.5])
        let result = Tensor<D>(m1.shape)
        
        add(m1, v1, result: result)
        iadd(result, v2)
        
        let expected = Tensor<D>([[2.5, 3.5, 4.5, 5.5], [6.5, 7.5, 8.5, 9.5]])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorSub() {
        let m1 = Tensor<D>([1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = Tensor<D>([8, 7, 6, 5, 4, 3, 2, 1])
        let result = Tensor<D>(m1.shape)
        
        sub(m1, m2, result: result)
        
        let expected = Tensor<D>([-7, -5, -3, -1, 1, 3, 5, 7])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorSub2() {
        let m1 = Tensor<D>([1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = Tensor<D>([8, 7, 6, 5, 4, 3, 2, 1])
        
        let result = m1 - m2
        
        let expected = Tensor<D>([-7, -5, -3, -1, 1, 3, 5, 7])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorScalarMul1() {
        let v = Tensor<D>([1, 2, 3, 4, 5, 6, 7, 8])
        let s:Double = 0.5
        let result = Tensor<D>(v.shape)
        
        mul(v, rhs: s, result: result)
        
        let expected = Tensor<D>([0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorScalarMul2() {
        let v = Tensor<D>([1, 2, 3, 4, 5, 6, 7, 8])
        let s:Double = 0.5
        let result = Tensor<D>(v.shape)
        
        mul(s, v, result: result)
        
        let expected = Tensor<D>([0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorScalarMul3() {
        let v = Tensor<D>([1, 2, 3, 4, 5, 6, 7, 8])
        let s:Double = 0.5
        let result = v*s
        
        let expected = Tensor<D>([0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorScalarMul4() {
        let v = Tensor<D>([1, 2, 3, 4, 5, 6, 7, 8])
        let s:Double = 0.5
        let result = s*v
        
        let expected = Tensor<D>([0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testMatrixScalarMul1() {
        let m = Tensor<D>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let result = 0.5*m
        
        let expected = Tensor<D>([[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0]])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testMatrixScalarMul2() {
        let m = Tensor<D>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let result = m*0.5
        
        let expected = Tensor<D>([[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0]])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testMatrixRowVectorMul1() {
        let m = Tensor<D>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let v = Tensor<D>(colvector: [2, 1])
        
        let result = m*v
        
        let expected = Tensor<D>([[2, 4, 6, 8], [5, 6, 7, 8]])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testDotProductVectorVector1() {
        let m = Tensor<D>([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        let v = Tensor<D>(colvector: [1, 2, 3])
        
        // 3x3*3x1 - > 3x1
        let result = Tensor<D>(Extent(3, 1))
        dot(m, v, result: result)
        
        let expected = Tensor<D>([1, 2, 3])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testDotProductVectorVector2() {
        let v1 = Tensor<D>(colvector: [1, 2, 3, 4])
        let v2 = Tensor<D>(rowvector: [2, 2, 2, 2])
        let v3 = v1+v1
        
        let result:Double = v2 ⊙ v3
        XCTAssertEqual(result, 40)
    }
    
    func testDotProductMatrixVector() {
        let m = Tensor<D>([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        let v = Tensor<D>(colvector: [1, 2, 3])
        
        let result:Tensor<D> = m ⊙ v
        
        let expected = Tensor<D>([1, 2, 3])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testDotProductMatrixMatrix() {
        let m = Tensor<D>([[1, 2, 3], [3, 4, 5]])
        let result:Tensor<D> = m ⊙ m.transpose()
        
        let expected = Tensor<D>([[14, 26], [26, 50]])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testOuterProduct1() {
        let v1 = Tensor<D>([1, 2, 3])
        let v2 = Tensor<D>([1, 2, 3])
        let result = Tensor<D>(Extent(3, 3))
        
        outer(v1, v2, result: result)
        
        let expected = Tensor<D>([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testOuterProduct2() {
        let v1 = Tensor<D>(colvector: [1, 2, 3, 4])
        let v2 = Tensor<D>(rowvector: [2, 2, 2, 2])
        let result = v1⊗v2
        
        let expected = Tensor<D>([  [  2,   2,  2,  2],
                                    [  4,   4,  4,  4],
                                    [  6,   6,  6,  6],
                                    [  8,   8,  8,  8]])
        
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorSum() {
        let v = Tensor<D>([1, 2, 3, 4, 5, 6, 7, 8, 9])
        let s = sum(v)
        XCTAssertEqual(s, 45.0)
    }
    
    func testVectorSum2() {
        let v = Tensor<D>(rowvector: [1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        // summing across the rows should give a vector
        let s = sum(v, axis: 0)
        let expected = Tensor<D>([1, 2, 3, 4, 5, 6, 7, 8, 9])
        XCTAssert(isClose(s, expected, eps: 10e-4))
        
        // summing across the columns should give a scalar
        let s2 = Double(sum(v, axis: 1))
        XCTAssertEqual(s2, 45.0)
    }
    
    func testVectorSum3() {
        let v = Tensor<D>(colvector: [1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        // summing across the columns should give a vector
        let s = sum(v, axis: 1)
        let expected = Tensor<D>([1, 2, 3, 4, 5, 6, 7, 8, 9])
        XCTAssert(isClose(s, expected, eps: 10e-4))
        
        // summing across the rows should give a scalar
        let s2 = Double(sum(v, axis: 0))
        XCTAssertEqual(s2, 45.0)
    }
    
    func testVectorMax() {
        let v = Tensor<D>([5, 1, 3, 2, 6, 10, 0, 42, 5, 2, 1])
        let s = max(v)
        XCTAssertEqual(s, 42.0)
    }
    
    func testMatrixSum1() {
        let v = Tensor<D>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let s = sum(v, axis: 0)
        let expected = Tensor<D>([6, 8, 10, 12])
        XCTAssert(isClose(s, expected, eps: 10e-4), "Not close")
    }
    
    func tesMatrixSum2() {
        let v = Tensor<D>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let s = sum(v, axis: 1)
        let expected = Tensor<D>([10, 26])
        XCTAssert(isClose(s, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorPower() {
        let v = Tensor<F>([1, 2, 3, 4, 5, 6, 7, 8, 9])
        let p = v**2.0
        
        let expected:[Float] = [1, 4, 9, 16, 25, 36, 49, 64, 81]
        
        var k = 0
        for i in p.indices() {
            XCTAssertEqual(p[i], expected[k])
            k += 1
        }
    }

    func testConcat1() {
        let v1 = Tensor<D>(rowvector: [1, 2, 3, 4])
        let v2 = Tensor<D>(rowvector: [5, 6, 7, 8])
        let result = concat(v1, v2, axis: 1)
        
        let expected = Tensor<D>(rowvector: [1, 2, 3, 4, 5, 6, 7, 8])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }

    func testConcat2() {
        let v1 = Tensor<D>(colvector: [1, 2, 3, 4])
        let v2 = Tensor<D>(colvector: [5, 6, 7, 8])
        let result = concat(v1, v2, axis: 0)
        
        let expected = Tensor<D>([[1, 2, 3, 4, 5, 6, 7, 8]])
        XCTAssert(isClose(result, expected.transpose(), eps: 10e-4), "Not close")
    }
    
//    func testConcat3() {
//        let v1 = ColumnVector<D>([1, 2, 3, 4])
//        let v2 = RowVector<D>([5, 6, 7, 8])
//        
//        var error = false
//        do {
//            concat(v1, v2, axis: 0)
//        } catch TensorError.SizeMismatch {
//            error = true
//        } catch {
//            XCTFail()
//        }
//        
//        if !error {
//            XCTFail()
//        }
//    }
    
    func testConcat4() {
        let v1 = Tensor<D>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let v2 = Tensor<D>([[9, 10, 11, 12], [13, 14, 15, 16]])
        let result = concat(v1, v2, axis: 0)
        
        let expected = Tensor<D>([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
//    func testConcat5() {
//        let v1 = ColumnVector<D>([1, 2, 3, 4])
//        let v2 = ColumnVector<D>([5, 6, 7, 8])
//        
////        var error = false
//        concat(v1, v2, axis: 1)
////        } catch TensorError.IllegalAxis {
////            error = true
////        } catch {
////            XCTFail()
////        }
////        
////        if !error {
////            XCTFail()
////        }
//    }
    
    func testConcat6() {
        let v1 = Tensor<D>(rowvector: [1, 2, 3, 4])
        let v2 = Tensor<D>(rowvector: [5, 6, 7, 8])
        
        let result = concat(v1, v2, axis: 1)
        
        let expected = Tensor<D>(rowvector: [1, 2, 3, 4, 5, 6, 7, 8])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
//    func testConcat7() {
//        let v1 = ColumnVector<D>([1, 2, 3, 4])
//        let v2 = RowVector<D>([5, 6, 7, 8])
//        
////        var error = false
////        do {
//        concat(v1, v2, axis: 0)
////        } catch TensorError.SizeMismatch {
////            error = true
////        } catch {
////            XCTFail()
////        }
////        
////        if !error {
////            XCTFail()
////        }
//    }
    
    func testConcat8() {
        let v1 = Tensor<D>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let v2 = Tensor<D>([[9, 10, 11, 12], [13, 14, 15, 16]])
        let result = concat(v1, v2, axis: 1)
        
        let expected = Tensor<D>([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16]])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testFloatFromTensor() {
        let t = Tensor<F>(Extent(1, 1))
        t[0, 0] = 1.0
        let f = Float(t)
        XCTAssertEqual(f, 1.0)
        
        let m = Tensor<F>([[2.0]])
        let f2 = Float(m)
        XCTAssertEqual(f2, 2.0)
    }

    func testBroadcast0() {
        let v1 = Tensor<D>(rowvector: [1, 2, 3]) // shape: (1, 3)
        let s1 = calculateBroadcastStride(v1, shape: Extent(3, 3))
        XCTAssertEqual(s1, [1, 0])

        let m1 = Tensor<D>([[1, 2, 3]]) // shape: (1, 3)
        let s3 = calculateBroadcastStride(m1, shape: Extent(3, 3))
        XCTAssertEqual(s3, [1, 0])
        
        let v2 = Tensor<D>(colvector: [1, 2, 3]) // shape: (3, 1)
        let s2 = calculateBroadcastStride(v2, shape: Extent(3, 3))
        XCTAssertEqual(s2, [0, 1])

        let m2 = Tensor<D>([[1], [2], [3]]) // shape: (3, 1)
        let s4 = calculateBroadcastStride(m2, shape: Extent(3, 3))
        XCTAssertEqual(s4, [0, 1])

        let m3 = Tensor<D>([[1, 2, 3], [4, 5, 6]]) // shape: (2, 3)
        let s5 = calculateBroadcastStride(m3, shape: Extent(2, 2, 3))
        XCTAssertEqual(s5, [1, 3, 0])
    }
    
    func testBroadcast1() {
        let v = Tensor<D>(rowvector: [1, 2, 3])
        let b = broadcast(v, shape: Extent(3, 3))
        
        let expected = [1.0, 2.0, 3.0,
                        1.0, 2.0, 3.0,
                        1.0, 2.0, 3.0]
        
        var k = 0
        for i in b.indices() {
            XCTAssertEqual(b[i], expected[k])
            k += 1
        }
    }
    
    func testBroadcast2() {
        let v = Tensor<D>(colvector: [1, 2, 3])
        let b = broadcast(v, shape: Extent(3, 3))
        
        let expected = [1.0, 1.0, 1.0,
                        2.0, 2.0, 2.0,
                        3.0, 3.0, 3.0]
        
        var k = 0
        for i in b.indices() {
            XCTAssertEqual(b[i], expected[k])
            k += 1
        }
    }
    
    func testBroadcast3() {
        let m = Tensor<D>([[1, 2, 3], [4, 5, 6]])
        let b = broadcast(m, shape: Extent(2, 2, 3))
        
        let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        
        for (k, i) in b.indices().enumerated() {
            XCTAssertEqual(b[i], expected[k])
        }
    }
    
    func testBroadcast4() {
        let m = Tensor<D>([[1, 2, 3], [4, 5, 6]])
        let v = Tensor<D>(rowvector: [1, 2, 3])
        
        let (m2, v2) = broadcast(m, v)
        
        XCTAssertEqual(m2.shape, v2.shape)
    }
    
//    func testBroadcastLast() {
////        let shape = Extent(1, 3)
////        let newShape = Extent(3, 3)
////        let t = Tensor<D>(shape: shape)
//        let v1 = RowVector<D>([1, 2, 3])
//        print(v1.shape.dims)
//        
//        let newStride1 = calculateBroadcastStride(v1, shape: Extent(3, 3))
//        print("newStride1 = \(newStride1)")
//        let b1 = Tensor<D>(tensor: v1, shape: Extent(3, 3), stride: newStride1)
//        print(String(b1))
//        
//        let v2 = ColumnVector<D>([1, 2, 3])
//        print(v2.shape.dims)
//        
//        let newStride2:[Int] = calculateBroadcastStride(v2, shape: Extent(3, 3))
//        print("newStride2 = \(newStride2)")
//        let b2 = Tensor<D>(tensor: v2, shape: Extent(3, 3), stride: newStride2)
//        print(String(b2))
//
////        for i in b.indices() {
////            print(b[i])
////        }
////        let v1:Double = b[0, 0]
////        let v2:Double = b[0, 1]
////        let v3:Double = b[1, 0]
//    }
    
//    func testConv2D() {
//        let image = Tensor<F>([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
//        let kernel = Tensor<F>([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
//        let result = conv2d(image, kernel: kernel)
//        print(result)
//        let expected = Tensor<F>([[-13, -20, -17], [-18, -24, -18], [13, 20, 17]])
//        XCTAssert(isClose(result, expected, eps: 10e-4))
//    }
//    
//    func testConv2DWithPadding() {
//        let image = Tensor<F>([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
//        let kernel = Tensor<F>([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
//        let result = conv2d(image, kernel: kernel, padding: [2, 2])
//        let expected = Tensor<F>([  [0.0,	0.0,	0.0,	0.0,	0.0],
//                                    [0.0,	-1.0,	-4.0,	-8.0,	0.0],
//                                    [0.0,	-4.0,	-13.0,	-20.0,	0.0],
//                                    [0.0,	-6.0,	-18.0,	-24.0,	0.0],
//                                    [0.0,	0.0,	0.0,	0.0,	0.0]])
//
//        print(result)
////        XCTAssert(isClose(result, expected, eps: 10e-4))
//    }


//    func testConv2DWithStride() {
//        let image = Tensor<F>([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
//        let kernel = Tensor<F>([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
//        let result = conv2d(image, kernel: kernel, padding: [3, 3], stride: [2, 2])
//        print(result)
////        let expected = Tensor<F>([[-13, -20, -17], [-18, -24, -18], [13, 20, 17]])
////        XCTAssert(isClose(result, expected, eps: 10e-4))
//    }

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
