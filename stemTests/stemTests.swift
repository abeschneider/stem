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
    
    func testTensorCreate() {
        let array:[Double] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        let tensor = Tensor<NativeStorage<Double>>(array: array, shape: Extent(2, 5))
        
        XCTAssertEqual(tensor.shape, Extent(2, 5))
        
        var k = 0
        for i in 0..<tensor.shape[0] {
            for j in 0..<tensor.shape[1] {
                XCTAssertEqual(tensor[i, j], array[k])
                k += 1
            }
        }
    }
    
    func testCBlasTensorCreate() {
        // NB: storage for CBlas follows a column-major format
        let array:[Double] = asColumnMajor((0..<10).map { Double($0) }, rows: 2, cols: 5)
        let tensor = Tensor<CBlasStorage<Double>>(array: array, shape: Extent(2, 5))
        
        XCTAssertEqual(tensor.shape, Extent(2, 5))

        let expected = (0..<10).map { Double($0) }
        
        var k = 0
        for i in 0..<tensor.shape[0] {
            for j in 0..<tensor.shape[1] {
                XCTAssertEqual(tensor[i, j], expected[k])
                k += 1
            }
        }
    }
    
    func testTensorView1() {
        let array = (0..<100).map { Double($0) }
        let tensor1 = Tensor<NativeStorage<Double>>(array: array, shape: Extent(10, 10))
        let tensor2 = tensor1[1..<3, 1..<3]
        
        let expected:[Double] = [11, 12, 21, 22]
        
        var k = 0
        for i in tensor2.storageIndices() {
            XCTAssertEqual(tensor2.storage[i], expected[k])
            k += 1
        }
    }
    
    func asColumnMajor(array:[Double], rows:Int, cols:Int) -> [Double] {
        let m = Tensor<NativeStorage<Double>>(array: array, shape: Extent(rows, cols))
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
    
    func testCBlasTensorView1() {
        let array = asColumnMajor((0..<100).map { Double($0) }, rows: 10, cols: 10)
        let tensor1 = Tensor<CBlasStorage<Double>>(array: array, shape: Extent(10, 10))
        let tensor2 = tensor1[1..<3, 1..<3]
        
        let expected:[Double] = [11, 12, 21, 22]
        
        var k = 0
        for i in tensor2.storageIndices() {
            XCTAssertEqual(tensor2.storage[i], expected[k])
            k += 1
        }
    }
    
    func testTensorView2() {
        let tensor1 = Tensor<NativeStorage<Double>>(shape: Extent(3, 5))

        // top row
        let tensor2 = tensor1[0, 0..<5]
        for i in tensor2.storageIndices() {
            tensor2.storage[i] = 1
        }
        
        // second column
        let tensor3 = tensor1[0..<3, 1]
        for i in tensor3.storageIndices() {
            tensor3.storage[i] = 2
        }
        
        // lower right area
        let tensor4 = tensor1[1..<3, 2..<5]
        for i in tensor4.storageIndices() {
            tensor4.storage[i] = 3
        }
        
        let expected:[Double] = [1, 2, 1, 1, 1,
                                 0, 2, 3, 3, 3,
                                 0, 2, 3, 3, 3]
        
        var k = 0
        for i in tensor1.storageIndices() {
            XCTAssertEqual(tensor1.storage[i], expected[k])
            k += 1
        }
    }
    
    func testCBlasTensorView2() {
        let tensor1 = Tensor<CBlasStorage<Double>>(shape: Extent(3, 5))
        
        // top row
        let tensor2 = tensor1[0, 0..<5]
        for i in tensor2.storageIndices() {
            tensor2.storage[i] = 1
        }
        
        // second column
        let tensor3 = tensor1[0..<3, 1]
        for i in tensor3.storageIndices() {
            tensor3.storage[i] = 2
        }
        
        // lower right area
        let tensor4 = tensor1[1..<3, 2..<5]
        for i in tensor4.storageIndices() {
            tensor4.storage[i] = 3
        }
        
        let expected:[Double] = [1, 2, 1, 1, 1,
                                 0, 2, 3, 3, 3,
                                 0, 2, 3, 3, 3]
        
        var k = 0
        for i in tensor1.storageIndices() {
            XCTAssertEqual(tensor1.storage[i], expected[k])
            k += 1
        }
    }
    
    func testTensorTranspose() {
        let array = (0..<10).map { Double($0) }
        let tensor1 = Tensor<NativeStorage<Double>>(array: array, shape: Extent(2, 5))
        let tensor2 = tensor1.transpose()
        
        // verify dimensions are correct
        XCTAssertEqual(tensor2.shape, Extent(5, 2))
        
        let expected:[Double] = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
        var k = 0
        
        for i in tensor2.storageIndices() {
            XCTAssertEqual(tensor2.storage[i], expected[k])
            k += 1
        }
        
        // as a sanity check, try transposing back
        let tensor3 = tensor2.transpose()
        XCTAssertEqual(tensor3.shape, Extent(2, 5))
        
        k = 0
        for i in tensor3.storageIndices() {
            XCTAssertEqual(tensor3.storage[i], array[k])
            k += 1
        }
    }
    
    func testCBlasTensorTranspose() {
        let array = asColumnMajor((0..<10).map { Double($0) }, rows: 2, cols: 5)
        let tensor1 = Tensor<CBlasStorage<Double>>(array: array, shape: Extent(2, 5))
        let tensor2 = tensor1.transpose()
        
        // verify dimensions are correct
        XCTAssertEqual(tensor2.shape, Extent(5, 2))
        
        let expected:[Double] = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
        var k = 0
        
        for i in tensor2.storageIndices() {
            XCTAssertEqual(tensor2.storage[i], expected[k])
            k += 1
        }
    }
    
    func testTensorReshape() {
        let array = (0..<20).map { Double($0) }
        let tensor1 = Tensor<NativeStorage<Double>>(array: array, shape: Extent(2, 10))
        let tensor2 = tensor1.reshape(Extent(4, 5))
        
        // verify dimensions are correct
        XCTAssertEqual(tensor2.shape, Extent(4, 5))

        // verify contents are still valid
        var k = 0
        for i in tensor2.storageIndices() {
            XCTAssertEqual(tensor2.storage[i], array[k])
            k += 1
        }
    }
    
    func testCBlasTensorReshape() {
        // TODO: check this is correct
        let array = asColumnMajor((0..<20).map { Double($0) }, rows: 4, cols: 5)
        let tensor1 = Tensor<CBlasStorage<Double>>(array: array, shape: Extent(2, 10))
        let tensor2 = tensor1.reshape(Extent(4, 5))
        
        // verify dimensions are correct
        XCTAssertEqual(tensor2.shape, Extent(4, 5))
        
        let expected = (0..<20).map { Double($0) }
        
        // verify contents are still valid
        var k = 0
        for i in tensor2.storageIndices() {
            XCTAssertEqual(tensor2.storage[i], expected[k])
            k += 1
        }
    }
    
    func testTensorExternalStorage() {
        let array = (0..<100).map { Double($0) }
        let tensor1 = Tensor<NativeStorage<Double>>(array: array, shape: Extent(5, 10), offset: 0)
        let tensor2 = Tensor<NativeStorage<Double>>(array: array, shape: Extent(5, 10), offset: 50)
        
        var k = 0
        for i in tensor1.storageIndices() {
            XCTAssertEqual(tensor1.storage[i], array[k])
            k += 1
        }
        
        for i in tensor2.storageIndices() {
            XCTAssertEqual(tensor2.storage[i], array[k])
            k += 1
        }
    }

    func testStorageIndex1() {
        let array = (0..<20).map { Double($0) }
        let tensor = Tensor<NativeStorage<Double>>(array: array, shape: Extent(2, 10))
        let indices = tensor.storageIndices()
        
        for (i, index) in GeneratorSequence(indices).enumerate() {
            XCTAssertEqual(i, index)
        }
    }
    
    func testStorageIndex2() {
        let array = (0..<100).map { Double($0) }
        let tensor1 = Tensor<NativeStorage<Double>>(array: array, shape: Extent(10, 10))
        let tensor2 = tensor1[5..<10, 5..<10]
        var indices = tensor2.storageIndices()

        let expected:[Int] = [  55, 56, 57, 58, 59,
                                65, 66, 67, 68, 69,
                                75, 76, 77, 78, 79,
                                85, 86, 87, 88, 89,
                                95, 96, 97, 98, 99]

        var i:Int = 0
        for j in 0..<tensor2.shape[0] {
            for k in 0..<tensor2.shape[1] {
                let idx = indices.next()!
                XCTAssertEqual(idx, expected[i])
                XCTAssertEqual(tensor2.storage[idx], tensor2[j, k])
                i += 1
            }
        }
    }
    
    func testStorageIndex3() {
        let array = (0..<15).map { Double($0) }
        let tensor = Tensor<NativeStorage<Double>>(array: array, shape: Extent(3, 5))
        let subtensor = tensor[0, 0..<5]
        XCTAssertEqual(subtensor.shape.dims, [1, 5])
    }
    
    func testCreateVector() {
        let array = (0..<20).map { Double($0) }
        let v = Vector<NativeStorage<Double>>(array)
        
        for i in 0..<v.shape[0] {
            XCTAssertEqual(v[i], array[i])
        }
    }
    
    func testCreateVectorFromMatrix() {
        let array = (0..<15).map { Double($0) }
        let tensor = Tensor<NativeStorage<Double>>(array: array, shape: Extent(3, 5))
        
        let vector1 = Vector<NativeStorage<Double>>(tensor[0, 0..<5])
        let expected1:[Double] = [0, 1, 2, 3, 4]

        var k = 0
        for i in vector1.storageIndices() {
            XCTAssertEqual(vector1.storage[i], expected1[k])
            k += 1
        }

        let vector2 = Vector<NativeStorage<Double>>(tensor[0..<3, 1])
        let expected2:[Double] = [1, 6, 11]

        k = 0
        for i in vector2.storageIndices() {
            XCTAssertEqual(vector2.storage[i], expected2[k])
            k += 1
        }
        
        // verify transposition works on a vector created from a slice
        let vector3 = vector2.transpose()
        
        k = 0
        for i in vector3.storageIndices() {
            // NB: expected value should not change from vector2
            XCTAssertEqual(vector3.storage[i], expected2[k])
            k += 1
        }
    }
    
    func testCreateMatrix() {
        let array:[[Double]] = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        let matrix = Matrix<NativeStorage<Double>>(array)
        
        XCTAssertEqual(matrix.shape, Extent(3, 4))

        // verify indexing works correctly
        for i in 0..<matrix.shape[0] {
            for j in 0..<matrix.shape[1] {
                XCTAssertEqual(matrix[i, j], array[i][j])
            }
        }
    }
    
    func testCreateMatrixFromMatrix() {
        let array:[[Double]] = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        let matrix1 = Matrix<NativeStorage<Double>>(array)
        let matrix2 = matrix1[1..<3, 2..<4]
        
        let expected2:[[Double]] = [[6, 7],
                                    [10, 11]]
        
        for i in 0..<matrix2.shape[0] {
            for j in 0..<matrix2.shape[1] {
                XCTAssertEqual(matrix2[i, j], expected2[i][j])
            }
        }
        
        let matrix3 = matrix2.transpose()
        
        let expected3:[[Double]] = [[6, 10],
                                    [7, 11]]
        
        for i in 0..<matrix3.shape[0] {
            for j in 0..<matrix3.shape[1] {
                XCTAssertEqual(matrix3[i, j], expected3[i][j])
            }
        }
    }
    
    func testMatrixTranspose() {
        let array:[[Double]] = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        let matrix1 = Matrix<NativeStorage<Double>>(array)
        let matrix2 = matrix1.transpose()
        
        // verify dimensions are correct
        XCTAssertEqual(matrix2.shape, Extent(5, 2))
        
        let expected:[Double] = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
        var k = 0
        
        for i in matrix2.storageIndices() {
            XCTAssertEqual(matrix2.storage[i], expected[k])
            k += 1
        }
        
        // as a sanity check, try transposing back
        let matrix3 = matrix2.transpose()
        XCTAssertEqual(matrix3.shape, Extent(2, 5))
        
        let expected2 = (0..<10).map { Double($0) }
        
        k = 0
        for i in matrix3.storageIndices() {
            XCTAssertEqual(matrix3.storage[i], expected2[k])
            k += 1
        }
    }
    
    func testMatrixTransposeOnReshape() {
        let array:[[Double]] = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        let m = Matrix<NativeStorage<Double>>(array)
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
        let v = Vector<NativeStorage<Double>>([1, 2, 3, 4])
        let expected:[Double] = [1, 2, 3, 4]
        
        for i in 0..<v.shape[0] {
            XCTAssertEqual(v[i], expected[i])
        }
    }
    
    func testVectorIndexing2() {
        let v = Vector<NativeStorage<Double>>([1, 2, 3, 4, 5, 6, 7, 8])
        let expected = Vector<NativeStorage<Double>>([3, 4, 5, 6])
        
        let v2 = v[2..<6]
        XCTAssert(isClose(v2, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorIndexing3() {
        let v = Vector<NativeStorage<Double>>([1, 2, 3, 4, 5, 6, 7, 8])
        
        v[2..<6] = Vector<NativeStorage<Double>>([0, 0, 0, 0])
        
        let expected = Vector<NativeStorage<Double>>([1, 2, 0, 0, 0, 0, 7, 8])
        XCTAssert(isClose(v, expected, eps: 10e-4), "Not close")
    }
    
    func testMatrixIndexing1() {
        let m = Matrix<NativeStorage<Double>>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let expected:[[Double]] = [[1, 2, 3, 4], [5, 6, 7, 8]]
        
        for i in 0..<m.shape[0] {
            for j in 0..<m.shape[1] {
                XCTAssertEqual(m[i, j], expected[i][j])
            }
        }
    }
    
    func testMatrixIndexing2() {
        let m = Matrix<NativeStorage<Double>>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let expected = Matrix<NativeStorage<Double>>([[2, 3], [6, 7]])
        
        let m2 = m[0..<2, 1..<3]
        XCTAssert(isClose(m2, expected, eps: 10e-4), "Not close")
    }
    
    func testMatrixIndexing3() {
        let m = Matrix<NativeStorage<Double>>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let expected = Matrix<NativeStorage<Double>>([[1, 0, 0, 4], [5, 0, 0, 8]])
        
        m[0..<2, 1..<3] = Matrix<NativeStorage<Double>>([[0, 0], [0, 0]])
        XCTAssert(isClose(m, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorToString() {
        let v = Vector<CBlasStorage<Double>>([1, 2, 3, 4, 5])
        
        let expected = "[1.0,\t2.0,\t3.0,\t4.0,\t5.0]"
        XCTAssertEqual(String(v), expected)
    }
    
    func testMatrixToString() {
        let m = Matrix<NativeStorage<Double>>([[1, 2, 3, 4], [5, 6, 7, 8]])
        
        // layout should be row-major
        let expected = "[[1.0,\t2.0,\t3.0,\t4.0]\n [5.0,\t6.0,\t7.0,\t8.0]]"
        XCTAssertEqual(String(m), expected)
    }

    func testVectorAdd() {
        let m1 = Vector<NativeStorage<Double>>([1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = Vector<NativeStorage<Double>>([8, 7, 6, 5, 4, 3, 2, 1])
        let result = Vector<NativeStorage<Double>>(rows: m1.shape[0])
        
        try! add(left: m1, right: m2, result: result)
        
        let expected = Vector<NativeStorage<Double>>([9, 9, 9, 9, 9, 9, 9, 9])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testCBlasVectorAdd() {
        let m1 = Vector<CBlasStorage<Double>>([1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = Vector<CBlasStorage<Double>>([8, 7, 6, 5, 4, 3, 2, 1])
        let result = Vector<CBlasStorage<Double>>(rows: m1.shape[0])
        
        try! add(left: m1, right: m2, result: result)
        
        let expected = Vector<CBlasStorage<Double>>([9, 9, 9, 9, 9, 9, 9, 9])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorAdd2() {
        let m1 = Vector<NativeStorage<Double>>([1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = Vector<NativeStorage<Double>>([8, 7, 6, 5, 4, 3, 2, 1])
        
        let result = try! m1 + m2
        
        let expected = Vector<NativeStorage<Double>>([9, 9, 9, 9, 9, 9, 9, 9])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorAdd3() {
        let m1 = Vector<NativeStorage<Double>>([1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = Vector<NativeStorage<Double>>([8, 7, 6, 5, 4, 3, 2, 1])
        
        let result = try! m1 + m2 + m1 + m2
        
        let expected = Vector<NativeStorage<Double>>([18, 18, 18, 18, 18, 18, 18, 18])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testCBlasVectorAdd2() {
        let m1 = Vector<CBlasStorage<Double>>([1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = Vector<CBlasStorage<Double>>([8, 7, 6, 5, 4, 3, 2, 1])
        
        let result = try! m1 + m2
        
        let expected = Vector<CBlasStorage<Double>>([9, 9, 9, 9, 9, 9, 9, 9])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testMatrixAdd1() {
        let m1 = Matrix<NativeStorage<Double>>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let m2 = Matrix<NativeStorage<Double>>([[8, 7, 6, 5], [4, 3, 2, 1]])
        let result = Matrix<NativeStorage<Double>>(rows: m1.shape[0], cols: m1.shape[1])
        
        try! add(left: m1, right: m2, result: result)
        
        let expected = Matrix<NativeStorage<Double>>([[9, 9, 9, 9], [9, 9, 9, 9]])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testCBlasMatrixAdd1() {
        let m1 = Matrix<CBlasStorage<Double>>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let m2 = Matrix<CBlasStorage<Double>>([[8, 7, 6, 5], [4, 3, 2, 1]])
        let result = Matrix<CBlasStorage<Double>>(rows: m1.shape[0], cols: m1.shape[1])
        
        try! add(left: m1, right: m2, result: result)
        
        let expected = Matrix<CBlasStorage<Double>>([[9, 9, 9, 9], [9, 9, 9, 9]])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testMatrixVectorAdd1() {
        let m1 = Matrix<NativeStorage<Double>>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let v1 = RowVector<NativeStorage<Double>>([1, 1, 1, 1])
        let v2 = ColumnVector<NativeStorage<Double>>([0.5, 0.5])
        let result = Matrix<NativeStorage<Double>>(rows: m1.shape[0], cols: m1.shape[1])
        
        try! add(left: m1, right: v1, result: result)
        try! iadd(left: result, right: v2)
        
        let expected = Matrix<NativeStorage<Double>>([[2.5, 3.5, 4.5, 5.5], [6.5, 7.5, 8.5, 9.5]])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testCBlasMatrixVectorAdd1() {
        let m = Matrix<CBlasStorage<Double>>([[1, 2, 3, 4], [5, 6, 7, 8]])
        
        let v1 = RowVector<CBlasStorage<Double>>([1, 1, 1, 1])
        let v2 = ColumnVector<CBlasStorage<Double>>([0.5, 0.5])
        let result = Matrix<CBlasStorage<Double>>(rows: m.shape[0], cols: m.shape[1])
        
        try! add(left: m, right: v1, result: result)
        try! iadd(left: result, right: v2)
        
        let expected = Matrix<CBlasStorage<Double>>([[2.5, 3.5, 4.5, 5.5], [6.5, 7.5, 8.5, 9.5]])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorSub() {
        let m1 = Vector<NativeStorage<Double>>([1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = Vector<NativeStorage<Double>>([8, 7, 6, 5, 4, 3, 2, 1])
        let result = Vector<NativeStorage<Double>>(rows: m1.shape[0])
        
        sub(left: m1, right: m2, result: result)
        
        let expected = Vector<NativeStorage<Double>>([-7, -5, -3, -1, 1, 3, 5, 7])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testCBlasVectorSub() {
        let m1 = Vector<CBlasStorage<Double>>([1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = Vector<CBlasStorage<Double>>([8, 7, 6, 5, 4, 3, 2, 1])
        let result = Vector<CBlasStorage<Double>>(rows: m1.shape[0])
        
        sub(left: m1, right: m2, result: result)
        
        let expected = Vector<CBlasStorage<Double>>([-7, -5, -3, -1, 1, 3, 5, 7])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorSub2() {
        let m1 = Vector<NativeStorage<Double>>([1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = Vector<NativeStorage<Double>>([8, 7, 6, 5, 4, 3, 2, 1])
        
        let result = m1 - m2
        
        let expected = Vector<NativeStorage<Double>>([-7, -5, -3, -1, 1, 3, 5, 7])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testCBlasVectorSub2() {
        let m1 = Vector<CBlasStorage<Double>>([1, 2, 3, 4, 5, 6, 7, 8])
        let m2 = Vector<CBlasStorage<Double>>([8, 7, 6, 5, 4, 3, 2, 1])
        
        let result = m1 - m2
        
        let expected = Vector<CBlasStorage<Double>>([-7, -5, -3, -1, 1, 3, 5, 7])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorScalarMul1() {
        let v = Vector<NativeStorage<Double>>([1, 2, 3, 4, 5, 6, 7, 8])
        let s:Double = 0.5
        let result = Vector<NativeStorage<Double>>(rows: v.shape[0])
        
        mul(left: v, right: s, result: result)
        
        let expected = Vector<NativeStorage<Double>>([0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorScalarMul2() {
        let v = Vector<NativeStorage<Double>>([1, 2, 3, 4, 5, 6, 7, 8])
        let s:Double = 0.5
        let result = Vector<NativeStorage<Double>>(rows: v.shape[0])
        
        mul(left: s, right: v, result: result)
        
        let expected = Vector<NativeStorage<Double>>([0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorScalarMul3() {
        let v = Vector<NativeStorage<Double>>([1, 2, 3, 4, 5, 6, 7, 8])
        let s:Double = 0.5
        let result = v*s
        
        let expected = Vector<NativeStorage<Double>>([0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testVectorScalarMul4() {
        let v = Vector<NativeStorage<Double>>([1, 2, 3, 4, 5, 6, 7, 8])
        let s:Double = 0.5
        let result = s*v
        
        let expected = Vector<NativeStorage<Double>>([0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testMatrixScalarMul1() {
        let m = Matrix<NativeStorage<Double>>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let result = 0.5*m
        
        let expected = Matrix<NativeStorage<Double>>([[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0]])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testMatrixScalarMul2() {
        let m = Matrix<NativeStorage<Double>>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let result = m*0.5
        
        let expected = Matrix<NativeStorage<Double>>([[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0]])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testMatrixRowVectorMul1() {
        let m = Matrix<NativeStorage<Double>>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let v = ColumnVector<NativeStorage<Double>>([2, 1])
        
        let result = try! m*v
        
        let expected = Matrix<NativeStorage<Double>>([[2, 4, 6, 8], [5, 6, 7, 8]])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testDotProduct1() {
        let m = Matrix<NativeStorage<Double>>([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        let v = ColumnVector<NativeStorage<Double>>([1, 2, 3])
        
        let result = ColumnVector<NativeStorage<Double>>(rows: 3)
        try! dot(left: m, right: v, result: result)
        
        let expected = Vector<NativeStorage<Double>>([1, 2, 3])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testDotProduct2() {
        let m = Matrix<NativeStorage<Double>>([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        let v = ColumnVector<NativeStorage<Double>>([1, 2, 3])
        
        let result = try! m⋅v
        
        let expected = Vector<NativeStorage<Double>>([1, 2, 3])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testDotProduct3() {
        let v1 = ColumnVector<NativeStorage<Double>>([1, 2, 3, 4])
        let v2 = RowVector<NativeStorage<Double>>([2, 2, 2, 2])
        let v3 = try! v1+v1

        let result = try! v2⋅v3
        XCTAssertEqual(result, 40)
    }
    
    func testOuterProduct1() {
        let v1 = Vector<NativeStorage<Double>>([1, 2, 3])
        let v2 = Vector<NativeStorage<Double>>([1, 2, 3])
        let result = Matrix<NativeStorage<Double>>(rows: 3, cols: 3)
        
        outer(left: v1, right: v2, result: result)
        
        let expected = Matrix<NativeStorage<Double>>([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testOuterProduct2() {
        let v1 = ColumnVector<NativeStorage<Double>>([1, 2, 3, 4])
        let v2 = RowVector<NativeStorage<Double>>([2, 2, 2, 2])
        let v3 = try! v1+v1
        
        let expected = Matrix<NativeStorage<Double>>([
            [  4,   8,  12,  16],
            [  4,   8,  12,  16],
            [  4,   8,  12,  16],
            [  4,   8,  12,  16]])
        
        let result = try! v2⨯v3
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testNativeVectorSum() {
        let v = Vector<NativeStorage<Double>>([1, 2, 3, 4, 5, 6, 7, 8, 9])
        let s = sum(v)
        XCTAssertEqual(s, 45.0)
    }
    
    func testNativeVectorPower() {
        let v = Vector<NativeStorage<Double>>([1, 2, 3, 4, 5, 6, 7, 8, 9])
        let p = v**2.0
        
        let expected:[Double] = [1, 4, 9, 16, 25, 36, 49, 64, 81]
        
        var k = 0
        for i in p.storageIndices() {
            XCTAssertEqual(p.storage[i], expected[k])
            k += 1
        }
    }

    func testConcat1() {
        let v1 = RowVector<NativeStorage<Double>>([1, 2, 3, 4])
        let v2 = RowVector<NativeStorage<Double>>([5, 6, 7, 8])
        let result = try! concat(v1, v2, axis: 0)
        
        let expected = RowVector<NativeStorage<Double>>([1, 2, 3, 4, 5, 6, 7, 8])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }

    func testConcat2() {
        let v1 = ColumnVector<NativeStorage<Double>>([1, 2, 3, 4])
        let v2 = ColumnVector<NativeStorage<Double>>([5, 6, 7, 8])
        let result = try! concat(v1, v2, axis: 0)
        
        let expected = Matrix<NativeStorage<Double>>([[1, 2, 3, 4], [5, 6, 7, 8]])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testConcat3() {
        let v1 = ColumnVector<NativeStorage<Double>>([1, 2, 3, 4])
        let v2 = RowVector<NativeStorage<Double>>([5, 6, 7, 8])
        
        var error = false
        do {
            try concat(v1, v2, axis: 0)
        } catch TensorError.SizeMismatch {
            error = true
        } catch {
            XCTFail()
        }
        
        if !error {
            XCTFail()
        }
    }
    
    func testConcat4() {
        let v1 = Matrix<NativeStorage<Double>>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let v2 = Matrix<NativeStorage<Double>>([[9, 10, 11, 12], [13, 14, 15, 16]])
        let result = try! concat(v1, v2, axis: 0)
        
        let expected = Matrix<NativeStorage<Double>>([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testConcat5() {
        let v1 = ColumnVector<NativeStorage<Double>>([1, 2, 3, 4])
        let v2 = ColumnVector<NativeStorage<Double>>([5, 6, 7, 8])
        
        var error = false
        do {
            try concat(v1, v2, axis: 1)
        } catch TensorError.IllegalAxis {
            error = true
        } catch {
            XCTFail()
        }
        
        if !error {
            XCTFail()
        }
    }
    
    func testConcat6() {
        let v1 = RowVector<NativeStorage<Double>>([1, 2, 3, 4])
        let v2 = RowVector<NativeStorage<Double>>([5, 6, 7, 8])
        
        let result = try! concat(v1, v2, axis: 1)
        let expected = ColumnVector<NativeStorage<Double>>([1, 2, 3, 4, 5, 6, 7, 8])
        XCTAssert(isClose(result, expected, eps: 10e-4), "Not close")
    }
    
    func testConcat7() {
        let v1 = ColumnVector<NativeStorage<Double>>([1, 2, 3, 4])
        let v2 = RowVector<NativeStorage<Double>>([5, 6, 7, 8])
        
        var error = false
        do {
            try concat(v1, v2, axis: 0)
        } catch TensorError.SizeMismatch {
            error = true
        } catch {
            XCTFail()
        }
        
        if !error {
            XCTFail()
        }
    }
    
    func testConcat8() {
        let v1 = Matrix<NativeStorage<Double>>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let v2 = Matrix<NativeStorage<Double>>([[9, 10, 11, 12], [13, 14, 15, 16]])
        let result = try! concat(v1, v2, axis: 1)
        
        let expected = Matrix<NativeStorage<Double>>([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
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
