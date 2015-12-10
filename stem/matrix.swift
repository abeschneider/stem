//
//  matrix.swift
//  stem
//
//  Created by Abe Schneider on 12/10/15.
//  Copyright © 2015 none. All rights reserved.
//

import Foundation

public class Matrix<StorageType:Storage>: Tensor<StorageType> {
    public init(_ array:[[StorageType.ElementType]], copyTransposed:Bool=false) {
        let rows = array.count
        let cols = array[0].count
        
        if (!copyTransposed) {
            super.init(shape: Extent(rows, cols))
        } else {
            super.init(shape: Extent(cols, rows))
        }
        
        // copy array
        var indices = storageIndices()
        
        // TODO: look into simplifying this code
        if (!copyTransposed) {
            for i in 0..<rows {
                for j in 0..<cols {
                    let index = indices.next()!
                    storage[index] = array[i][j]
                }
            }
        } else {
            for j in 0..<cols {
                for i in 0..<rows {
                    let index = indices.next()!
                    storage[index] = array[i][j]
                }
            }
        }
    }
    
    public init(rows:Int, cols:Int) {
        super.init(shape: Extent(rows, cols))
    }
    
    public init(_ matrix:Matrix, dimIndex:[Int]?=nil) {
        super.init(matrix, dimIndex: dimIndex)
    }
    
    public override func transpose() -> Matrix {
        let newDimIndex = Array(dimIndex.reverse())
        return Matrix(self, dimIndex: newDimIndex)
    }
}
