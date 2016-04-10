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
        var indices = self.indices()
        
        // TODO: look into simplifying this code
        if (!copyTransposed) {
            for i in 0..<rows {
                for j in 0..<cols {
                    let offset = calculateOffset(indices.next()!)
                    storage[offset] = array[i][j]
                }
            }
        } else {
            for j in 0..<cols {
                for i in 0..<rows {
                    let offset = calculateOffset(indices.next()!)
                    storage[offset] = array[i][j]
                }
            }
        }
    }
    
    public init(_ tensor:Tensor<StorageType>) {
        // verify we're being pass a vector
        assert(tensor.shape.span == 2)
        super.init(tensor)
    }
    
    public override init(storage:StorageType, shape:Extent, view:StorageView<StorageType>?=nil, offset:Int?=nil) {
        super.init(storage: storage, shape: shape, view: view, offset: offset)
    }
    
    public init(rows:Int, cols:Int) {
        super.init(shape: Extent(rows, cols))
    }
    
    public override init(shape:Extent) {
        assert(shape.count == 2)
        super.init(shape: shape)
    }
    
    public init(_ matrix:Matrix, dimIndex:[Int]?=nil, view:StorageView<StorageType>?=nil) {
        super.init(matrix, dimIndex: dimIndex, view: view)
    }
    
    public override func transpose() -> Matrix {
        let newDimIndex = Array(dimIndex.reverse())
        let newShape = Extent(view.shape.reverse())
        let newOffset = Array(view.offset.reverse())
        let newView = StorageView<StorageType>(shape: newShape, offset: newOffset)
        return Matrix(self, dimIndex: newDimIndex, view: newView)
    }
}
