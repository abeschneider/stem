//
//  vector.swift
//  stem
//
//  Created by Abe Schneider on 12/10/15.
//  Copyright © 2015 none. All rights reserved.
//

import Foundation

/*
The following classes add constraints to the Tensor class. This can
be useful for dispatching to functions based on those constraints
(e.g. if a function needs a RowVector, any type of Vector, or a
Matrix).
*/

// can be either row or column vector
public class Vector<StorageType:Storage>: Tensor<StorageType> {
    public init(_ array:[StorageType.ElementType], transposed:Bool=false) {
        super.init(array: array, shape: Extent(array.count))
        self.transposed = transposed
    }
    
    public init(_ tensor:Tensor<StorageType>) {
        assert(tensor.shape.span == 1)
        super.init(tensor)
    }
    
    public init(_ vector:Vector<StorageType>, dimIndex:[Int]?=nil, view:StorageView<StorageType>?=nil) {
        super.init(vector, dimIndex: dimIndex, view: view)
    }
    
    public init(rows:Int) {
        super.init(shape: Extent(rows))
    }
    
    public init(cols:Int) {
        super.init(shape: Extent(cols))
        transposed = true
    }
    
    public override init(storage:StorageType, shape:Extent, view:StorageView<StorageType>?=nil, offset:Int?=nil) {
        super.init(storage: storage, shape: shape, view: view, offset: offset)
    }
    
    public override func transpose() -> Vector {
        let newDimIndex = Array(dimIndex.reverse())
        let newShape = Extent(view.shape.reverse())
        let newOffset = Array(view.offset.reverse())
        let newView = StorageView<StorageType>(shape: newShape, offset: newOffset)
        return Vector(self, dimIndex: newDimIndex, view: newView)
    }
}

// constrained to be just a column vector
public class ColumnVector<StorageType:Storage>: Vector<StorageType> {
    public init(_ array:[StorageType.ElementType]) {
        super.init(array, transposed: false)
    }
    
    public override init(_ tensor:Tensor<StorageType>, dimIndex:[Int]?=nil, view:StorageView<StorageType>?=nil) {
        // verify we're being pass a vector
        assert(tensor.shape.span == 1)
        super.init(tensor)
    }
    
//    public override init(_ vector:Vector<StorageType>, dimIndex:[Int]?=nil, view:StorageView<StorageType>?=nil) {
//        super.init(vector, dimIndex: dimIndex, view: view)
//    }
    
    public override init(rows:Int) {
        super.init(rows: rows)
    }
    
//    public override init(storage:StorageType, shape:Extent, view:StorageView<StorageType>?=nil, offset:Int?=nil) {
//        super.init(storage: storage, shape: shape, view: view, offset: offset)
//    }
    
    public override func transpose() -> RowVector<StorageType> {
        let newDimIndex = Array(dimIndex.reverse())
        let newShape = Extent(view.shape.reverse())
        let newOffset = Array(view.offset.reverse())
        let newView = StorageView<StorageType>(shape: newShape, offset: newOffset)
        return RowVector<StorageType>(self, dimIndex: newDimIndex, view: newView)
    }
}

// constrained to be just a row vector
public class RowVector<StorageType:Storage>: Vector<StorageType> {
    public init(_ array:[StorageType.ElementType]) {
        super.init(array, transposed: true)
    }
    
    public override init(_ tensor:Tensor<StorageType>) {
        // verify we're being passed a vector
        assert(tensor.shape.span == 1)
        
        // verify the vector lies along the row-dimension
        assert(tensor.shape[0] > 0)
        
        super.init(tensor)
        transposed = true
    }
    
    public override init(_ vector:Vector<StorageType>, dimIndex:[Int]?=nil, view:StorageView<StorageType>?=nil) {
        super.init(vector, dimIndex: dimIndex, view: view)
    }
    
    public override init(storage:StorageType, shape:Extent, view:StorageView<StorageType>?=nil, offset:Int?=nil) {
        super.init(storage: storage, shape: shape, view: view, offset: offset)
    }
    
    public override init(cols:Int) {
        super.init(cols: cols)
    }
    
    public override func transpose() -> ColumnVector<StorageType> {
        let newDimIndex = Array(dimIndex.reverse())
        return ColumnVector<StorageType>(self, dimIndex: newDimIndex)
    }
}
