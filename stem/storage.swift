//
//  storage.swift
//  stem
//
//  Created by Abe Schneider on 11/11/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation

public enum DimensionOrder {
    case ColumnMajor
    case RowMajor
}

public protocol Storage {
    associatedtype ElementType:NumericType
    
    var size:Int { get }
    var order:DimensionOrder { get }
    
    init(size:Int, value:ElementType)
    init(array:[ElementType])
    init(storage:Self)
    init(storage:Self, copy:Bool)
    
    subscript(index:Int) -> ElementType {get set}
    
    func calculateOrder(dims:Int) -> [Int]
    func calculateOrder(values:[Int]) -> [Int]
}

public func calculateStride(shape:Extent) -> [Int] {
    var stride = Array<Int>(count:shape.count, repeatedValue: 0)
    
    var mult = 1
    stride[0] = 1
    
    var j = 0
    for i in 1..<shape.count {
        stride[i] = shape[i-1]*mult
        mult *= shape[i-1]
        j += 1
    }
    
    return stride
}
