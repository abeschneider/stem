//
//  storage.swift
//  stem
//
//  Created by Abe Schneider on 11/11/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation

public enum DimensionOrder {
    case columnMajor
    case rowMajor
}

public protocol Storage {
    associatedtype ElementType:NumericType
    
    var size:Int { get }
    var order:DimensionOrder { get }
    
    init(size:Int, value:ElementType)
    init(array:[ElementType])
    init(storage:Self, offset:Int)
    init(storage:Self, copy:Bool)
    
    subscript(index:Int) -> ElementType {get set}
    
    func calculateOrder(_ dims:Int) -> [Int]
    func calculateOrder(_ values:[Int]) -> [Int]
}

// TODO: move into Tensor.swift
public func calculateStride(_ shape:Extent) -> [Int] {
    var stride = Array<Int>(repeating: 0, count: shape.count)
    
    var mult = 1
    stride[0] = 1
    
    for i in 1..<shape.count {
        stride[i] = shape[i-1]*mult
        mult *= shape[i-1]
    }
    
    return stride
}
