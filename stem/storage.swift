//
//  storage.swift
//  stem
//
//  Created by Abe Schneider on 11/11/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation

public enum MatrixOrder {
    case ColumnMajor, RowMajor
}

public protocol NumericType: AbsoluteValuable, FloatingPointType, Comparable {
    func +(lhs: Self, rhs: Self) -> Self
    func -(lhs: Self, rhs: Self) -> Self
    func *(lhs: Self, rhs: Self) -> Self
    func /(lhs: Self, rhs: Self) -> Self
    func %(lhs: Self, rhs: Self) -> Self

    init(_ v: Int)
}

extension Double: NumericType {}
extension Float: NumericType {}

public protocol Storage {
    typealias ElementType:NumericType
    
    var order:MatrixOrder { get }
    var shape:Extent { get }
//    var stride:[Int] { get }
    
    // creates new storage with given shape
    init(shape:Extent)
    init(array:[ElementType], shape:Extent)
    
    subscript(index:Int) -> ElementType {get set}
    
    func calculateStride(shape:Extent) -> [Int]
}
