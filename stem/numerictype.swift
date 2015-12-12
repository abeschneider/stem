//
//  numerictype.swift
//  stem
//
//  Created by Abe Schneider on 12/11/15.
//  Copyright © 2015 none. All rights reserved.
//

import Foundation

public protocol NumericType: AbsoluteValuable, Comparable, FloatingPointType {
    func +(lhs: Self, rhs: Self) -> Self
    func -(lhs: Self, rhs: Self) -> Self
    func *(lhs: Self, rhs: Self) -> Self
    func /(lhs: Self, rhs: Self) -> Self
    func %(lhs: Self, rhs: Self) -> Self
    
    init(_ v:Double)
}

extension Float: NumericType {}
extension Double: NumericType {}

