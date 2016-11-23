//
//  gradient.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/12/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation

public protocol GradientType: OpType {
    func reset()
}

public protocol Gradient: GradientType {
    associatedtype OpType
    
    init(op:OpType)
}

public protocol Differentiable {
    func gradient() -> GradientType
}

public func gradient(_ op:Differentiable) -> GradientType {
    return op.gradient()
}
