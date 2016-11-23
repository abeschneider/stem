//
//  expr.swift
//  stem
//
//  Created by Abe Schneider on 12/23/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation

//class Operation {
//    
//}
//
//class Expression {
//    
//}

protocol Expression {
    typealias StorageType:Storage
    func eval() -> Tensor<StorageType>
}

class Variable<S:Storage>: Expression {
    typealias StorageType = S
    
    var value:Tensor<StorageType>
    
    init(value:Tensor<StorageType>) {
        self.value = value
    }
    
    func eval() -> Tensor<StorageType> {
        return value
    }
}

class UnaryExpression<S:Storage>: Expression {
    typealias StorageType = S
    
    func eval() -> Tensor<StorageType> {
        
    }
}