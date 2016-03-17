//
//  storage.swift
//  stem
//
//  Created by Abe Schneider on 11/11/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation

public protocol Storage {
    associatedtype ElementType:NumericType
    
    var size:Int { get }
    
    init(size:Int)
    init(array:[ElementType])
    init(storage:Self)
    init(storage:Self, copy:Bool)
    
    subscript(index:Int) -> ElementType {get set}
    
    // returns the stride for each dimension to traverse memory
    func calculateStride(shape:Extent) -> [Int]
}
