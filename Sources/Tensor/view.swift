//
//  view.swift
//  stem
//
//  Created by Abe Schneider on 11/11/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation

public struct StorageView<StorageType:Storage> {
    // the dimensions of the Tensor
    public var shape:Extent
    
    // offset within storage
    public var offset:[Int]
    
    public init(shape:Extent, offset:[Int]?=nil) {
        self.shape = shape
        self.offset = offset ?? [Int](repeating: 0, count: shape.count)
    }
}
