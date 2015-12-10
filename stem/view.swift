//
//  view.swift
//  stem
//
//  Created by Abe Schneider on 11/11/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation

public struct StorageView<StorageType:Storage> {
    public var shape:Extent
    
    // offset within storage
    public var offset:[Int]
    
    public init(shape:Extent, offset:[Int]) {
        self.shape = shape
        self.offset = offset
    }
}
