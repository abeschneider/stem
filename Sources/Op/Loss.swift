//
//  loss.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/12/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation
import Tensor

public protocol Loss: Differentiable, OpType {
    associatedtype StorageType:Storage
    
    var value:StorageType.ElementType { get }
}
