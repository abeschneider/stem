//
//  Data.swift
//  stem
//
//  Created by Abraham Schneider on 11/28/16.
//
//

import Foundation
import Tensor

// want to figure out a way of tying together DataLoaders
// .. it would be nice if there was one DataLoader per stream,
// but we need a way of connecting operations between Input/Labels.
// However, we also want to allow for combinations like multiple Inputs/Labels
// and no Labels
//public struct DataIterator<T>: IteratorProtocol {
//    func shuffle() {}
//    func next() -> T {}
//}

/*
 Use-case
 
 data.shuffle()
 for (image, label) in data {
    train(image, label)
 }
 
 */

public protocol Shuffable {
    mutating func shuffle()
}

public protocol SupervisedData {
    associatedtype StorageType:Storage
    
    // TODO: consider making these functions that take as a parameter the batch size
    var imageSize:Extent { get }
    var labelSize:Extent { get }
    
    func next() -> (Tensor<StorageType>, Tensor<StorageType>)?
}
