//
//  tree.swift
//  stem
//
//  Created by Schneider, Abraham R. on 6/5/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation

//class Tree<Element: Comparable> {
//    let value: Element
//    // entries < value go on the left
//    let left: Tree<Element>?
//    // entries > value go on the right
//    let right: Tree<Element>?
//
//    init(value:Element, left:Tree<Element>?, right:Tree<Element>?) {
//        self.value = value
//        self.left = left
//        self.right = right
//    }
//}

//indirect enum Tree<S:Storage> {
//    case Empty
//    case Node(Tensor<S>, Tree<S>?, Tree<S>?)
//
//    init() {
//        self = .Empty
//    }
//
//    init(_ value:Tensor<S>, left:Tree<S>?, right:Tree<S>?) {
//        self = .Node(value, left, right)
//    }
//
//    init(_ value:Tensor<S>) {
//        self = .Node(value, nil, nil)
//    }
//}

/*
 
 ((A B) (C D))
 
 */
//public class RAE<S:Storage>: Op<S>, Collection {
//    public typealias StorageType = S
//    
//    public var ops:[Op<StorageType>] = []
//    
//    public init() {
//        
//    }
//}