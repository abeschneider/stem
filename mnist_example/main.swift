//
//  main.swift
//  mnist_example
//
//  Created by Schneider, Abraham R. on 11/18/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation
import stem

typealias D = NativeStorage<Double>

func copy(from: Array<UInt8>, to: Tensor<NativeStorage<Int>>) {
    var i = 0
    for index in to.indices() {
        to[index] = Int(from[i])
        i += 1
    }
}

func loadData(filename:String) -> Tensor<NativeStorage<Int>>? {
    let data = NSData(contentsOfFile: filename)
    
    let header = Array<UInt32>(repeating: 0, count: 4)
    data?.getBytes(UnsafeMutableRawPointer(mutating: header), length: 4*8)
    
    let magic = Int32(bigEndian: Int32(header[0]))
    let num_images = Int(Int32(bigEndian: Int32(header[1])))
    let num_rows = Int(Int32(bigEndian: Int32(header[2])))
    let num_cols = Int(Int32(bigEndian: Int32(header[3])))
    
    if magic != 2051 { return nil }
    
    print("images: \(num_images), rows: \(num_rows), cols: \(num_cols)")
    
    let images = Tensor<NativeStorage<Int>>(Extent(Int(num_images), Int(num_rows), Int(num_cols)))
    let num_bytes = num_rows*num_cols
    let buffer = Array<UInt8>(repeating: 0, count: Int(num_bytes))
    
    print(images[0, all, all].shape)
    for i in 0..<num_images {
        data?.getBytes(UnsafeMutableRawPointer(mutating: buffer), length: Int(num_bytes))
//        copy(from: buffer, to: images[i, all, all])
    }
    
    return images
}




let image_filename = "/Users/ars3432/Downloads/train-images-idx3-ubyte"



let net = SequentialOp<D>()
net.append(op: Conv2dOp(filterSize: Extent(3, 3)))
net.append(op: PoolingOp(poolingSize: Extent(2, 2), stride: Extent(2, 2), evalFn: max))

//let data = loadData(filename: image_filename)
//print(data)

let images = Tensor<NativeStorage<Int>>(Extent(Int(5), Int(10), Int(10)))
print(images[0, all, all].shape)
