//: Playground - noun: a place where people can play

import Cocoa
import stem

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

func copy(from: [UInt8], offset: Int, to: Tensor<NativeStorage<Int>>) {
    var i = 0
    for index in to.indices() {
        to[index] = Int(from[i + offset])
        i += 1
    }
}

func toByteArray<T>(_ value: T) -> [UInt8] {
    var value = value
    return withUnsafeBytes(of: &value) { Array($0) }
}

func fromByteArray<T>(_ value: [UInt8], _: T.Type) -> T {
    return value.withUnsafeBytes {
        $0.baseAddress!.load(as: T.self)
    }
}

func loadData(filename:String) -> Tensor<NativeStorage<Int>>? {
    let data = NSData(contentsOfFile: filename)
    
    let header = [UInt32](repeating: 0, count: 4)
    data?.getBytes(UnsafeMutableRawPointer(mutating: header), length: 16)
    
    let magic = Int32(bigEndian: Int32(header[0]))
    let num_images = 5 //Int(Int32(bigEndian: Int32(header[1])))
    let num_rows = Int(Int32(bigEndian: Int32(header[2])))
    let num_cols = Int(Int32(bigEndian: Int32(header[3])))
    
    if magic != 2051 { return nil }
    
    let images = Tensor<NativeStorage<Int>>(Extent(Int(num_images), Int(num_rows), Int(num_cols)))
    let num_bytes = num_rows*num_cols*num_images
    let buffer = [UInt8](repeating: 0, count: Int(num_bytes))
    
    let ptr = UnsafeMutableRawPointer(mutating: buffer)
    data?.getBytes(ptr, range: NSRange(location: 16, length: num_bytes))

    // instead of copying, is it possible to set the storage to buffer and make each digit
    // a separate view?
    for i in 0..<num_images {
        let offset = i*(28*28)
        copy(from: buffer, offset: offset, to: images[i, all, all])
    }
    
    return images
}

// TODO: get rid of index .. just do a subview of the tensor
func makeImage(from:Tensor<NativeStorage<Int>>, index:Int) -> NSImage {
    let mem = from.storage.array.memory
    var idata = [UInt8](repeating: 0, count: 28*28)
    let offset = index*(28*28)
    for i in 0..<idata.count {
        idata[i] = UInt8(mem[i + offset])
    }
    let sdata = Data(bytes: idata)
    
    let im = CGImage(width: 28, height: 28, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: 28, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: 0), provider: CGDataProvider(data: sdata as CFData)!, decode: nil, shouldInterpolate: false, intent: CGColorRenderingIntent.defaultIntent)
    
    return NSImage(cgImage: im!, size: NSSize(width: 28, height: 28))
}


let image_filename = "/Users/ars3432/Downloads/train-images-idx3-ubyte"
let data = loadData(filename: image_filename)


for i in 0..<5 {
    let image = makeImage(from: data!, index: i)
}

//let imageView = NSImageView(frame: NSRect(x: 0, y: 0, width: 28, height: 28))
//imageView.image = dim
//if #available(OSX 10.12, *) {
//    let imageView = NSImageView(image: dim)
//} else {
//    // Fallback on earlier versions
//    print("not supported")
//}

//let backgroudView = NSView(frame: NSRect(x: 0, y: 0, width: 28, height: 28))
//backgroudView.addSubview(dim)

//let net = SequentialOp<D>()
//net.append(op: Conv2dOp(filterSize: Extent(3, 3)))
//net.append(op: PoolingOp(poolingSize: Extent(2, 2), stride: Extent(2, 2), evalFn: max))
//
////let data = loadData(filename: image_filename)
////print(data)
//
//let images = Tensor<NativeStorage<Int>>(Extent(Int(5), Int(10), Int(10)))
//print(images[0, all, all].shape)
