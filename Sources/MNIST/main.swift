//
//  main.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/25/16.
//
//

// loads MNIST dataset and saves it in STEM format

import Foundation
import AppKit
import Tensor

typealias D = NativeStorage<Float>
typealias I = NativeStorage<UInt8>

func copy(from: [UInt8], offset: Int, to: Tensor<I>) {
    var i = 0
    for index in to.indices() {
        to[index] = UInt8(from[i + offset])
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

func loadData(filename:String) -> Tensor<I>? {
    let data = NSData(contentsOfFile: filename)
    
    let header = [UInt32](repeating: 0, count: 4)
    data?.getBytes(UnsafeMutableRawPointer(mutating: header), length: 16)
    
    let magic = Int32(bigEndian: Int32(header[0]))
    let num_images = Int(Int32(bigEndian: Int32(header[1])))
    let num_rows = Int(Int32(bigEndian: Int32(header[2])))
    let num_cols = Int(Int32(bigEndian: Int32(header[3])))
    
    if magic != 2051 { return nil }
    
    let num_bytes = num_rows*num_cols*num_images
    let buffer = [UInt8](repeating: 0, count: Int(num_bytes))
    
    let ptr = UnsafeMutableRawPointer(mutating: buffer)
    data?.getBytes(ptr, range: NSRange(location: 16, length: num_bytes))
    
    let storage = I(array: buffer)
    return Tensor<I>(storage: storage, shape: Extent(num_images, num_rows, num_cols))
}

// TODO: get rid of index .. just do a subview of the tensor
func makeImage(from:Tensor<I>, index:Int) -> NSImage {
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

func toImage(_ tensor:Tensor<NativeStorage<UInt8>>) -> NSImage {
    let width = tensor.shape[0]
    let height = tensor.shape[1]
    var bytes = [UInt8](repeating: 0, count: tensor.shape.elements)
    var k = 0
    for j in tensor.indices() {
        bytes[k] = tensor[j]
        k += 1
    }
    
    let data = Data(bytes: bytes)
    let im = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: 28, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: 0), provider: CGDataProvider(data: data as CFData)!, decode: nil, shouldInterpolate: false, intent: CGColorRenderingIntent.defaultIntent)
    
    return NSImage(cgImage: im!, size: NSSize(width: width, height: height))
}

func saveData(filename:String, images:Tensor<I>) throws {
//    let image_filename = "/Users/ars3432/Downloads/train-images-idx3-ubyte"
//    let images = loadData(filename: input)
    let data:Data = serialize(tensor: images)!
    try data.write(to: URL(fileURLWithPath: filename))
}


func parseOptions(args:[String], options:Dictionary<String, String>) {
    for arg in args {
        arg.self
        if (options[arg] != nil) {
            
        }
    }
}

let data = try Data(contentsOf: URL(fileURLWithPath: "mnist.bin"))
let images:Tensor<I> = deserialize(data: data)!

for i in 0..<10 {
    var imageTensor = images[i, all, all]
    imageTensor = imageTensor.reshape(Extent(imageTensor.shape[1], imageTensor.shape[2]))
    
    // NB: Bug here .. reshape causes the view to get reset
    print(imageTensor)
    let image = toImage(imageTensor)
    let data = image.tiffRepresentation!
    try data.write(to: URL(fileURLWithPath: "image_\(i).tiff"))
}

//if CommandLine.arguments[0] == "generated"
