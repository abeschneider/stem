//
//  mnist.swift
//  stem
//
//  Created by Abraham Schneider on 11/27/16.
//
//

import Foundation
import AppKit
import Tensor
import Gzip


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

func readImageData(url:URL) -> Tensor<I>? {
    let zippedData = try! Data(contentsOf: url)
    print("read in \(zippedData.count) bytes")
    
    let data = try! zippedData.gunzipped()
    print("uncompressed to \(data.count) bytes")
    
    let headerCount = 4
    let headerSize = headerCount*MemoryLayout<UInt32>.size
    let header = [UInt32](repeating: 0, count: headerCount)
    let headerPtr:UnsafeMutablePointer<UInt32> = UnsafeMutablePointer(mutating: header)
    let _ = data.copyBytes(to: UnsafeMutableBufferPointer(start: headerPtr, count: headerCount), from: 0..<headerSize)
    
    let magic = Int32(bigEndian: Int32(header[0]))
    let numImages = Int(Int32(bigEndian: Int32(header[1])))
    let numRows = Int(Int32(bigEndian: Int32(header[2])))
    let numCols = Int(Int32(bigEndian: Int32(header[3])))
    
    if magic != 2051 { return nil }
    print("found \(numImages) images at resolution (\(numRows), \(numCols))")
    
    let numBytes = numRows*numCols*numImages
    let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: Int(numBytes))
    data.copyBytes(to: buffer, from: headerSize..<numBytes)
    let array = Array<UInt8>(UnsafeMutableRawBufferPointer(start: buffer, count: numBytes))
    let storage = I(array: array)
    return Tensor<I>(storage: storage, shape: Extent(numImages, numRows, numCols))
}

func readLabelData(url:URL) -> Tensor<I>? {
    let zippedData = try! Data(contentsOf: url)
    print("read in \(zippedData.count) bytes")
    
    let data = try! zippedData.gunzipped()
    print("uncompressed to \(data.count) bytes")
    
    let headerCount = 2
    let headerSize = headerCount*MemoryLayout<UInt32>.size
    let header = [UInt32](repeating: 0, count: headerCount)
    let headerPtr:UnsafeMutablePointer<UInt32> = UnsafeMutablePointer(mutating: header)
    let _ = data.copyBytes(to: UnsafeMutableBufferPointer(start: headerPtr, count: headerCount), from: 0..<headerSize)
    
    let magic = Int32(bigEndian: Int32(header[0]))
    let numLabels = Int(Int32(bigEndian: Int32(header[1])))
    
    if magic != 2049 { return nil }
    print("found \(numLabels) labels")
    
    let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: Int(numLabels))
    data.copyBytes(to: buffer, from: headerSize..<numLabels)
    let array = Array<UInt8>(UnsafeMutableRawBufferPointer(start: buffer, count: numLabels))
    let storage = I(array: array)
    return Tensor<I>(storage: storage, shape: Extent(numLabels))
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
