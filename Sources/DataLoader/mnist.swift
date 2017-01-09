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

public typealias I = NativeStorage<UInt8>

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

/*
public struct MNIST: DataLoader {
    public var values:[String:Tensor<I>]
    
    public init(cache:String="mnist.bin") {
        values = [:]
        
        if !FileManager.default.fileExists(atPath: cache) {
            let trainImages = readImageData(url: URL(string: "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")!)!
            let trainLabels = readLabelData(url: URL(string: "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")!)!
            let testImages = readImageData(url: URL(string: "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")!)!
            let testLabels = readLabelData(url: URL(string: "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")!)!

            values = ["train_images": trainImages, "train_labels": trainLabels, "test_images":  testImages, "test_labels":  testLabels]
            let data:Data? = serialize(tensors:values)
            try data!.write(to: URL(fileURLWithPath: cache))
        } else {
            let data = try Data(contentsOf: URL(fileURLWithPath: cache))
            values = deserialize(data: data)
        }
    }
    
    public func next<S:Storage>() -> Tensor<S>? {
        return nil
    }
    
    public func get(_ name:String) -> Tensor<I>? {
        return values[name]!
    }

}*/

/*
 http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
 http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
 http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
 http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
 */
public class MNISTData: Sequence, IteratorProtocol, Shuffable, SupervisedData {
    public typealias StorageType = NativeStorage<Float>
    
    public var images:Tensor<I>
    public var labels:Tensor<I>
    public var indices:[Int]
    public var index:Int
    public var batchSize:Int
    
    public var imageSize:Extent { return Extent(batchSize, 28, 28) }
    public var labelSize:Extent { return Extent(batchSize) }
    
    public var count:Int { return images.shape[0] }
    
    public init(imageData:URL, labelData:URL, batchSize:Int, cache:String="mnist.bin") {
        self.batchSize = batchSize
        
        if !FileManager.default.fileExists(atPath: cache) {
            images = readImageData(url: imageData)!
            labels = readLabelData(url: labelData)!

            let values = ["images": images, "labels": labels]
            let data:Data? = serialize(tensors: values)
            try! data!.write(to: URL(fileURLWithPath: cache))
        } else {
            let data = try! Data(contentsOf: URL(fileURLWithPath: cache))
            let values:[String:Tensor<I>] = deserialize(data: data)
            images = values["images"]!
            labels = values["labels"]!
        }
        
        let count = images.shape[0]
        indices = [Int](0..<count)
        index = 0
    }
    
    public func shuffle() {
        indices.shuffled()
    }
    
    public func next() -> (Tensor<NativeStorage<Float>>, Tensor<NativeStorage<Float>>)? {
        let start:Int = index
        let end:Int = index+batchSize
        let sz:Int = end >= self.count ? self.count-1 : end
        
        let image = Tensor<I>(imageSize)
        let label = Tensor<I>(labelSize)
        for i in 0..<sz {
            let index = indices[i]
            image[i, all, all] = images[index, all, all]
            label[i] = labels[index]
        }
        
//        let image = images[indices[start..<end], all, all]
//        let label = labels[indices[start..<end]]
        index += batchSize
        if index >= indices.count {
            // reset for next time
            index = 0
            shuffle()
            
            return nil
        }
        
        let floatImage:Tensor<StorageType> = asType(image)
//        let floatLabel:Tensor<StorageType> = asType(label)
        let floatLabel:Tensor<StorageType> = asType(label)
//        let floatLabel = Tensor<NativeStorage<Float>>([labelValue])
        return (floatImage, floatLabel)
    }
}

