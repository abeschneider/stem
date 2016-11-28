//
//  main.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/25/16.
//
//

// loads MNIST dataset and saves it in STEM format

import Foundation
import Tensor

typealias D = NativeStorage<Float>
typealias I = NativeStorage<UInt8>

enum ANSIColors: String, CustomStringConvertible {
    case black = "\u{001B}[0;30m"
    case red = "\u{001B}[0;31m"
    case green = "\u{001B}[0;32m"
    case yellow = "\u{001B}[0;33m"
    case blue = "\u{001B}[0;34m"
    case magenta = "\u{001B}[0;35m"
    case cyan = "\u{001B}[0;36m"
    case white = "\u{001B}[0;37m"
    
    var description : String {
        return self.rawValue
    }
}

func saveData(filename:String, images:Tensor<I>) throws {
    let data:Data = serialize(tensor: images)!
    try data.write(to: URL(fileURLWithPath: filename))
    print("wrote \(data.count) bytes to \"\(filename)\"")
}

func parseOptions(args:[String]) -> [String:String] {
    var options = [String:String]()

    var i = 1
    while i < args.count-1 {
        let option = String(args[i].characters.dropFirst())
        let value = args[i+1]
        options[option] = value
        i += 2
    }
    
    return options
}

func generateImages(filename:String, path:String) throws {
    let data = try Data(contentsOf: URL(fileURLWithPath: filename))
    let datasets:[String:Tensor<I>] = deserialize(data: data)
    
    // just select training data for now
    let images = datasets["train_images"]!

    for i in 0..<images.shape[0] {
        var imageTensor = images[i, all, all]
        imageTensor = imageTensor.reshape(Extent(imageTensor.shape[1], imageTensor.shape[2]))
        
        let image = toImage(imageTensor)
        let data = image.tiffRepresentation!
        try data.write(to: URL(fileURLWithPath: "\(path)\\image_\(i).tiff"))
    }
}

func generateData(filename:String) throws{
    print("\(ANSIColors.yellow)loading training images...\(ANSIColors.white)")
    let trainImages = readImageData(url: URL(string: "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")!)!
    
    print("\(ANSIColors.yellow)loading training labels...\(ANSIColors.white)")
    let trainLabels = readLabelData(url: URL(string: "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")!)!
    
    print("\(ANSIColors.yellow)loading testing images...\(ANSIColors.white)")
    let testImages = readImageData(url: URL(string: "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")!)!
    
    print("\(ANSIColors.yellow)loading testing labels...\(ANSIColors.white)")
    let testLabels = readLabelData(url: URL(string: "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")!)!
    
    print("\(ANSIColors.yellow)serializing...\(ANSIColors.white)")
    let data:Data? = serialize(tensors:[
        "train_images": trainImages,
        "train_labels": trainLabels,
        "test_images":  testImages,
        "test_labels":  testLabels])
    
    print("\(ANSIColors.yellow)saving to \"\(filename)\"\(ANSIColors.white)")
    try data!.write(to: URL(fileURLWithPath: filename))
}

let options = parseOptions(args: CommandLine.arguments)

let mode = options["mode"] ?? "generate"
switch mode {
case "display":
    try generateImages(filename: options["data"]!, path: options["path"]!)
case "generate":
    try generateData(filename: options["out"]!)
default:
    try generateData(filename: options["out"]!)
}


