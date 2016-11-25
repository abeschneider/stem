//
//  serialization.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/23/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation

public typealias TensorMessage = Stem_Serialize_tensor
public typealias TensorProperties = Stem_Serialize_header

//public func deserialize<S:Storage>(data:[UInt8]) -> Tensor<S> {
//    assertionFailure()
////    return nil
//    return Tensor<S>()
//}

public func fromProto<S:Storage>(msg:TensorMessage) -> Tensor<S> {
    let props:TensorProperties = msg.properties
    let shape = Extent(props.shape.map { Int($0) })
    
    let storage = msg.data.withUnsafeBytes { (ptr:UnsafePointer<UInt8>) -> S in
        ptr.withMemoryRebound(to: S.ElementType.self, capacity: shape.elements) {
            let buffer = UnsafeBufferPointer(start: $0, count: 25)
            return S(array: Array<S.ElementType>(buffer))
        }
    }
    
    return Tensor<S>(storage: storage, shape: shape)
}

public func deserialize<S:Storage>(data:[UInt8]) -> Tensor<S>? {
    do {
        let msg = try TensorMessage(protobufBytes: data)
        return fromProto(msg: msg)
    } catch {
        return nil
    }
}

public func deserialize<S:Storage>(data:Data) -> Tensor<S>? {
    do {
        let msg = try TensorMessage(protobuf: data)
        return fromProto(msg: msg)
    } catch {
        return nil
    }
}

public func toProto<E:NumericType>(tensor:Tensor<NativeStorage<E>>) -> TensorMessage {
    var msg = TensorMessage()
    msg.properties.shape = tensor.shape.dims.map { Int64($0) }
    msg.data = Data(buffer: UnsafeBufferPointer(start: &tensor.storage.array.memory, count: tensor.storage.array.memory.count))
    return msg
}

public func serialize<S:Storage>(tensor:Tensor<S>) -> [UInt8] {
    assertionFailure()
    return []
}

public func serialize<E:NumericType>(tensor:Tensor<NativeStorage<E>>) -> [UInt8] {
    var msg = TensorMessage()
    msg.properties.shape = tensor.shape.dims.map { Int64($0) }
    msg.data = Data(buffer: UnsafeBufferPointer(start: &tensor.storage.array.memory, count: tensor.storage.array.memory.count))

    do {
        return try msg.serializeProtobufBytes()
    } catch {
        return []
    }
}

public func serialize<E:NumericType>(tensor:Tensor<NativeStorage<E>>) -> Data? {
    var msg = TensorMessage()
    msg.properties.shape = tensor.shape.dims.map { Int64($0) }
    msg.data = Data(buffer: UnsafeBufferPointer(start: &tensor.storage.array.memory, count: tensor.storage.array.memory.count))
    
    do {
        return try msg.serializeProtobuf()
    } catch {
        return nil
    }
}

