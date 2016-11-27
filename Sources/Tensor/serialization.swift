//
//  serialization.swift
//  stem
//
//  Created by Schneider, Abraham R. on 11/23/16.
//  Copyright © 2016 none. All rights reserved.
//

import Foundation

public typealias TensorMessage = Stem_Serialize_Tensor
public typealias TensorProperties = Stem_Serialize_Properties
public typealias TensorView = Stem_Serialize_View
public typealias TensorDictionary = Stem_Serialize_Dictionary

public func toProto<E:NumericType>(tensor:Tensor<NativeStorage<E>>) -> TensorMessage {
    var msg = TensorMessage()
    msg.view.shape = tensor.view.shape.dims.map { Int64($0) }
    msg.view.offset = tensor.view.offset.map  { Int64($0) }
    msg.properties.type = String(describing: E.self)
    msg.properties.stride = tensor.stride.map { Int64($0) }
    msg.properties.dimIndex = tensor.dimIndex.map { Int64($0) }
    msg.storage_.size = Int64(tensor.storage.array.memory.count)
    msg.storage_.data = Data(buffer: UnsafeBufferPointer(start: &tensor.storage.array.memory, count: tensor.storage.array.memory.count))
    return msg
}

public func toProto<E:NumericType>(tensors:[String:Tensor<NativeStorage<E>>]) -> TensorDictionary {
    var msg = TensorDictionary()
    
    for (key, tensor) in tensors {
        msg.items[key] = toProto(tensor: tensor)
    }
    
    return msg
}

public func fromProto<S:Storage>(msg:TensorMessage) -> Tensor<S> {
    let props:TensorProperties = msg.properties
    let type = props.type
    let stride = props.stride.map { Int($0) }
    let dimIndex = props.dimIndex.map { Int($0) }
    
    let view:TensorView = msg.view
    let shape = Extent(view.shape.map { Int($0) })
    let size = msg.storage_.size
    
    let storage = msg.storage_.data.withUnsafeBytes { (ptr:UnsafePointer<UInt8>) -> S in
        ptr.withMemoryRebound(to: S.ElementType.self, capacity: Int(size)) {
            let buffer = UnsafeBufferPointer(start: $0, count: Int(msg.storage_.size))
            return S(array: Array<S.ElementType>(buffer))
        }
    }
    
    return Tensor<S>(storage: storage, shape: shape, dimIndex: dimIndex, stride: stride)
}

public func fromProto<S:Storage>(msg:TensorDictionary) -> [String:Tensor<S>] {
    var items:[String:Tensor<S>] = [:]
    
    for (key, tensorMsg) in msg.items {
        items[key] = fromProto(msg: tensorMsg)
    }
    
    return items
}

public func serialize<S:Storage>(tensor:Tensor<S>) -> [UInt8] {
    assertionFailure()
    return []
}

public func serialize<E:NumericType>(tensor:Tensor<NativeStorage<E>>) -> [UInt8] {
    do {
        let msg = toProto(tensor: tensor)
        return try msg.serializeProtobufBytes()
    } catch {
        return []
    }
}

public func serialize<E:NumericType>(tensor:Tensor<NativeStorage<E>>) -> Data? {
    do {
        let msg = toProto(tensor: tensor)
        return try msg.serializeProtobuf()
    } catch {
        return nil
    }
}

public func serialize<E:NumericType>(tensors:[String:Tensor<NativeStorage<E>>]) -> [UInt8] {
    do {
        let msg = toProto(tensors: tensors)
        return try msg.serializeProtobufBytes()
    } catch {
        return []
    }
}

public func serialize<E:NumericType>(tensors:[String:Tensor<NativeStorage<E>>]) -> Data? {
    do {
        let msg = toProto(tensors: tensors)
        return try msg.serializeProtobuf()
    } catch {
        return nil
    }
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

public func deserialize<S:Storage>(data:Data) -> [String:Tensor<S>] {
    do {
        let msg = try TensorDictionary(protobuf: data)
        return fromProto(msg: msg)
    } catch {
        return [:]
    }
}

