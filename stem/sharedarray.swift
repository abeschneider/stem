//
//  sharedarray.swift
//  stem
//
//  Created by Abe Schneider on 11/10/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation

public class SharedArray<T where T:Equatable>: CollectionType {
    public var memory:[T]
    public var offset:Int
    
    public var startIndex:Int { return offset }
    public var endIndex:Int { return memory.count + offset }
    
    init(_ values:T...) {
        memory = values
        offset = 0
    }
    
    init(_ values:[T], offset:Int=0) {
        memory = values
        self.offset = offset
    }
    
    init(count:Int, repeatedValue:T) {
        memory = [T](count:count, repeatedValue:repeatedValue)
        offset = 0
    }
    
    public subscript(index:Int) -> T {
        get { return memory[index+offset] }
        set { memory[index+offset] = newValue }
    }
    
    func copy(array:SharedArray<T>) {
        memory = array.memory
        offset = array.offset
    }
}

extension SharedArray: CustomStringConvertible {
    public var description:String {
        let values = memory.map { String($0) }.joinWithSeparator(", ")
        return "[\(values)]"
    }
}
