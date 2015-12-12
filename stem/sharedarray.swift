//
//  sharedarray.swift
//  stem
//
//  Created by Abe Schneider on 11/10/15.
//  Copyright © 2015 Abe Schneider. All rights reserved.
//

import Foundation

class SharedArray<T where T:Equatable>: CollectionType {
    var memory:[T]
    
    var startIndex:Int { return 0 }
    var endIndex:Int { return memory.count }
    
    init(_ values:T...) {
        memory = values
    }
    
    init(_ values:[T]) {
        memory = values
    }
    
    init(count:Int, repeatedValue:T) {
        memory = [T](count:count, repeatedValue:repeatedValue)
    }
    
    subscript(index:Int) -> T {
        get { return memory[index] }
        set { memory[index] = newValue }
    }
    
    func copy(array:SharedArray<T>) {
        self.memory = array.memory
    }
}

extension SharedArray: CustomStringConvertible {
    var description:String {
        let values = memory.map { String($0) }.joinWithSeparator(", ")
        return "[\(values)]"
    }
}
