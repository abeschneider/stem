//: Playground - noun: a place where people can play

import Cocoa
import stem

var x = 0

func makeFunc() -> () -> Int {
    return { () -> Int in
        x += 1
        return x
    }
}

let f = makeFunc()
f()
f()
f()
