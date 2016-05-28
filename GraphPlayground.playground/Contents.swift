////: Playground - noun: a place where people can play
//
import Cocoa
import stem

typealias D = NativeStorage<Double>


let input = Symbol<D>(Extent(3))
let target = Symbol<D>(Extent(5))
let linear = Linear<D>(inputSize: 3, outputSize: 5)
let sigmoid = Sigmoid<D>(size: 5)
let loss = L2Loss<D>(target: target)

let f = Sequence<D>(input, linear, sigmoid, loss)

var lst:[Op<D>] = []
for op in f.ops {
    print("op = \(op)")
    if let dop = op as? Differentiable {
        let grad = dop.gradient()
        lst.insert(grad as! Op<D>, atIndex: 0)
    }
}

print(lst)
let b = Sequence<D>(lst, key: "gradOutput")

f.apply()
b.apply()
