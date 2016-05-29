////: Playground - noun: a place where people can play
//
import Cocoa
import stem

typealias D = NativeStorage<Double>


let input = Symbol<D>(Extent(3))
let linear = Linear<D>(inputSize: 3, outputSize: 5)
let sigmoid = Sigmoid<D>(size: 5)

let target = Symbol<D>(Extent(5))
let loss = L2Loss<D>(target: target)

let f = Sequence<D>(input, linear, sigmoid, loss)

let optimizer = GradientDescentOptimizer<D>(f, alpha: Symbol<D>(0.1))

for i in 0..<100 {
    optimizer.apply()
    print(loss.value)
}
