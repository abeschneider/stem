////: Playground - noun: a place where people can play
//
import Cocoa
import stem

typealias D = NativeStorage<Double>

let size = 10
// [(input (+) out) -> T -> tanh, (input (+) out) -> T -> sigmoid] -> mul
// [mul, (input (+) out)] -> T -> sigmoid] -> sum
// sum -> state -> out

let state = Constant<D>(zeros(Extent(size)))
let input = Constant<D>(zeros(Extent(size)))
let output = Constant<D>(zeros(Extent(size)))

//let inputTransform = Sequence<D>(
//    ConcatOp<D>(input, output),
//    LinearOp<D>(outputSize: size),
//    SigmoidOp<D>()
//)
//
//let inputGate = Sequence<D>(
//    ConcatOp<D>(input, output),
//    LinearOp<D>(outputSize: size),
//    TanhOp<D>()
//)
//
//let forgetGate = Sequence<D>(
//    ConcatOp<D>(input, output),
//    LinearOp<D>(outputSize: size),
//    SigmoidOp<D>()
//)
//
//let inputValue = MulOp(inputTransform, inputGate)
//let forgetValue = MulOp(state, forgetGate)
