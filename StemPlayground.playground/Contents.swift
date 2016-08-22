//: STEM - Swift Tensor Extension for Machine-learning
//: ==================================================

// standard imports
import Cocoa
import stem
import XCPlayground


//: Convenience aliases
//: -------------------
typealias F = NativeStorage<Float>
typealias I = NativeStorage<Int>

var d = OrderedDictionary<F>()
d["l1"] = LinearOp<F>(inputSize: 5, outputSize: 10)
d["l2"] = LinearOp<F>(inputSize: 5, outputSize: 5)
d["l1"] = LinearOp<F>(inputSize: 5, outputSize: 20)

print(d.keys)
print(d.orderedValues)
print(d.values)


//: Creating vectors
//: ----------------
let v1 = Tensor<F>([1, 2, 3, 4])
String(v1)

let v2 = Tensor<F>(rowvector: [1, 2, 3, 4])
String(v2)

let v3 = Tensor<F>([1, 2, 3, 4], axis: 0)
String(v3)

let v4 = Tensor<F>([1, 2, 3, 4], axis: 1)
String(v4)

let v5 = Tensor<F>([1, 2, 3, 4], axis: 2)
String(v5)


// integer vectors
let iv1 = Tensor<I>(rowvector: [1, 2, 3, 4, 5])
String(iv1)

//: Vector indexing
//: ---------------

// single element (results in a scalar)
let d1:Float = v1[0]
let d2:Float = v1[1]

v1[0] = 10
v1[1] = 20

// ranges
let vr1 = v1[0...2]
String(vr1)

String(v1)


vr1[0...2] = Tensor<F>([1, 2, 3])
String(vr1)

String(v1)

//: Vector operations
//: -----------------

let v7 = v1+v1
let v8 = 0.5*v1
let v9 = v1**2

let v10 = Tensor<F>(rowvector: [2, 2, 2, 2])
String(v10)


// Hadamard product
let v11 = v10*v2
String(v11)

// dot product
let v12:Float = v10⊙(v2+v2)
String(v12)

// outer product
let v13 = v10⊗(v2+v2)
String(v13)


//: Matrix creation
//: ---------------
let m1 = Tensor<F>([[1, 2, 3], [4, 5, 6]])
String(m1)

let m2 = Tensor<F>(Extent(3, 3))
String(m2)

//: Matrix indexing
//: ---------------
let d3:Float = m1[0, 0]
let d4:Float = m1[1, 2]
let vr2 = m1[0..<2, 1..<3]
String(vr2)

m2[0..<2, 1..<3] = Tensor<F>([[1, 1], [1, 1]])
String((m2))

String(m2[all, 0])

String(m2[all, 1])

String(m2[all, 2])

//: Matrix operations
//: -----------------
String(m1)

let v14 = Tensor<F>(colvector: [1, 1, 1])
let v15:Tensor<F> = m1⊙v14
String(v15)

let m3:Tensor<F> = Tensor<F>([[1, 2, 3], [1, 2, 3]])
let m4:Tensor<F> = m1⊙m3.transpose()
String(m4)


let s = Extent(3)
let s2 = Extent(s, over: 5)
s2