//: STEM - Swift Tensor Extension for Machine-Learning
//: ==================================================

import Cocoa
import stem
import XCPlayground

//: Convenience aliases
//: -------------------
typealias Vec = Vector<NativeStorage<Double>>
typealias RowVec = RowVector<NativeStorage<Double>>
typealias ColVec = ColumnVector<NativeStorage<Double>>
typealias Mat = Matrix<NativeStorage<Double>>

//: Creating vectors
//: ----------------
let v1 = Vec([1, 2, 3, 4])
print("\(v1)")

let v2 = ColVec([1, 2, 3, 4])
print("\(v2)")

let v3 = RowVec([1, 2, 3, 4])
print("\(v3)")

let v4 = Vec([1, 2, 3, 4], axis: 0)
print("\(v4)")

let v5 = Vec([1, 2, 3, 4], axis: 1)
print("\(v5)")

let v6 = Vec([1, 2, 3, 4], axis: 2)
print("\(v6)")

//: Vector indexing
//: ---------------

// single element (results in a scalar)
let d1:Double = v1[0]
let d2:Double = v1[1]
// NB: v1[0] alone results in ambigous error, look into (return value disambiguiates)

let r = 0..<0
r.count



//: Vector operations
//: -----------------

let v7 = try v1+v1
let v8 = 0.5*v1
let v9 = v1**2

let v10 = RowVec([2, 2, 2, 2])


try v2+v2
let v11 = try v10*v2
let test = try v2+v2
let v12 = try dot(left: v10, right: test)
//let v12 = try v2*v10
//print("\(v11)")
//print("\(v10[1])")


//: Creating matrices

//let v2 = 0.5*v
//
//let v3 = try v ⊕ v2
//
//let v4 = v**2
//
////: outer product
//
////v⊗v2
//let v5 = RowVec(v)
//let v6 = ColVec(v2)
//
//let v7 = v5⊗v6
//print("\(v7)")
