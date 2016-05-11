////: Playground - noun: a place where people can play
//
import Cocoa
import stem


typealias D = NativeStorage<Double>

//let m = Matrix<D>([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
//let v = Matrix<D>([[1, 2, 3]])
////let v = Matrix<D>([[1], [2], [3]])
//let z = RowVector<D>([1, 2 ,3])
//let u = Matrix<D>([[1, 2], [2, 3], [3, 4]])
//
//
//let s = NativeStorage<Double>(size: 9)
//s.calculateStride(Extent(0, 3))
//s.calculateStride(Extent(3, 0))
//s.calculateStride(Extent(3, 3, 0))
//
//v.shape.dims
//v.stride
//let v2 = Tensor(tensor: v, shape: Extent(3, 1), stride: [1, 1])
//String(v2)
//
//z.stride
//z.shape.dims
//let z2 = Tensor(tensor: z, shape: Extent(3, 3), stride: [0, 1])
//String(z2)
//
//let z3 = Tensor(tensor: z, shape: Extent(3, 3, 3), stride: [1, 0, 0])
//String(z3)
////z.shape.dims
////z.stride
////let z2_stride = z.storage.calculateStride(Extent(0, 3))
//
//u.shape.dims
//u.stride
//let u2 = Tensor(tensor: u, shape:Extent(3, 2), stride: [2, 1])
//String(u2)
//
//let t = RowVector<D>([1, 2, 3, 4])
//t.shape.dims
//let t2 = Tensor(tensor: t, shape: Extent(3, 4), stride: [0, 1])
//String(t2)
//
//let r = RowVector<D>([5, 6, 7])
//r.shape.dims
//let r2 = Tensor<D>(tensor: r, shape: Extent(3, 4), stride: [1, 0])
//String(r2)
//
//String(t2*r2)


/*
 t:[1, 3] -> t[3, 3]
 */
//func broadcast<S>(tensor:Tensor<S>, shape:Extent) -> [Int] {
//    let diff = shape.count - tensor.shape.count
//    
//    var strides = [Int](count: shape.count, repeatedValue: 0)
//    for i in 0..<tensor.shape.count {
//        strides[i+diff] = tensor.shape[i]
//    }
//
//    for i in 0..<tensor.shape.count {
//        let newShape = shape[i+diff]
//        let oldShape = tensor.shape[i]
//        if newShape != oldShape {
//            // can only expand dimensions of 1
//            precondition(oldShape == 1)
//            
////            let axis = tensor.shape.count + i
//            strides[i] = 0
//        }
//    }
//    
//    return strides
////    return Tensor<S>(tensor: t, shape: shape, stride: stride)
//}

let ns = NativeStorage<Double>(array: [3, 2])
let cs = CBlasStorage<Double>(array: [3, 3])
let stride1 = ns.calculateOrder(calculateStride(Extent(3, 2)))
let stride2 = cs.calculateOrder(calculateStride(Extent(3, 3)))


let t1 = Tensor<D>(shape: Extent(1, 3))
let t2 = Tensor<D>(shape: Extent(3, 1))

let s1 = broadcast(t1, shape: Extent(3, 3))
let s2 = broadcast(t2, shape: Extent(3, 3))

let b1 = Tensor<D>(tensor: t1, shape: Extent(3, 3), stride: s1)
//String(t2)

//v.shape = Extent(3, 3)
//v.dimIndex = [1, 0, 1]
//let v3 = Tensor(tensor: v, shape: Extent(3, 3), stride: <#T##[Int]#>)
//String(v)

//let z2 = Tensor(tensor: z, shape: Extent(3, 3), stride: z2_stride)
//String(z2)

//let W = Symbol<D>(value: Tensor<D>(shape: Extent(3, 3)))
//let bias = Symbol<D>(value: ColumnVector<D>(rows: 3))
//let input = Symbol<D>(value: RowVector<D>(cols: 3))
//
//// TODO: Provide method to construct that doesn't require input. This
//// will allow convenience methods of construction like:
////  sequence(Linear(W, b), Sigmoid()),
//// which would create the topology along with the traversal.
//
//let linear = Linear<D>(input: input, weight: W, bias: bias)
//let sigmoid = Sigmoid<D>(input: linear.outputs[0])
//
//let seq = SequentialTraversal<D>()
//seq.add(linear)
//seq.add(sigmoid)
//seq.update()


// graph is defined by:
// topology + traversal
