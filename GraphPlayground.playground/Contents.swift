////: Playground - noun: a place where people can play
//
import Cocoa
import stem

typealias D = NativeStorage<Double>

//let input = Symbol<D>(uniform(Extent(5)))
//let target = Symbol<D>(uniform(Extent(3)))
//let loss = L2Loss<D>(target: target)
//
//let net = Sequence<D>(
//    input,
//    Linear<D>(inputSize: 5, outputSize: 3),
//    Sigmoid<D>(size: 3),
//    loss
//)
//
//let l = Linear<D>(inputSize: 5, outputSize: 3)
//let l2 = copy(l, shared: true)
//let l3 = copy(l, shared: false)
//
//l.weight[0, 0] = 1
//
//let s:D.ElementType = l.weight[0, 0]
//let s2:D.ElementType = l2.weight[0, 0]
//let s3:D.ElementType = l3.weight[0, 0]



//let cnet = net.clone(true)


//let optimizer = GradientDescentOptimizer<D>(net, alpha: Symbol<D>(0.1))

//for i in 0..<100 {
//    optimizer.apply()
//    print(loss.value)
//}


func unroll(op:Op<D>, count:Int) -> Sequence<D> {
    var lst:[Op<D>] = []
    
    for _ in 0..<count {
        lst.append(copy(op, shared: true))
    }
    
    return Sequence<D>(lst)
}

let layer = Sequence<D>(Linear<D>(inputSize: 5, outputSize: 5),
                        Sigmoid<D>(size: 5))

let target = Symbol<D>(uniform(Extent(5)))
let net = Sequence<D>(unroll(layer, count: 3), L2Loss<D>(target: target))


//let v1:Tensor<D> = uniform(Extent(3))
//let v2:Tensor<D> = uniform(Extent(3))
//let v3:Tensor<D> = uniform(Extent(3))
//let v4:Tensor<D> = uniform(Extent(3))
//let v5:Tensor<D> = uniform(Extent(3))

//class Tree<Element: Comparable> {
//    let value: Element
//    // entries < value go on the left
//    let left: Tree<Element>?
//    // entries > value go on the right
//    let right: Tree<Element>?
//    
//    init(value:Element, left:Tree<Element>?, right:Tree<Element>?) {
//        self.value = value
//        self.left = left
//        self.right = right
//    }
//}

//indirect enum Tree<S:Storage> {
//    case Empty
//    case Node(Tensor<S>, Tree<S>?, Tree<S>?)
//    
//    init() {
//        self = .Empty
//    }
//    
//    init(_ value:Tensor<S>, left:Tree<S>?, right:Tree<S>?) {
//        self = .Node(value, left, right)
//    }
//    
//    init(_ value:Tensor<S>) {
//        self = .Node(value, nil, nil)
//    }
//}
//
//let tree:Tree<D> = Tree(v1,
//                        left: Tree(v2),
//                        right: Tree(v3, left: Tree(v4), right: Tree(v5)))
//
//func traverse(tree:Tree<D>) {
//    switch tree {
//    case let .Node(value, left, right):
//        print("value: \(value)")
//        if let l = left {
//            traverse(l)
//        }
//        if let r = right {
//            traverse(r)
//        }
//        break
//    case .Empty:
//        break
//    }
//}
//
//traverse(tree)
//
//
