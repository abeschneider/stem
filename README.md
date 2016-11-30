# Swift Tensor Engine for Machine-learning (STEM)

## Overview

STEM provides an extensible Tensor library for Swift with a primary focus on the development of Machine Learning algorithms. The focus of STEM is: (1) allowing research on new topologies (especially recurrent ones) easy; (2) making it simple to define new type of operations; (3) providing a mechanism for multiple types of optimization (e.g. gradient descent, message passing, and genetic algorithms); and (4) being flexible! The use of the Swift language helps support these goals along with having multiple other benefits:

1. By using a language based on the LLVM means that Swift can both be interpreted as well as compiled. Thus, unlike Torch and Theano, only a single language is required. Bridging code between two language both complicates debugging code as well as adds overhead to the runtime of the code. 
2. An additional benefit of using a single language is that defining new operations is much faster to develop.
3. Construction of network is done be explicitely connecting different modules together (via `connect`). Most other libraries simplify network construction by either constructing networks based on translating mathematical expressions or allowing operations to be sequentially laid out. This complicates the process of making networks with non-sequential topologies (e.g. recurrent or skip networks).
4. Allows for good design. One example is that Swift `Extension`s allow an operation to be defined independent of its optimization method. For example, `LinearOp` does not have a concept of how it should be optimized. The extension `LinearGrad`, which adheres to the `Differentiable` protocol allows the operation to be used in gradient descent. Another extension can be proposed to allow for other optimization to be used.

STEM is split into multiple sections (similar to Torch). Currently it consists of:

1. A Tensor library allowing for multiple storage backings. Swift's ability to define new operators makes writing out equations simple and intuitive.
2. A random library for generating random Tensors
3. A computational graph library for composing operations
4. Optimization using gradient descent

For more information, please visit the documentation: 

http://stem.readthedocs.org/en/latest/


*Please note, this project is still very new, and is still very much in the proof-of-concept phase. There are still many functions waiting to be written. While I believe the majority of the core design is in place, more work is required. Thus, use at your own risk, and contributors welcomed!*
