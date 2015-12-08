# Swift Tensor Extensions for Machine-learning STEM

## Overview

STEM provides an extensible Tensor library with a primary focus for Swift. While there are many Tensor libraries out there (Numpy, TensorFlow, Torch, etc.), they do not support Swift.


### Why Swift?

There are several reasons Swift is a good candidate for writing Machine Learning algorithms:

1. It's compiled:

 a. None of the overhead normally associated with interpreted languages
 b. It's easy to call compiled libraries


2. It's simple to use, but has powerful syntax:

	a. Operator overloading
	b. Generics have much of the same power that templates have, but without having to resort to messy template hacking

3. It's strongly typed:

	a. Mistakes can be discovered quickly
	b. Can dispatch based on argument type

4. It supports good design

	a. Important for writing machine-learning algorithms for real-world problems

5. Playground provides a very nice method to document algorithms


## Design
The design of *STEM* is to provide a fast and flexible library. Rather than focus on a creating large framework with a million features, the goal is to provide the tools necessary to create those frameworks. Much of the code was inspired by Numpy, Theano, and Torch.

### Tensors
A `Tensor` in STEM is defined as a set of methods that operator on a `StorageView`, where the `StorageView` is defined as a window into a given `Storage`. The flexibility in STEM comes from the fact that `Storage` is a generic parameter to the `Tensor` class. This allows new types of `Storage` to be defined (e.g. BLAS, GPU, etc.) while still maintaining the same interface by allowing how the data is stored as well as providing a mechanism for dispatching to different methods based on each `Storage` type (e.g. `CBlasStorage` will cause CBlas methods to be called on the `Tensor` operations).

### Vectors and Matrices

STEM defines the following types:

* Vector
* RowVector
* ColumnVector
* Matrix

Each type is a subclass of `Tensor`, and provide:

1. Constraints on the `Tensor`s
2. Initializers that use those constraints to simplify construction
3. The correct methods will be called based on the type (e.g. vector*vector could mean dot product or outer product depending on whether they are `RowVectors` or `ColumnVectors`)
