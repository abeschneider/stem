# Swift Tensor Extensions for Machine-learning STEM

## Overview

STEM provides an extensible Tensor library with a primary focus for Swift. While there are many Tensor libraries out there (Numpy, TensorFlow, Torch, etc.), there currently do not directly support Swift. Direct support of Swift is important for speed and to fully take advantage of the language.

### Why Swift?
There are a properties of Swift that make it a good candidate for developing machine learning algorithms:

1. Runs on the LLVM
2. Allows for advanced operator overloading
3. Has support of a large company

#### LLVM support
Julia was on the first language to demonstrate the advantages of running on the LLVM. The language can be treated as both interpreted and compiled. This means that code can be written and tested easily while the end result can be optimized and fast.

While languages such as Python have had a lot of work to make them faster (Numba, Psyco, Theano, etc.), the language is still ultimately interpreted. Optimizations can also be added by writing a library in C/C++ (e.g. TensorFlow). However, this requires a bridge between the compiled library and Python to be maintained. While the work can be minimized by either using tools like Swig or Boost::Python, debugging can still be a pain because it can be difficult to debug both Python and C/C++ at the same time.

By writing for a language that runs on the LLVM means that development can be significantly increased through faster feedback and debugging tools and the resulting code can be optimized.

Finally, by not having to deal with language bridges means that there are no speed penalties from making calls between languages. These penalties exist for libraries such as Numpy, Torch, and TensorFlow (if used in Python).

#### Why not Julia?
Julia is a great language. However, to date it currently does not have the same support for developing large applications. Julia's type system is still evolving to allow for features that Swift already supports and the tools for developing in Julia are still works in progress. Swift, on the the other hand, has XCode, which can provide advanced debugging as well as the ability to compile as you type. Both of these features makes finding and fixing bug take much less time.

### Operator overloading
Swift allows custom operators to be defined. Anyone who has used or develop a math library will recognize the ability to define operators can make writing equations much easier. It can be argued that Python's ability to overload operators is one of the important features that made it popular for machine learning.

### Support

In it's current state, Swift is not perfect. However, Apple has a vested interest in fixing Swift bugs and making it fast enough to run on embedded systems.

## Design
The design of *STEM* came from looking at how many different libraries work and going through many different iterations. The focus on the design was on both flexibility and speed over features. The plan is to keep STEM simple and allow other libraries layered on top to provide more complex functionality.

### Tensors
`Tensor`s in STEM is defined as a set of method that operator on a `StorageView`, where the `StorageView` maintains a window into a given `Storage`. The flexibility in STEM comes from the fact that `Storage` is a generic parameter to the `Tensor` class. This allows new types of `Storage` to be defined (e.g. BLAS, GPU, etc.) while still maintaining the same interface.

### Vectors and Matrices

Both the `Vector` and `Matrix` class are subclasses of `Tensor`. They provide two things:

1. Constraints on the `Tensor`s
2. Initializers that use those constraints to simplify construction
