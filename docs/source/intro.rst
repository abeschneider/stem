Introduction
============
The Swift Tensor Engine for Machine-Learning (STEM) is a Tensor library for Swift geared for Machine Learning. Similar to Numpy and Torch.Tensor, it is intended to provide a basis that other libraries can built on.

What are STEM's goals?
-----------------------
The main goal of STEM is to provide an easy to use Tensor library similar to Numpy or Torch to support
Machine Learning and scientific computing.

What aren't STEM's goals?
-------------------------
There are many uses for a Tensor library. STEM isn't trying to create an all encompassing framework.
Instead its goal is to provide a basis on which many other libraries can be written against.

Thus, STEM is not attempting to create a graphing library, Machine Learning library (though that is
being written in conjunction to STEM), or an automatic differentiation library. These

Why Swift?
----------
While Swift is still a new language, it has many qualities that make it great for Machine Learning:

* It's compiled:
	- None of the overhead normally associated with interpreted languages
	- It's easy to call compiled libraries
	- The LLVM allows it to also be interpreted

* It's simple to use, but has powerful syntax:
	- Operator overloading
	- Generics
	- While C++11 has made great strides to creating an easier to use language, it still has many aspects to it that make it difficult for a non-software engineer to write research code (e.g. template hacking)

* It's strongly typed:
	- Mistakes can be discovered quickly
	- Can dispatch based on argument type (Python functions can get messy because it cannot dispatch based on parameter type)

* It supports good design
	- Important for writing machine-learning algorithms for real-world problems
	- Better science through better code
	- Allows new ideas to be explored more easily

* Playground
	- Provides good method to document algorithms
	- Display input, equations, and output along with the code


Why not Numpy/Theano/Torch/TensorFlow?
--------------------------------------
One of the main problems with the current approaches is that they require developing across multiple languages. This imposes a cost both in terms of development time as well runtime. In order to keep code fast, Numpy and Torch have C-backends. Theano compiles the Python code into either C or GPU code. While it's theoretically possible to use TensorFlow entirely in C++, see previous section. The separation of fast code from the rest of the code base means there is an added penalty to developing new methods. Additionally, it usually means there is added cost in calling external code.

Swift on the other hand is a compile language that is easy to use. Because it is compiled, it can even call C code without the penalty usually associated with interpreted languages.
