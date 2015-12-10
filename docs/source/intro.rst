Introduction
============
The Swift Tensor Engine for Machine-Learning (STEM) is a Tensor library for Swift geared for Machine Learning. Similar to Numpy and Torch.Tensor, it is intended to provide a basis that other libraries can built on.

Why Swift?
----------
While Swift is still a new language, it has many qualities that make it great for Machine Learning:

1. It's compiled:

	a. None of the overhead normally associated with interpreted languages b. It's easy to call compiled libraries

2. It's simple to use, but has powerful syntax:

	a. Operator overloading 
	b. Generics 
	c. While C++11 has made great strides to creating an easier to use language, it still has many aspects to it that make it difficult for a non-software engineer to write research code (e.g. template hacking)

3. It's strongly typed:

	a. Mistakes can be discovered quickly 
	b. Can dispatch based on argument type (Python functions can get messy because it cannot dispatch based on parameter type)

4. It supports good design

	a. Important for writing machine-learning algorithms for real-world problems
	b. Better science through better code
	c. Allows new ideas to be explored more easily

5. Playground provides a very nice method to document algorithms

	a. Display input, equations, and output along with the code


Why not Numpy/Theano/Torch/TensorFlow?
--------------------------------------
One of the main problems with the current approaches is that they require developing across multiple languages. This imposes a cost both in terms of development time as well runtime. In order to keep code fast, Numpy and Torch have C-backends. Theano compiles the Python code into either C or GPU code. While it's theoretically possible to use TensorFlow entirely in C++, see previous section. The separation of fast code from the rest of the code base means there is an added penalty to developing new methods. Additionally, it usually means there is added cost in calling external code.

Swift on the other hand is a compile language that is easy to use. Because it is compiled, it can even call C code without the penalty usually associated with interpreted languages.
