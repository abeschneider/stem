Getting Started
===============
The main type in STEM is the `Tensor` class. It is a generic that is parameterized by a `Storage` protocol, which provides STEM the flexibility to run on accelerated platforms such as BLAS and GPUs. 

To create a new `Tensor`::

	let tensor = Tensor<NativeStorage<Double>>(shape: Extent(2, 5))

creates a `Tensor` that occupies 2 dimensions. The `Extent` struct is used whenever a spatial-dimension is required. `NativeStorage` is the default storage type that is unaccelerated, and `Double` specifies the element type of the `Storage` class. To use the object you can write::

	for i in 0..<tensor.shape[0] {
		for j in 0..<tensor.shape[1] {
			tensor[i, j] = i*j
		}
	}

Because most of the time you don't need to change the storage-type, you can simply the syntax by writing::

	typealias T = Tensor<NativeStorage<Double>>
	let tensor = T(shape: Extent(2, 5))

STEM has several other types that can be used for dispatch purposes are:

* `Matrix`
* `Vector`
* `RowVector`
* `ColumnVector`

These classes can help simplify code (e.g. if you are passed a `Vector` you know it occupies a single dimension). This can be especially helpful for dealing with accelerated code that may have separate functions for matrices and vectors. The classes also provide simplified syntax for construction::

	typealias Mat = Matrix<NativeStorage<Double>>
	typealias RowVec = RowVector<NativeStorage<Double>>

	let m = Mat([[1, 2, 3], [4, 5, 6]])
	let v = RowVec([1, 1])

STEM overloads mathematical operators for `Tensor`s::

	let result = m + v
	print(result)

will produce::

	[[2.000 3.000 4.000]
	 [5.000 6.000 7.000]]

Note, that underneath the operators are a collection of function. The above code is equivalent to::

	let result = Mat(shape: m.shape)
	add(left: m, right: v, result: result)
	print(result)
