===============
Tensor Overview
===============

.. |STEM| replace:: **STEM**
.. |Tensor| replace:: ``Tensor``
.. |Vector| replace:: ``Vector``
.. |Matrix| replace:: ``Matrix``
.. |Number| replace:: ``NumericType``

.. _Synopsis:

Synopsis
--------

First, we will define a few convenience aliases:

.. code:: swift

  typealias RowVector<NativeStorage<Double>> RowVec
  typealias ColumnVector<NativeStorage<Double>> ColVec
  typealias Matrix<NativeStorage<Double>> Mat
  typealias Tensor<NativeStorage<Double>> T

Next, we can create two vectors:

.. code:: swift

  // construct vector with values
  let v1 = RowVec([1, 2, 3])

  // construct vector by size
  let v2 = ColVec(rows: 5)

  // construct vector using another vector
  let v3 = RowVec(v1)

  // set v2's values manually
  for i in 0..<5 {
    v2[i] = 2*i
  }


|STEM| supports standard linear algebra operators:

.. code:: swift

  // take the outer product (results in a matrix)
  let m1 = v1⨯v2

  // add two vectors together
  let v4 = v1+v3

  // multiply by a scalar
  let v5 = 0.5*v1

|STEM| also supports advanced indexing (similar to Numpy and Matlab):

.. code:: swift

  let v6 = v2[1..<4]
  let m2 = m1[1..<4, 0..<2]

As |STEM|'s name implies N-dimensional Tensors are supported. Both the |Vector|
and |Matrix| classes are specializations of the |Tensor| class. These
specializations allow for simpler construction methods as well as the
use of accelerated libraries such as **CBLAS** and **CUDA** or **OpenCL**
through function overloading.

Function overloading also allows |STEM| to support broadcasting:

.. code:: swift

  let m3 = Mat([[1, 2, 3], [4, 5, 6]])
  let v7 = RowVec([1, 1, 1])
  let v8 = ColVec([2, 2])

  // applies v7 to each row of m3
  let m4 = m3+v7

  // applies v8 to each column of m3
  let m5 = m3+v8

.. _Storage:

Storage
-------
All |Tensor| s have an associated ``Storage`` class that is responsible for
the allocated memory. The two built-in ``Storage`` types are: ``NativeStorage``
and ``CBlasStorage``. Other storage types (e.g. **CUDA** or **OpenCL**) can
be added without requiring any rewrite of the main library. Because the ``Storage``
type determines which functions get called. If no methods have been specified
for the ``Storage`` class, ``NativeStorage`` will be called by default.

The ``Storage`` protocol is defined as:

.. code:: swift

  public protocol Storage {
    associatedtype ElementType:NumericType

    var size:Int { get }
    var order:DimensionOrder { get }

    init(size:Int)
    init(array:[ElementType])
    init(storage:Self)
    init(storage:Self, copy:Bool)

    subscript(index:Int) -> ElementType {get set}

    // returns the order of dimensions to traverse
    func calculateOrder(dims:Int) -> [Int]

    // re-order list in order of dimensions to traverse
    func calculateOrder(values:[Int]) -> [Int]
  }

An implementation of ``Storage`` determines the allocation through the ``init``
methods, ``subscript`` determines how the storage gets indexed, and ``calculateStride``
allows the ``Storage`` to be iterated through in a sequential fashion.

The |Tensor| class frequently makes use of the generator ``IndexGenerator`` to iterate
through the ``Storage`` class. This provides a convenient way to access all the
elements without knowing the underyling memory allocation.

To do so, the |Tensor| class defined the method:

.. code:: swift

  public func indices(order:DimensionOrder?=nil) -> GeneratorSequence<IndexGenerator> {
    if let o = order {
        return GeneratorSequence<IndexGenerator>(IndexGenerator(shape, order: o))
    } else {
        return GeneratorSequence<IndexGenerator>(IndexGenerator(shape, order: storage.order))
    }
  }

which can be used like:

.. code:: swift

  func fill<StorageType:Storage>(tensor:Tensor<StorageType>, value:StorageType.ElementType) {
      for i in tensor.indices() {
          tensor.storage[i] = value
      }
  }

However, as mentioned previously, if an optimized version for a particular |Tensor|
operation exists, you can write:

.. code:: swift

  // This will be used if the Tensor's storage type is CBlasStorage for doubles,
  // an alternative can be specified for Floats separately.
  func fill(tensor:Tensor<CBlasStorage<Double>>, value:StorageType.ElementType) {
    // call custom library
  }


.. csv-table:: Storage Types
  :header: "Type", "Description"
  :widths: 20, 20

  "NativeStorage", "Unaccelerated using row-major memory storage"
  "CBlasStorage", "CBLAS acceleration using column-major storage"
  "GPUStorage", "(Not Implemented) GPU acceleration using row-memory storage"

.. _Tensor_Class:

Tensor Class
------------
The ``Tensor`` class is parameterized by the ``Storage`` type, allowing instances
of the class to maintain a pointer to the underlying memory. The ``Tensor`` class
also has an instance of ``ViewType``, which allows different views of the same
memory to be constructed, and the array ``dimIndex``, which determines the order
that the dimensions in the ``Tensor`` are traversed. These features allow for
multiple ``Tensor`` s to provide a different view to the same memory (e.g. a slice
of a ``Tensor`` can be created by changing the ``ViewType`` instance, or a
``Tensor`` can be transposed by shuffling ``dimIndex``).

.. note::

  Throughout the documentation ``Tensor<S>`` indicates the parameterization of
  the ``Tensor`` class by ``Storage`` type ``S``, and ``NumericType`` refers to
  ``S.NumericType`` (see section on ``Storage`` for details).

.. _Tensor_Construction:

Tensor Construction
-------------------
.. function:: Tensor<S>(_ shape:Extent)

  Constructs a tensor with the given shape.

.. function:: Tensor<S>([NumericType], axis:Int)

  Constructs a vector along the given axis

.. function:: Tensor<S>(colvector:[NumericType])

  Constructs a row vector (equivalent to ``Tensor<S>([NumericType], axis:0)``)

.. function:: Tensor<S>([[NumericType]])

  Constructs a matrix

.. _Indexing:

Indexing
--------
|STEM| supports single indexing as well as slice indexing. Given a |Tensor| T:

To index element (i, j, k):

.. code:: swift

  let value = T[i, j, k]
  T[i, j, k] = value

To index the slices (if:il, jf:jl, kf:kl):

.. code:: swift

  let T2 = T[if...il, jf...jl, kf...kl]
  T[if...il, jf...jl, kf...kl] = T2

Views
------
Views in |STEM| are instances of |Tensor| that point to the same ``Storage``
as another |Tensor| but with different bounds and/or ordering of dimensions. Views
are most commonly created whenever a slice indexing is used.

A copy of a view can be made by using the ``copy`` function.
