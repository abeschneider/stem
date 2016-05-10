==============
Tensor library
==============

.. |STEM| replace:: **STEM**
.. |Tensor| replace:: ``Tensor``
.. |Vector| replace:: ``Vector``
.. |Matrix| replace:: ``Matrix``
.. |Number| replace:: ``NumericType``

Overview
--------
The documentation is split into several sections:

* :ref:`Synopsis`: Quick overview of library
* :ref:`Storage`: Discussion of how Tensors are storage and how this affects acceleration
* :ref:`Tensor_Class`: Discussion of the Tensor class
* :ref:`Tensor_Construction`: Discussion of how to construct new instances of Tensors
* :ref:`Functions`: Discussion of operator overloading in |STEM|, the standard set of operators, and how to define new operators
* :ref:`Indexing`: Discussion of how to index into Tensors

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

To do so, the |Tensor| class defined the methid:

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
The |Tensor| class hold an instance of ``Storage`` along with a view into
the storage. Multiple instances of |Tensor| may point to the same ``Storage``
providing different views of the same data. This allows operations such as indexing
to operate in an efficient manner without requiring copies of the memory to be made.

.. _Tensor_Construction:

Tensor Construction
-------------------
The |Tensor| class comes with three constructors. To construct a |Tensor| with a given shape:

.. code:: swift

  init(shape:Extent)


To create a view of a |Tensor|, where ``window`` is an array of ``Range`` with
each element representing a single dimension:

.. code:: swift

  init(_ tensor:Tensor, window:[Range<Int>])


To create a view of a |Tensor| with the ability to shuffle the dimensions, where
``dimIndex`` is the order of the dimensions and ``view`` is the view used:

.. code:: swift

  init(_ tensor:Tensor, dimIndex:[Int]?=nil, view:StorageView<StorageType>?=nil, copy:Bool=false)


Functions
---------

.. function:: add(lhs:Tensor<S>, rhs:Tensor<S>, result:Tensor<S>)
              add(lhs:Tensor<S>, rhs:NumericType, result:Tensor<S>)
              add(lhs:NumericType, rhs:Tensor<S>, result:Tensor<S>)
              +(lhs:Tensor<S>, rhs:Tensor<S>) -> Tensor<S>
              +(lhs:Tensor<S>, rhs:NumericType) -> Tensor<S>
              +(lhs:NumericType, rhs:Tensor<S>) -> Tensor<S>

   Adds ``lhs`` and ``rhs`` together, either place results in ``result``
   or returning its value. If the shape of ``lhs`` and ``rhs`` do not
   match, will either use ``broadcast`` to match their size, or use
   optimized methods to accomplish the same purpose.

.. function:: sub(lhs:Tensor<S>, rhs:Tensor<S>, result:Tensor<S>)
              sub(lhs:Tensor<S>, rhs:NumericType, result:Tensor<S>)
              sub(lhs:NumericType, rhs:Tensor<S>, result:Tensor<S>)
              -(lhs:Tensor<S>, rhs:Tensor<S>) -> Tensor<S>
              -(lhs:Tensor<S>, rhs:NumericType) -> Tensor<S>
              -(lhs:NumericType, rhs:Tensor<S>) -> Tensor<S>

  Subtracts ``lhs`` from ``rhs``, either place results in ``result``
  or returning its value. If the shape of ``lhs`` and ``rhs`` do not
  match, will either use ``broadcast`` to match their size, or use
  optimized methods to accomplish the same purpose.

.. function:: mul(lhs:Tensor<S>, rhs:Tensor<S>, result:Tensor<S>)
              mul(lhs:Tensor<S>, rhs:NumericType, result:Tensor<S>)
              mul(lhs:NumericType, rhs:Tensor<S>, result:Tensor<S>)
              *(lhs:Tensor<S>, rhs:Tensor<S>) -> Tensor<S>
              *(lhs:Tensor<S>, rhs:NumericType) -> Tensor<S>
              *(lhs:NumericType, rhs:Tensor<S>) -> Tensor<S>

   Performs element-wise multiplication between ``lhs`` and ``rhs``,
   either place results in ``result`` or returning its value. If the
   shape of ``lhs`` and ``rhs`` do not match, will either use ``broadcast``
   to match their size, or use optimized methods to accomplish the same purpose.

.. function:: div(lhs:Tensor<S>, rhs:Tensor<S>, result:Tensor<S>)
              div(lhs:Tensor<S>, rhs:NumericType, result:Tensor<S>)
              div(lhs:NumericType, rhs:Tensor<S>, result:Tensor<S>)
              /(lhs:Tensor<S>, rhs:Tensor<S>) -> Tensor<S>
              /(lhs:Tensor<S>, rhs:NumericType) -> Tensor<S>
              /(lhs:NumericType, rhs:Tensor<S>) -> Tensor<S>

  Performs element-wise division between ``lhs`` and ``rhs``,
  either place results in ``result`` or returning its value. If the
  shape of ``lhs`` and ``rhs`` do not match, will either use ``broadcast``
  to match their size, or use optimized methods to accomplish the same purpose.

.. function:: iadd(lhs:Tensor<S>, rhs:Tensor<S>)
              iadd(lhs:Tensor<S>, rhs:NumericType)
              iadd(lhs:NumericType, rhs:Tensor<S>)
              +=(lhs:Tensor<S>, rhs:Tensor<S>) -> Tensor<S>
              +=(lhs:Tensor<S>, rhs:NumericType) -> Tensor<S>
              +=(lhs:NumericType, rhs:Tensor<S>) -> Tensor<S>

   Adds ``lhs`` to ``rhs`` and stores result in ``lhs``,
   either place results in ``result`` or returning its value. If the
   shape of ``lhs`` and ``rhs`` do not match, will either use ``broadcast``
   to match their size, or use optimized methods to accomplish the same purpose.

.. function:: isub(lhs:Tensor<S>, rhs:Tensor<S>)
              isub(lhs:Tensor<S>, rhs:NumericType)
              isub(lhs:NumericType, rhs:Tensor<S>)
              -=(lhs:Tensor<S>, rhs:Tensor<S>) -> Tensor<S>
              -=(lhs:Tensor<S>, rhs:NumericType) -> Tensor<S>
              -=(lhs:NumericType, rhs:Tensor<S>) -> Tensor<S>

  Subtracts ``rhs`` from ``lhs`` and stores result in ``lhs``,
  either place results in ``result`` or returning its value. If the
  shape of ``lhs`` and ``rhs`` do not match, will either use ``broadcast``
  to match their size, or use optimized methods to accomplish the same purpose.

.. function:: imul(lhs:Tensor<S>, rhs:Tensor<S>)
              imul(lhs:Tensor<S>, rhs:NumericType)
              imul(lhs:NumericType, rhs:Tensor<S>)
              *=(lhs:Tensor<S>, rhs:Tensor<S>) -> Tensor<S>
              *=(lhs:Tensor<S>, rhs:NumericType) -> Tensor<S>
              *=(lhs:NumericType, rhs:Tensor<S>) -> Tensor<S>

  Multiplies ``lhs`` by ``rhs`` and stores result in ``lhs``,
  either place results in ``result`` or returning its value. If the
  shape of ``lhs`` and ``rhs`` do not match, will either use ``broadcast``
  to match their size, or use optimized methods to accomplish the same purpose.

.. function:: idiv(lhs:Tensor<S>, rhs:Tensor<S>)
              idiv(lhs:Tensor<S>, rhs:NumericType)
              idiv(lhs:NumericType, rhs:Tensor<S>)
              /=(lhs:Tensor<S>, rhs:Tensor<S>) -> Tensor<S>
              /=(lhs:Tensor<S>, rhs:NumericType) -> Tensor<S>
              /=(lhs:NumericType, rhs:Tensor<S>) -> Tensor<S>

  Divides ``lhs`` by ``rhs`` and stores results in ``lhs``,
  either place results in ``result`` or returning its value. If the
  shape of ``lhs`` and ``rhs`` do not match, will either use ``broadcast``
  to match their size, or use optimized methods to accomplish the same purpose.

.. function:: pow(lhs:Tensor<S>, exp:NumericType) -> Tensor<S>
              **(lhs:Tensor<S>, exp:NumericType) -> Tensor<S>

  Raises every element of ``lhs`` by ``exp``.

.. function:: exp(op::Tensor<S>) -> Tensor<S>

  Returns an element-wise exponentiation of ``op``.

.. function:: dot(lhs:Tensor<S>, rhs:Tensor<S>) -> NumericType
              ⊙(lhs:Tensor<S>, rhs:Tensor<S>) -> NumericType

  Returns the dot product between two vectors.
  Both ``lhs`` and ``rhs`` must be vectors.

.. function:: outer(lhs:Tensor<S>, rhs:Tensor<S>) -> NumericType
              ⊗(lhs:Tensor<S>, rhs:Tensor<S>) -> NumericType

  Returns the outer product between two vectors.
  Both ``lhs`` and ``rhs`` must be vectors.

.. function:: abs(op:Tensor<S>) -> Tensor<S>

  Performs an elementwise ``abs`` on Tensor.

.. function:: concat(op1:Tensor<S>, op2:Tensor<S>, ..., opN:Tensor<S>, axis:Int)

  Concats N Tensors along ``axis``.

.. function:: vstack(op1:Tensor<S>, op2:Tensor<S>) -> Tensor<S>

  Returns a new Tensor that is a composite of ``op1`` and ``op2`` vertically
  stacked.

.. function:: hstack(op1:Tensor<S>, op2:Tensor<S>) -> Tensor<S>

  Returns a new Tensor that is a composite of ``op1`` and ``op2`` horizontally
  stacked.

.. function:: broadcast(op1:Tensor<S>, op2:Tensor<S>) -> (Tensor<S>, Tensor<S>)

  Returns two Tensors of the same shape broadcasted from ``op1`` and ``op2``.
  Need to go into much more detail about broadcasting.

.. function:: reduce(op:Tensor<S>, fn:(Tensor<S>, Tensor<S>) -> NumericType) -> NumericType

  Applies ``fn`` to elements of Tensor, returns a scalar.

.. function:: sum(op:Tensor<S>, axis:Int) -> Tensor<S>
              sum(op:Tensor<S>) -> NumericType

  When ``axis`` is specified, sums ``op`` along ``axis`` and returns resulting
  Tensor. When no ``axis`` is specified, returns entire Tensor summed.

.. function:: max(op:Tensor<S>, axis:Int) -> Tensor<S>
              max(op:Tensor<S>) -> NumericType

  When ``axis`` is specified, find the maximum value ``op`` across ``axis``
  and returns resulting Tensor. When no ``axis`` is specified, returns
  maximum value of entire Tensor.

.. function:: min(op:Tensor<S>, axis:Int) -> Tensor<S>
              min(op:Tensor<S>) -> NumericType

  When ``axis`` is specified, find the minimum value ``op`` across ``axis``
  and returns resulting Tensor. When no ``axis`` is specified, returns
  minimum value of entire Tensor.

.. function:: fill(op:Tensor<S>, value:NumericType)

  Sets all elements of ``op`` to ``value``.

.. function:: copy(op:Tensor<S>) -> Tensor<S>
              copy(from:Tensor<S>, to:Tensor<S>)

    First form creates a new Tensor and copies ``op`` into it. The second form
    copies ``op`` into an already existing Tensor.

.. function:: map(op:Tensor<S>, fn:(NumericType) -> NumericType) -> Tensor<S>

  Applies ``fn`` to each element of Tensor and returns resulting Tensor.

TODO
+++++
* norm(|Tensor|, axis: ``axis``)
* hist(|Tensor|, bins:``Int``) -> |Vector|
* isClose(``Tensor1``, ``Tensor2``) -> ``Bool``

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
