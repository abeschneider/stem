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
  let m1 = v1*v2

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
    typealias ElementType:NumericType

    init(size:Int)
    init(array:[ElementType])
    init(storage:Self)
    init(storage:Self, copy:Bool)

    var size:Int { get }
    subscript(index:Int) -> ElementType {get set}
    func calculateStride(shape:Extent) -> [Int]
  }

An implementation of ``Storage`` determines the allocation through the ``init``
methods, ``subscript`` determines how the storage gets indexed, and ``calculateStride``
allows the ``Storage`` to be iterated through in a sequential fashion.

The |Tensor| class frequently makes use of the generator ``TensorStorageIndex`` to iterate
through the ``Storage`` class. This provides a convenient way to access all the
elements without knowing the underyling memory allocation.

To do so, the |Tensor| class defined the methid:

.. code:: swift

  public func storageIndices() -> GeneratorSequence<TensorStorageIndex<StorageType>> {
    return GeneratorSequence<TensorStorageIndex<StorageType>>(TensorStorageIndex<StorageType>(self))
  }

which can be used like:

.. code:: swift

  func fill<StorageType:Storage>(tensor:Tensor<StorageType>, value:StorageType.ElementType) {
      for i in tensor.storageIndices() {
          tensor.storage[i] = value
      }
  }

However, as mentioned previously, if an optimized version for a particular |Tensor|
operation exists, you can write:

.. code:: swift

  // This will be used if the Tensor's storage type is CBlasStorage for doubles,
  // an alternative can be specified for Floats separately.
  func fill(tensor:Tensor<CBlasStorage<Double>>, value:StorageType.ElementType) {
    // ..
  }

.. _Tensor_Class:

Tensor Class
------------
The |Tensor| class hold an instance of ``Storage`` along with a view into
the storage. Multiple instances of |Tensor| may point to the same ``Storage``
providing different views of the same data. This allows operations such as indexing
to operate in an efficient manner without requiring copies of the memory to be made.

Subclasses of |Tensor| include:

* |Vector|
* ``RowVector``
* ``ColumnVector``
* |Matrix|

These subclasses provide convience constructors as well a the ability to
provide function overloading to handle special cases (e.g. broadcasting).

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

Vector
++++++

To construct a |Vector| along an arbritrary axis with contents read from an array:

.. code:: swift

  init(_ array:[StorageType.ElementType], axis:Int=0)

To construct a view of a |Vector|:

.. code:: swift

  init(_ vector:Vector<StorageType>, dimIndex:[Int]?=nil, view:StorageView<StorageType>?=nil)

To construct a ``RowVector`` (i.e. a |Vector| that lies along dimension 0) with contents read from an array:

.. code:: swift

  init(_ array:[StorageType.ElementType])

To construct a ``ColumnVector`` (i.e. a |Vector| that lies along dimension 1) with contents read from an array:

.. code:: swift

  init(_ array:[StorageType.ElementType])

Matrix
++++++

To construct a |Matrix| with contents from a 2 dimensional array:

.. code:: swift

  init(_ array:[[StorageType.ElementType]], copyTransposed:Bool=false)

To construct a |Matrix| view:

.. code:: swift

  init(storage:StorageType, shape:Extent, view:StorageView<StorageType>?=nil, offset:Int?=nil)

.. _Functions:

Functions
---------

.. csv-table:: Addition
  :header: "Operator", "Equivalent function", "Broadcasts"
  :widths: 20, 20, 5

  "|Tensor| + |Tensor| -> |Tensor|", "add(left:|Tensor|, right:|Tensor|, result:|Tensor|)", "False"
  "|Matrix| + ``RowVector`` -> |Matrix|", "add(left:|Matrix|, right:``RowVector``, result:|Matrix|)", "True"
  "|Matrix| + ``ColumnVector`` -> |Matrix|", "add(left:|Matrix|, right:``ColumnVector``)", "True"
  "|Tensor| += |Tensor|", "iadd(left:|Tensor|, right:|Tensor|)", "False"
  "|Vector| += |Vector|", "iadd(left:|Vector|, right:|Vector|)", "False"
  "|Matrix| += ``RowVector``", "iadd(left:|Matrix|, right:``RowVector``)", "True"
  "|Matrix| += ``ColumnVector``", "iadd(left:|Matrix|, right:``ColumnVector``)", "True"

.. csv-table:: Subtraction
  :header: "Operator", "Equivalent function", "Broadcasts"
  :widths: 20, 20, 5

  "|Tensor| - |Tensor| -> |Tensor|", "sub(left:|Tensor|, right:|Tensor|, result:|Tensor|)", "False"
  "|Matrix| - ``RowVector`` -> |Matrix|", "sub(left:|Matrix|, right:``RowVector``, result:|Matrix|)", "True"
  "|Matrix| - ``ColumnVector`` -> |Matrix|", "sub(left:|Matrix|, right:``ColumnVector``)", "True"
  "|Tensor| -= |Tensor|", "isub(left:|Tensor|, right:|Tensor|)", "False"
  "|Vector| -= |Vector|", "isub(left:|Vector|, right:|Vector|)", "False"
  "|Matrix| -= ``RowVector``", "isub(left:|Matrix|, right:``RowVector``)", "True"
  "|Matrix| -= ``ColumnVector``", "isub(left:|Matrix|, right:``ColumnVector``)", "True"

.. csv-table:: Elementwise Multiplication
  :header: "Operator", "Equivalent function", "Broadcasts"
  :widths: 20, 20, 5

  "|Tensor| * |Tensor| -> |Tensor|", "mul(left:|Tensor|, right:|Tensor|, result:|Tensor|)", "False"
  "|Tensor| * |Number| -> |Tensor|", "mul(left:|Number|, right:|Tensor|, result:|Tensor|)", "False"
  "|Number| * |Tensor| -> |Tensor|", "mul(left:|Tensor|, right:|Number|, result:|Tensor|)", "False"
  "|Vector| *= |Vector|", "imul(left:|Vector|, right:|Vector|)", "False"
  "|Matrix| *= ``RowVector``", "imul(left:|Matrix|, right:``RowVector``)", "True"
  "|Matrix| *= ``ColumnVector``", "imul(left:|Matrix|, right:``ColumnVector``)", "True"
  "|Tensor| *= |Number|", "imul(left:|Tensor|, right:|Number|)", "False"

.. csv-table:: Elementwise Division
  :header: "Operator", "Equivalent function", "Broadcasts"
  :widths: 20, 20, 5

  "|Tensor| / |Number| -> |Tensor|", "div(left:|Tensor|, right:|Number|, result:|Tensor|)", "False"
  "|Vector| / |Vector| -> |Vector|", "div(left:|Vector|, right:|Vector|, result:|Vector|)", "False"
  "|Matrix| / ``RowVector`` -> |Matrix|", "div(left:|Matrix|, right:``RowVector``, result:|Matrix|)", "True"
  "|Matrix| / ``ColumnVector`` -> |Matrix|", "div(left:|Matrix|, right:``ColumnVector``, result:|Matrix|)", "True"

Elementwise Exponentiation
++++++++++++++++++++++++++
* |Tensor| ^ |Number| -> |Tensor|
* pow(|Tensor|, |Number|) -> |Tensor|
* exp(|Tensor|) -> |Tensor|

Liner Algebra
+++++++++++++
* ``RowVector`` * ``ColumnVector`` -> |Number|
* |Matrix| * ``ColumnVector`` -> ``RowVector``
* |Matrix| * |Matrix| -> ``RowVector``
* dot(|Vector|, |Vector|) -> |Number|
* ``ColumnVector`` * ``RowVector`` -> |Matrix|
* outer(|Vector|, |Vector|) -> |Matrix|

Other
+++++
* abs(|Tensor|) -> |Tensor|
* concat(``Tensor1``, ``Tensor2``, ..., axis: ``axis``)
* vstack(``Tensor1``, ``Tensor2``)
* hstack(``Tensor1``, ``Tensor2``)
* sum(|Tensor|, axis: ``axis``)
* norm(|Tensor|, axis: ``axis``)
* max(|Tensor|, axis: ``axis``)
* fill(|Tensor|, |Number|)
* copy(from:|Tensor|, to:|Tensor|)
* copy(|Tensor|) -> |Tensor|
* map(|Tensor|, (|Number|) -> |Number|) -> |Tensor|
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
