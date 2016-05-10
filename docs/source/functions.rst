================
Tensor Functions
================

Basic Math
----------

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

.. function:: abs(op:Tensor<S>) -> Tensor<S>

  Performs an elementwise ``abs`` on Tensor.

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


Linear Algerbra
---------------

.. function:: dot(lhs:Tensor<S>, rhs:Tensor<S>) -> NumericType
              ⊙(lhs:Tensor<S>, rhs:Tensor<S>) -> NumericType

  Returns the dot product between two vectors.
  Both ``lhs`` and ``rhs`` must be vectors.

.. function:: outer(lhs:Tensor<S>, rhs:Tensor<S>) -> NumericType
              ⊗(lhs:Tensor<S>, rhs:Tensor<S>) -> NumericType

  Returns the outer product between two vectors.
  Both ``lhs`` and ``rhs`` must be vectors.


Tensor Manipulation
-------------------
.. function:: fill(op:Tensor<S>, value:NumericType)

  Sets all elements of ``op`` to ``value``.

.. function:: copy(op:Tensor<S>) -> Tensor<S>
              copy(from:Tensor<S>, to:Tensor<S>)

    First form creates a new Tensor and copies ``op`` into it. The second form
    copies ``op`` into an already existing Tensor.

.. function:: map(op:Tensor<S>, fn:(NumericType) -> NumericType) -> Tensor<S>

  Applies ``fn`` to each element of Tensor and returns resulting Tensor.

.. function:: reduce(op:Tensor<S>, fn:(Tensor<S>, Tensor<S>) -> NumericType) -> NumericType

  Applies ``fn`` to elements of Tensor, returns a scalar.

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

Statistics
----------

.. function:: norm(op:Tensor, axis:Int) -> Tensor<S>

  Calculates the norm along specified axis.

.. function:: hist(op:Tensor, bins:[Int]) -> Tensor<S>

  Returns a vector with size of ``bins`` with the resulting histogram of ``op``.


Other
-----

.. function:: isClose(lhs:Tensor<S>, rhs:Tensor<S>, eps:NumericType) -> Bool

  Returns if ``lhs`` is within the range of (``rhs`` + ``eps``, ``rhs`` - ``eps``)
