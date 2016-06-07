Neural Network library
======================

Overview
--------
The basic building block is an ``Op``, which is parameterized by the underyling storage type (see the ``Tensor`` documentation for more details). Every ``Op`` as the function ``apply``, which performs a transform on the ``Op`` s input updating the `output` variable. Currently ``Op`` s may have a variable number of inputs with a single output.

.. figure:: ../images/op_diagram.png
  :width: 400
  :align: center

  **Figure 1**: Computation graph formed by connecting multiple ``Op`` s together.

The design of the ``Op`` class is keep the details of how the optimization is performed separate from the transform itself. The protocol ``Differentiable`` provides a method to extend the ``Op`` class with the ``gradient`` method. The ``gradient`` methods returns a new ``Op`` whose transform is the derivative of the target ``Op``. For example:

.. code:: swift

  let linear = Linear<D>(inputSize: 5, outputSize: 3)
  let linearGrad = linear.gradient()

Some ``Op`` s, like ``Sequence`` are collections of other ``Op`` s, and will invoke (via ``apply``) each of the contained ``Op`` s in a pre-determined order. For example, a standard neural network can be created with:

.. code:: swift

  let net = Sequence<D>(
    Symbol<D>(zeros(Extent(5))),
    Linear<D>(inputSize: 5, outputSize: 3),
    Sigmoid<D>(size: 3),
    L2Loss<D>(size: 3))

  net.apply()


There are a few important items of note in this example:

1. The ``apply`` method takes no parameters and has no return value
2. The input to the first layer of the network is of the type ``Symbol``, which is also an ``Op``
3. The loss function ``L2Loss`` is also an ``Op``

The design decision to make everything an ``Op`` allows the creation of the computational graph. In this case, all of the ``Op`` s are also ``Differentiable``, and thus you can do:

.. code:: swift

  let netGrad = net.gradient()

Optimization can be performed with the following code:

.. code:: swift

  params = net.params()
  gradParams = netGrad.params()

  for i in 0..<iterations {
    netGrad.reset()

    net.apply()
    netGrad.apply()

    for (param, gradParam) in Zip2Sequences(params, gradParams) {
      param += 0.01*gradParam
    }
  }

However, this code can be simplified using an ``Optimizer``:

.. code::

  let alpha = Symbol<D>(0.01)
  let opt = GradientDescentOptimizer(net, alpha: alpha)

  for i in 0..<iterations {
    opt.apply()
  }

where ``GradientDescentOptimizer`` automatically constructs the gradient network and collects the parmaeters for both the forward and backward sequences.

One of the advantages to having everything an operation in the computation graph is that the ``alpha`` symbol can be set dynamically. For example, if a momentum optimization is desired, the ``alpha`` symbol can be computed from the current error.

The Op class
----------------
The ``Op`` class has the following properties:

* id: unique ID for instance of ``Op``
* inputs: collection of ``Op`` s
* output: result of the transform

and has the following methods defined:

* apply(): performs transform on inputs and stores results in ``output``
* params(): returns all the parameters of the transform (e.g. if its a ``Linear`` Op, then the parameters are ``weight`` and ``bias``).

----------
Op library
----------
.. function:: Linear

  Performs a linear transformation on input.

.. function:: Sigmoid

  Applies the sigmoid function to each element of the input.

.. function:: L2Loss

  Takes two inputs: ``value`` and ``target``. Calculates the square distance between the two.

-----------------
Creating a new Op
-----------------
Suppose you wanted to create an ``Op`` that takes the log of the input. The ``Log`` op can be defined as:

.. code:: swift

  public class Log<S:Storage where S.ElementType:FloatNumericType>: Op<S> {
    public init(size:Int) {
      super.init( inputs: [NoOp<S>()],
                  output: Tensor<S>(Extent(size)),
                  labels: ["input"])
    }

    public override func apply() {
      if output == nil || output!.shape != inputs[0].output!.shape {
        output = Tensor<S>(Extent(inputs[0].output!.shape))
      }

      log(inputs[0].output!, result: output!)
    }
  }

where the initialization defines a single input (``input``) that is currently not defined (the ``NoOp``) and the output is allocated as the size specified by the parameter. The ``apply`` function finds the maximum value in the input, divides each element of the input by that value, and stores in the result in ``output``.

The gradient of ``Log`` can be defined as:

.. code::Swift

  public class LogGrad<S:Storage where S.ElementType:FloatNumericType>: Op<S>, Gradient {
    public required init(op:Log<S>) {
      super.init( inputs: [op, op.inputs[0], NoOp<S>()],
                  output: Tensor<S>(op.output!.shape),
                  labels: ["op", "input", "gradOutput"])
    }

    public override func apply() {
      fill(output!, value: 1)
      output! /= inputs[1].output!
      output! *= inputs[2].output!
    }

    public func reset() {
      fill(output!, value: 0)
    }
  }

The ``Log`` gradient takes two additional inputs: the instance of the ``Log`` op its the gradient of, and ``gradOutput``, which is the gradient of the op's output.

Finally, to allow the gradient to be taken of ``Log``, the class must be extended to ``Differentiable``:

.. code:: swift

  extension Log:Differentiable {
    public func gradient() -> GradientType {
      return LogGrad<S>(op: self)
    }
  }

We can change the construction of our network by adding ``Log`` into the sequence:

.. code:: swift

  let net = Sequence<D>(
    Symbol<D>(zeros(Extent(5))),
    Log<D>(size: 5)
    Linear<D>(inputSize: 5, outputSize: 3),
    Sigmoid<D>(size: 3),
    L2Loss<D>(size: 3))

and have the optimization correctly calculate the derivative as before:

.. code:: swift

  let opt = GradientDescentOptimizer(net, alpha: alpha)

because ``GradientDescentOptimizer`` will automatically call ``gradient`` on each ``Op``, an instance of ``LogGradient`` will be created for each instance of ``Log``.

-----------------
Testing your Op
-----------------
It is always a good idea to do a gradient check on a newly created ``Op``. You can create a new unit test to do so:

.. code:: swift

  func testLogOpGradient() {
    let eps = 10e-6
    let input = Symbol<S>(uniform(Extent(10)))
    let gradOutput = Symbol<S>(zeros(Extent(10)))

    let log = Log<S>(size: 10)
    log.setInput("input", to: input)

    let logGrad = log.gradient() as! LogGrad<S>
    logGrad.setInput("gradOutput", to: gradOutput)

    // test gradient wrt to the input
    let inputError = checkGradient(log, grad: logGrad, params: input.output, gradParams: logGrad.output, eps: eps)
    XCTAssertLessThan(inputError, eps)
  }
