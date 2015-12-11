Random library
==============
To generate a random number you first need to create a `RandomNumberGenerator`::

	let rng = RandomNumberGenerator(seed)

The `seed` variable is optional, and if left out will be set the system's clock. 

The `random` library can be used in two ways:

	1. Fill an already existing `Tensor` with random values
	2. Create a new `Tensor` that is filled with random values
