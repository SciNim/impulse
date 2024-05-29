## Module that implements several procedures related to prime numbers
##
## Prime numbers are an essential building block of many algorithms in diverse
## areas such as cryptography, digital communications and many others.
## This module adds a function to generate rank-1 tensors of primes upto a
## certain value.

import arraymancer

proc primes*[T: SomeInteger | SomeFloat](upto: T): Tensor[T] =
  ## Generate a Tensor of prime numbers up to a certain value
  ##
  ## Return a Tensor of the prime numbers less than or equal to `upto`.
  ## A prime number is one that has no factors other than 1 and itself.
  ##
  ## Input:
  ##   - upto: Integer up to which primes will be generated
  ##
  ## Result:
  ##   - Integer Tensor of prime values less than or equal to `upto`
  ##
  ## Note:
  ##   - This function implements a "half" Sieve of Erathostenes algorithm
  ##     which is a classical Sieve of Erathostenes in which only odd numbers
  ##     are checked. Many examples of this algorithm can be found online.
  ##     It also stops checking after sqrt(upto)
  ##   - The memory required by this procedure is proportional to the input
  ##     number.
  when T is SomeFloat:
    if upto != round(upto):
      raise newException(ValueError,
        "`upto` value (" & $upto & ") must be a whole number")

  if upto < 11:
    # Handle the primes below 11 to simplify the general code below
    # (by removing the need to handle the few cases in which the index to
    # `isprime`, calculated based on `factor` is negative)
    # This is the minimum set of primes that we must handle, but we could
    # extend this list to make the calculation faster for more of the
    # smallest primes
    let prime_candidates = [2.T, 3, 5, 7].toTensor()
    return prime_candidates[prime_candidates <=. upto]

  # General algorithm (valid for numbers higher than 10)
  let prime_candidates = arange(3.T, T(upto + 1), 2.T)
  var isprime = ones[bool]((upto.int - 1) div 2)
  let max_possible_factor_idx = int(sqrt(upto.float)) div 2
  for factor in prime_candidates[_ ..< max_possible_factor_idx]:
    if isprime[(factor.int - 2) div 2]:
      isprime[(factor.int * 3 - 2) div 2 .. _ | factor.int] = false

  # Note that 2 is missing from the result, so it must be manually added to
  # the front of the result tensor
  return [2.T].toTensor().append(prime_candidates[isprime])
