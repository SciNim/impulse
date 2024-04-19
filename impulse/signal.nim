## Module that implements several signal processing and related functions.
##
## The implementations of many of these functions have been based on the
## corresponding functions in the Numpy and Scipy libraries.

import arraymancer
import std / strformat

# Constants used to implement the Modified Bessel function of the first kind
let i0A_values = [
    -4.41534164647933937950E-18,
    3.33079451882223809783E-17,
    -2.43127984654795469359E-16,
    1.71539128555513303061E-15,
    -1.16853328779934516808E-14,
    7.67618549860493561688E-14,
    -4.85644678311192946090E-13,
    2.95505266312963983461E-12,
    -1.72682629144155570723E-11,
    9.67580903537323691224E-11,
    -5.18979560163526290666E-10,
    2.65982372468238665035E-9,
    -1.30002500998624804212E-8,
    6.04699502254191894932E-8,
    -2.67079385394061173391E-7,
    1.11738753912010371815E-6,
    -4.41673835845875056359E-6,
    1.64484480707288970893E-5,
    -5.75419501008210370398E-5,
    1.88502885095841655729E-4,
    -5.76375574538582365885E-4,
    1.63947561694133579842E-3,
    -4.32430999505057594430E-3,
    1.05464603945949983183E-2,
    -2.37374148058994688156E-2,
    4.93052842396707084878E-2,
    -9.49010970480476444210E-2,
    1.71620901522208775349E-1,
    -3.04682672343198398683E-1,
    6.76795274409476084995E-1
  ]

let i0B_values = [
    -7.23318048787475395456E-18,
    -4.83050448594418207126E-18,
    4.46562142029675999901E-17,
    3.46122286769746109310E-17,
    -2.82762398051658348494E-16,
    -3.42548561967721913462E-16,
    1.77256013305652638360E-15,
    3.81168066935262242075E-15,
    -9.55484669882830764870E-15,
    -4.15056934728722208663E-14,
    1.54008621752140982691E-14,
    3.85277838274214270114E-13,
    7.18012445138366623367E-13,
    -1.79417853150680611778E-12,
    -1.32158118404477131188E-11,
    -3.14991652796324136454E-11,
    1.18891471078464383424E-11,
    4.94060238822496958910E-10,
    3.39623202570838634515E-9,
    2.26666899049817806459E-8,
    2.04891858946906374183E-7,
    2.89137052083475648297E-6,
    6.88975834691682398426E-5,
    3.36911647825569408990E-3,
    8.04490411014108831608E-1
  ]

proc chbevl[T: SomeFloat](x: T, vals: openArray[T]): T =
  var b0 = 0.0
  var b1 = 0.0
  var b2 = 0.0

  for it in vals:
    b2 = b1
    b1 = b0
    b0 = x * b1 - b2 + it

  return 0.5 * (b0 - b2)

proc i0_1[T: SomeFloat](x: T): T =
  ## "Inner" function for the Modified Bessel function of the first kind
  return exp(x) * chbevl(x / 2.0 - 2.0, i0A_values)

proc i0_2[T: SomeFloat](x: T): T =
  ## "Outer" function for the Modified Bessel function of the first kind
  return exp(x) * chbevl(32.0 / x - 2.0, i0B_values) / sqrt(x)

proc i0*[T: SomeFloat](x: T): T =
  ## Modified Bessel function of the first kind, order 0.
  ##
  ## Inputs:
  ## - x: Tensor of floats
  ##
  ## Result:
  ##   The modified Bessel function evaluated at each of the elements of `x`.
  ##
  ## Notes:
  ##
  ## This implementation is a port of scipy's `i0` function, which in turn
  ## uses the algorithm published by Clenshaw [1]_ and referenced by
  ## Abramowitz and Stegun [2]_, for which the function domain is
  ## partitioned into the two intervals [0,8] and (8,inf), and Chebyshev
  ## polynomial expansions are employed in each interval. Relative error on
  ## the domain [0,30] using IEEE arithmetic is documented [3]_ as having a
  ## peak of 5.8e-16 with an rms of 1.4e-16 (n = 30000).
  ##
  ## References:
  ## - [1] C. W. Clenshaw, "Chebyshev series for mathematical functions", in
  ##         *National Physical Laboratory Mathematical Tables*, vol. 5, London:
  ##         Her Majesty's Stationery Office, 1962.
  ## - [2] M. Abramowitz and I. A. Stegun, *Handbook of Mathematical
  ##         Functions*, 10th printing, New York: Dover, 1964, pp. 379.
  ##         https://personal.math.ubc.ca/~cbm/aands/page_379.htm
  ## - [3] https://metacpan.org/pod/distribution/Math-Cephes/lib/Math/Cephes.pod#i0:-Modified-Bessel-function-of-order-zero
  ##
  ## Examples:
  ## ```nim
  ## i0(0)
  ## # 1.0
  ## >>> i0([0, 1, 2, 3].toTensor)
  ## # Tensor[system.float] of shape "[4]" on backend "Cpu"
  ## #     1    1.26607    2.27959    4.88079
  ## ```
  if abs(x) <= 8.0:
    return i0_1(x)
  else:
    return i0_2(x)

makeUniversal(i0)

proc kaiser*[T: SomeFloat](size: int = 10, beta: T = 5.0): Tensor[T] =
  ## Return the Kaiser window of the given `size` and `beta` parameter
  ##
  ## The Kaiser window is a "taper" function formed by using a Bessel function.
  ##
  ## Input:
  ## - size: Number of points (or samples) in the output window. If zero,
  ##      an empty Tensor is returned. If negative an assertion is raised.
  ## - beta: "Shape" parameter for the window (as a float). Defaults to 5.0,
  ##         which generates a window similar to a Hamming.
  ##
  ## Result:
  ## - The generated Kaiser window, as a float Tensor with the maximum value
  ##   normalized to 1.0 (the value 1.0 only appears if the number of samples is odd).
  ##
  ## Notes:
  ## 1. The Kaiser window is defined as
  ##
  ## >  w(n) = I_0\\left( \\beta \\sqrt{1-\\frac{4n^2}{(M-1)^2}}
  ##             \\right)/I_0(\\beta)
  ##
  ## with
  ##
  ## > \\quad -\\frac{M-1}{2} \\leq n \\leq \\frac{M-1}{2},
  ##
  ## where `I_0` is the modified zeroth-order Bessel function.
  ##
  ## 2. The Kaiser window was named after Jim Kaiser, who discovered a simple
  ## approximation to the DPSS window based on Bessel functions. The Kaiser
  ## window is a very good approximation to the Digital Prolate Spheroidal
  ## Sequence, or Slepian window, which is the transform which maximizes the
  ## energy in the main lobe of the window relative to total energy.
  ##
  ## 3. The Kaiser window can approximate many other windows by varying the beta
  ## parameter.
  ##
  ## ====  =======================
  ## beta  Window shape
  ## ====  =======================
  ## 0.0   Rectangular
  ## 5.0   Similar to a Hamming
  ## 6.0   Similar to a Hanning
  ## 8.6   Similar to a Blackman
  ## ====  =======================
  ##
  ## A `beta` value of 14.0 is probably a good starting point if you do not
  ## want to use the default or one of the values in the list above.
  ##
  ## 4. As `beta` gets large, the window narrows, and so the number of samples
  ## needs to be large enough to sample the increasingly narrow spike,
  ## otherwise some `nan` values will be returned.
  ##
  ## 5. Most references to the Kaiser window come from the signal processing
  ## literature, where it is used as one of many windowing functions for
  ## smoothing values. It is also known as an apodization (which means
  ## "removing the foot", i.e. smoothing discontinuities at the beginning
  ## and end of the sampled signal) or tapering function.
  ##
  ## 5. This implementation is a port of scipy's `kaiser` function.
  ##
  ## References:
  ## - [1] J. F. Kaiser, "Digital Filters" - Ch 7 in "Systems analysis by
  ##       digital computer", Editors: F.F. Kuo and J.F. Kaiser, p 218-285.
  ##       John Wiley and Sons, New York, (1966).
  ## - [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
  ##       University of Alberta Press, 1975, pp. 177-178.
  ## - [3] Wikipedia, "Window function",
  ##       https://en.wikipedia.org/wiki/Window_function
  ## - [4] Code heavily inspired by (but much faster than) Numpy's
  ##       implementation,
  ##       https://github.com/numpy/numpy/blob/v1.26.0/numpy/lib/function_base.py#L3493
  ##
  ## Examples:
  ## ```nim
  ## import impulse / signal
  ## echo kaiser(4)
  ## # Tensor[system.float] of shape "[4]" on backend "Cpu"
  ## # 0.0367109     0.775322     0.775322    0.0367109
  ##
  ## echo kaiser(12, 14.0)
  ## # Tensor[system.float] of shape "[12]" on backend "Cpu"
  ## #    7.72687e-06    0.00346009     0.04652        0.229737
  ## #    0.599885       0.945675       0.945675       0.599885
  ## #    0.229737       0.04652        0.00346009     7.72687e-06
  ## ```

  doAssert size >= 0, "The size of the Kaiser window must be non-negative"
  if size == 1:
    return [T(1)].toTensor

  let n_vals = arange(T(0.0), T(size))
  let alpha = (T(size) - 1.0) / 2.0
  return i0(beta * sqrt(1.0 -. ((n_vals -. alpha) / alpha) ^. 2.0)) /. i0(beta)
