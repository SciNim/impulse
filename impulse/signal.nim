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

proc firls*(fir_order: int,
    bands, desired: Tensor[float],
    weights = newTensor[float](),
    symmetric = true,
    fs = 2.0): Tensor[float] =
  ## Design a linear phase FIR filter using least-squares error criterion
  ##
  ## Returns the coefficients of the linear phase FIR filter of the requested
  ## order and type which best fullfills the desired frequency response
  ## "constraints" described by the `bands` (i.e. frequency), `desired` (i.e. gain)
  ## and `weights` (e.i. constraint weights) input tensors.
  ##
  ## Depending on the order and the value of the `symmetric` argument, the
  ## output filter coefficients will correspond to a Type I, II, III or IV
  ## FIR filter.
  ##
  ## The constraints are specified as a pair of tensors describing "constrained
  ## frequency bands" indicating the start and end frequencies of each band `k`,
  ## as well as the gain of the filter at those edge frequencies.
  ##
  ## Inputs:
  ##   - fir_order: The "order" of the filter (which is 1 less than the length
  ##                of the output tensor).
  ##   - bands: 2-column matrix of non-overlapping, normalized frequency band
  ##            edges in ascending order. Each row in the matrix corresponds to
  ##            the edges of a constrained frequency band (the column contains
  ##            the start frequency of the band and the second column contains
  ##            the end frequency).
  ##            Frequencies between the specified bands are "unconstrained"
  ##            (i.e. ignored during the error minimization process).
  ##            Frequency values must be in the [0.0, fs/2] range (i.e. they
  ##            cannot be negative or exceed the Nyquist frequency).
  ##   - desired: 2-column matrix that specifies the gains of the desired
  ##              frequency response at the band "edges". Thus the lengths of
  ##              the `bands` and `desired` tensors must match. If they do not,
  ##              an exception is raised.
  ##              For each band `k` the desired filter frequency
  ##              response is such that its gain linearly changes from the
  ##              `desired[k, 0]` value at the start of the band (i.e. at the
  ##              `bands[k, 0]` frequency) to the value `desired[k, 1]` at the
  ##              end of the band (i.e. at `bands[k, 1]`).
  ##   - weights: Optional rank-1 Tensor of weights. Controls which frequency
  ##              response "contraints" are given more "weight" during the
  ##              least-squares error minimization process. The default is that
  ##              all constraints are given the same weight. If provided, its
  ##              length must be half the length of `bands` (i.e. there must be
  ##              one constraint per band). An exception is raised otherwise.
  ##   - symmetric: When `true` (the default), the result will be a symmetric
  ##                FIR filter (Type I when `fir_order` is even and Type II
  ##                when `fir_order` is odd). When `false`, the result will be
  ##                an anti-symmetric FIR filter (Type III when `fir_order` is
  ##                even and Type IV when `fir_order` is odd).
  ##   - fs: The sampling frequency of the signal (as a float). Each frequency
  ##         in `bands` must be between 0.0 and `fs/2` (inclusive).
  ##         Default is 2.0, which means that by default the band frequencies
  ##         are expected to be on the 0.0 to 1.0 range.
  ##
  ## Result:
  ##   - A Tensor containing the `fir_order + 1` taps of the FIR filter that
  ##     best approximates the desired frequency response (i.e. the filter that
  ##     minimizes the least-squares error vs the given constraints).
  ##
  ## Notes:
  ##   - Contrary to numpy's firls, the first argument is the FIR filter order,
  ##     not the FIR length. The filter length is `filter_order + 1`.
  ##   - Frequencies between "constrained bands" are considered "don't care"
  ##     regions for which the error is not minimized.
  ##   - When designing a filter with a gain other than zero at the Nyquist
  ##     frequency (i.e. when `bands[^1] != 0.0`), such as high-pass and
  ##     band-stop filters, the filter order must be even. An exception is
  ##     raised otherwise.
  ##   - The `bands` and `desired` can also be flat rank-1 tensors, as long as
  ##     their length is even (so that they can be reshaped into 2 column
  ##     matrices.
  ##
  ## Examples:
  ## ```nim
  ## # Example of an order 4, Type I, low-pass FIR filter which targets being
  ## # a pass-through in the 0.0 to 0.3 times Nyquist frequency range, while
  ## # filtering frequencies in the 0.4 to 1.0 Nyquist frequency range
  ## # Note that the range 0.3 to 0.4 is left unconstrained
  ## # Also note how the result length is 6 and the filter is symmetric
  ## # around the middle value:
  ## echo firls(4, [[0.0, 0.3], [0.4, 1.0]].toTensor, [[1.0, 1.0], [0.0, 0.0]].toTensor)
  ## # Tensor[system.float] of shape "[5]" on backend "Cpu"
  ## #     0.126496    0.278552    0.345068    0.278552    0.126496
  ##
  ## # Same filter as above, but using rank-1, even length tensors as inputs
  ## echo firls(4, [0.0, 0.3, 0.4, 1.0].toTensor, [1.0, 1.0, 0.0, 0.0].toTensor)
  ## # Tensor[system.float] of shape "[5]" on backend "Cpu"
  ## #     0.126496    0.278552    0.345068    0.278552    0.126496
  ##
  ## # Same filter as above, but using unnormalized frequencies and a sampling
  ## # frequency of 10.0 Hz (note how all the frequencies are 5 times greater
  ## # because the default fs is 2.0):
  ## echo firls(4,
  ##   [[0.0, 1.5], [2.0, 5.0]].toTensor,
  ##   [[1.0, 1.0], [0.0, 0.0]].toTensor,
  ##   fs = 10.0)
  ## # Tensor[system.float] of shape "[5]" on backend "Cpu"
  ## #     0.126496    0.278552    0.345068    0.278552    0.126496
  ##
  ## # Same kind of filter, but give more weight to the pass-through constraint
  ## echo firls(4,
  ##   [[0.0, 0.3], [0.4, 1.0]].toTensor,
  ##   [[1.0, 1.0], [0.0, 0.0]].toTensor,
  ##   weights = [1.0, 0.5].toTensor)
  ## # Tensor[system.float] of shape "[5]" on backend "Cpu"
  ## #     0.110353    0.284473    0.360868    0.284473    0.110353
  ##
  ## # A more complex, order 5, Type II, low-pass filter
  ## echo firls(5,
  ##   [[0.0, 0.3], [0.3, 0.6], [0.6, 1.0]].toTensor,
  ##   [[1.0, 1.0], [1.0, 0.2], [0.0, 0.0]].toTensor)
  ## # Tensor[system.float] of shape "[6]" on backend "Cpu"
  ## #     -0.0560333      0.146107      0.430716      0.430716      0.146107    -0.0560333
  ##
  ## # Example of an order 6 Type IV high-pass FIR filter:
  ## echo firls(6,
  ##   [[0.0, 0.4], [0.6, 1.0]].toTensor, [[0.0, 0.0], [0.9, 1.0]].toTensor,
  ##   symmetric = false)
  ## # Tensor[system.float] of shape "[7]" on backend "Cpu"
  ## #    -0.13945   0.285186  -0.258596   0   0.258596   -0.285186   0.13945
  ##
  ## Trying to design a high-pass filter with odd order generates an exception:
  ## echo firls(5,
  ##   [0.0, 0.5, 0.6, 1.0].toTensor, [0.0, 0.0, 1.0, 1.0].toTensor,
  ##   symmetric = false)
  ## # Filter order (5) must be even when the last
  ## # frequency is 1.0 and its gain is not 0.0 [ValueError]
  ## ```
  if fs <= 0:
    raise newException(ValueError,
      "Sampling frequency fs must be positive but got " & $fs)

  var bands = bands.flatten()
  let desired = desired.flatten()
  if bands.rank > 1:
    raise newException(ValueError,
      "Frequency band tensor rank (" & $bands.rank & ") is not 1")
  if desired.rank > 1:
    raise newException(ValueError,
      "Desired gain pair tensor rank (" & $desired.rank & ") is not 1")

  let nyquist_freq = fs / 2.0
  if max(bands) > nyquist_freq or min(bands) < 0.0:
    raise newException(ValueError,
      "Some frequency values are outside of the valid [0.0, " &
      $nyquist_freq & "] range:\n" &
      $bands)

  if (bands.len mod 2) != 0:
    raise newException(ValueError, "Frequency band tensor length (" &
      $bands.len & ") is not even")

  if bands.len != desired.len:
    raise newException(ValueError,
      "Frequency and desired gain tensors must have the same length (" &
      $bands.len & "!=" & $desired.len & ")")

  # Check whether the filter order is valid
  let even_order = (fir_order mod 2) == 0

  if not even_order and bands[^1].item == 1.0 and desired[^1].item != 0.0:
    raise newException(ValueError,
      "Filter order (" & $fir_order &
      ") must be even when the last frequency is 1.0 and its desired gain is not 0.0")

  let weights = if weights.len == 0:
      # When weights are not provided, make them constant
      # (i.e. all bands have the same weight)
      ones[float](bands.len div 2)
    else:
      weights
  if bands.len != 2 * weights.len:
    raise newException(ValueError,
      "Weigth tensor length (" & $weights.len &
      ") must be half the length of the frequency band and desired gain pair tensor lenghts (" &
      $bands.len & ")")

  # Note that df is only used
  let df = diff_discrete(bands, axis=0)
  if any(df <. 0.0):
    raise newException(ValueError,
      "Frequency band tensor values must increasing monotonically")

  # We must normalize the band frequencies to the Nyquist frequency, but we
  # must _also_ multiply them by 2.0. The result is that (internally to this
  # function) we normalize them by the sampling frequency, but externally
  # (i.e. in the function documentation) we say that we normalize to the
  # Nyquist frequency
  bands /= fs # Equivalent to `bands /= (0.5 * fs) * 2.0`!
  let constant_weights = all((weights -. weights[0]) ==. 0.0)
  # If there are no gaps between the constrained bands we consider that the
  # constraints are "continuous"
  let continuous_constraints = df.len == 1 or all(df[1 ..< df.len | 2] ==. 0.0)
  # Only use the matrix inversion method when needed, since it is much slower
  let requires_matrix_inversion = not (constant_weights and continuous_constraints)

  # The half "length" is the length of the "half" of the filter that is
  # symmetric or anti-symmetric with the other "half". "Half" is in quotes
  # because when fir_order is even an additional middle coefficient must
  # be added between the two "halfs"
  let half_length = fir_order / 2

  if symmetric:
    # Symmetric (i.e. Type I or Type II) linear phase FIR
    # Type I when fir_order is even, Type II otherwise
    var k = arange(floor(half_length) + 1.0)
    if not even_order:
      k +.= 0.5

    var sk_pos, sk_neg, q: Tensor[float]
    if requires_matrix_inversion:
      let k_right_tile = 2.0 * k.reshape_infer(-1, 1).tile(1, k.len)
      let k_down_tile = 2.0 * k.tile(k.len, 1)
      sk_pos = k_right_tile + k_down_tile
      sk_neg = k_right_tile - k_down_tile
      q = zeros[float](sk_pos.shape)
    if even_order:
      k = k[1.._]

    # Note that we order the operations and make some pre-calculations
    # to minimize the number of tensor ops
    let trig_base = 2.0 * PI * k
    let k_square_inv = 1.0 /. (k *. k)
    var b0 = 0.0
    var b = zeros[float](k.shape)
    for idx in countup(0, bands.len - 1, 2):
      let weight_pow = abs(weights[(idx+1) div 2])
      if requires_matrix_inversion:
        q +=
          weight_pow * 0.5 * bands[idx+1] * (sinc(sk_pos * bands[idx+1]) + sinc(sk_neg * bands[idx+1])) -
          weight_pow * 0.5 * bands[idx] * (sinc(sk_pos * bands[idx]) + sinc(sk_neg * bands[idx]))
      let slope = (desired[idx+1] - desired[idx]) / (bands[idx+1] - bands[idx])
      let b1 = desired[idx] - slope * bands[idx]
      if even_order:
        b0 += weight_pow *
          (b1 * (bands[idx+1] - bands[idx]) + slope / 2.0 * (bands[idx+1]^2 - bands[idx]^2))
      b += weight_pow *
        (slope / (4.0 * PI^2) * (cos(trig_base * bands[idx+1]) - cos(trig_base * bands[idx])) *. k_square_inv)
      b +=
        weight_pow * bands[idx+1] * (slope * bands[idx+1] + b1) * sinc(2.0 * k * bands[idx+1]) -
        weight_pow * bands[idx] * (slope * bands[idx] + b1) * sinc(2.0 * k * bands[idx])

    if even_order:
      b = concat([[b0].toTensor, b], axis = 0)

    var a: Tensor[float]
    if requires_matrix_inversion:
      a = solve(q, b)
    else:
      a = 4.0 * weights[0] * b
      if even_order:
        a[0] /= 2.0

    result = if even_order:
        # Type I FIR filter: symmetric, even order, odd length
        # Note that half_length, despite being a float, is an exact number when
        # fir_order is even!
        concat([a[half_length.int .. 1 | -1] * 0.5,
          [a[0]].toTensor,
          a[1 .. half_length.int] * 0.5], axis = 0)
      else:
        # Type II FIR filter: symmetric, odd order, even length
        0.5 * concat([a[_|-1], a], axis = 0)
  else:
    # Anti-symmetric (i.e. Type III or Type IV) linear phase FIR
    # Type III when fir_order is even, Type IV otherwise
    var k = if even_order:
        # Note that half_order is float but it is an exact number!
        arange(1.0, floor(half_length) + 1.0)
      else:
        arange(floor(half_length) + 1.0) +. 0.5

    var sk_pos, sk_neg, q: Tensor[float]
    if requires_matrix_inversion:
      let k_right_tile = 2.0 * k.reshape_infer(-1, 1).tile(1, k.len)
      let k_down_tile = 2.0 * k.tile(k.len, 1)
      # sk_pos is the base argument for the "positive" sincs
      sk_pos = k_right_tile + k_down_tile
      # sk_neg is the base argument for the "negative" sincs
      sk_neg = k_right_tile - k_down_tile
      q = zeros[float](sk_pos.shape)

    # Note that we order the operations and make some pre-calculations
    # to minimize the number of tensor ops
    let trig_base = 2.0 * PI * k
    let k_square_inv = 1.0 /. (k *. k)
    var b = zeros[float](k.shape)
    for idx in countup(0, bands.len - 1, 2):
      let weight_pow = abs(weights[(idx+1) div 2])
      if requires_matrix_inversion:
        q +=
          weight_pow * 0.5 * bands[idx+1] * (sinc(sk_pos * bands[idx+1]) - sinc(sk_neg * bands[idx+1])) -
          weight_pow * 0.5 * bands[idx] * (sinc(sk_pos * bands[idx]) - sinc(sk_neg * bands[idx]))
      let slope = (desired[idx+1] - desired[idx]) / (bands[idx+1] - bands[idx])
      let b1 = desired[idx] - slope * bands[idx]
      b += weight_pow *
        (slope / (4.0 * PI^2) * (sin(trig_base * bands[idx+1]) - sin(trig_base * bands[idx])) *. k_square_inv)
      b +=
        (weight_pow * (slope * bands[idx] + b1) * cos(trig_base * bands[idx]) -
        weight_pow * (slope * bands[idx+1] + b1) * cos(trig_base * bands[idx+1])) /. trig_base

    let a = if requires_matrix_inversion:
        solve(q, b)
      else:
        -4.0 * weights[0] * b

    result = if even_order:
        # Type III FIR filter: anti-symmetric, with a zero middle value
        0.5 * concat([a[_|-1], [0.0].toTensor, -a], axis = 0)
      else:
        # Type IV FIR filter: anti-symmetric, no middle value
        0.5 * concat([a[_|-1], -a], axis = 0)
