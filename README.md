# Impulse

Impulse will be a collection of signal processing primitives for Nim.

## FFT

The FFT part of this library wraps
[PocketFFT](https://gitlab.mpcdf.mpg.de/mtr/pocketfft).

For the C++ backend an optimized version of PocketFFT is used, in the
form of a header-only version,

https://github.com/mreineck/pocketfft

For the C backend we use the PocketFFT directly (the C library linked
before).

Note that the API differs slightly between the two. See the two
examples below.

### C example

The `pocketfft` submodule can to be imported manually using
`import impulse/fft/pocketfft` or one can simply import the `fft`
submodule as shown below.

```nim
import impulse/fft

template isClose(a1, a2, eps: untyped): untyped =
  for i in 0 ..< a1.len:
    doAssert abs(a1[i] - a2[i]) < eps, "Is: " & $a1[i] & " | " & $a2[i]

block Array:
  let dIn = [1.0, 2.0, 1.0, -1.0, 1.5]
  let dOut = fft(dIn, forward = true) # forward or backwards?
  echo dIn
  echo dOut
  isClose(dIn, fft(dOut, forward = false), eps = 1e-10)
  # [1.0, 2.0, 1.0, -1.0, 1.5]
  # @[4.5, 2.081559480312316, -1.651098762732523, -1.831559480312316, 1.608220406444071]
block Seq:
  let dIn = @[1.0, 2.0, 1.0, -1.0, 1.5]
  let dOut = fft(dIn, forward = true) # forward or backwards?
  echo dIn
  echo dOut
  isClose(dIn, fft(dOut, forward = false), eps = 1e-10)
  # @[1.0, 2.0, 1.0, -1.0, 1.5]
  # @[4.5, 2.081559480312316, -1.651098762732523, -1.831559480312316, 1.608220406444071]
block Tensor:
  let dIn = @[1.0, 2.0, 1.0, -1.0, 1.5].toTensor
  let dOut = fft(dIn, forward = true) # forward or backwards?
  echo dIn
  echo dOut
  isClose(dIn, fft(dOut, forward = false), eps = 1e-10)
  # Tensor[system.float] of shape "[5]" on backend "Cpu"
  #     1      2      1     -1    1.5
  # Tensor[system.float] of shape "[5]" on backend "Cpu"
  #     4.5     2.08156     -1.6511    -1.83156     1.60822
import std / complex
block Complex:
  let dIn = [complex(1.0), complex(2.0), complex(1.0), complex(-1.0), complex(1.5)]
  let dOut = fft(dIn, forward = true) # forward or backwards?
  echo dIn
  echo dOut
  isClose(dIn, fft(dOut, forward = false), eps = 1e-10)
  # [(1.0, 0.0), (2.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (1.5, 0.0)]
  # @[(4.5, 0.0), (2.081559480312316, -1.651098762732523), (-1.831559480312316, 1.608220406444071), (-1.831559480312316, -1.608220406444071), (2.081559480312316, 1.651098762732523)]
```

### C++ example

When compiling on the C++ backend, the API is a bit different:

```nim
import impulse/fft
import std / complex

let dIn = @[1.0, 2.0, 1.0, -1.0, 1.5]
var dOut = newSeq[Complex[float64]](dIn.len)

let dInDesc = DataDesc[float64].init(
  dIn[0].unsafeAddr, [dIn.len]
)
var dOutDesc = DataDesc[Complex[float64]].init(
  dOut[0].addr, [dOut.len]
)

let fft = FFTDesc[float64].init(
  axes = [0],
  forward = true
)

fft.apply(dOutDesc, dInDesc)
echo dIn
echo dOut
# @[1.0, 2.0, 1.0, -1.0, 1.5]
# @[(4.5, 0.0), (2.081559480312316, -1.651098762732523), (-1.831559480312316, 1.608220406444071), (0.0, 0.0), (0.0, 0.0)]
```

## LFSR

LFSR module which implements a Linear Feedback Shift Register that can be
used to generate pseudo-random boolean sequences. It supports both Fibonacci
and Galois LFSRs.

LFSRs used in many digital communication systems (including, for example LTE
and 5GNR). For more information see:

https://simple.wikipedia.org/wiki/Linear-feedback_shift_register

Note that this module uses Arraymancer under the hood, so it depends on it.
Also note that this code is heavily based on Nikesh Bajaj's pylfsr, which can
be found in https://pylfsr.github.io

### LFSR examples

The following example creates a Fibonacci-style LFSR with polynomial
`x^5 + x^3 + 1` and uses it to generate a pseudo-random sequence of 31 values.
Note how the `taps` argument is an integer tensor with values `[5, 3]`,
corresponding to the exponents of the coefficients of the polynomial, in
descending order. Exponent 0 was skipped because it is implicitly included
(if it is included it will be ignored). If the exponents are not in
descending order a ValueError exception will be raised.

```nim
import impulse/lfsr
import arraymancer

var fibonacci_lfsr = initLFSR(
  taps = [5, 3], # descending order and 0 can be omitted
  # The following 2 lines can be skipped in this case since they are the defaults
  # state = single_true,
  # conf = fibonacci
)

let sequence1 = fibonacci_lfsr.generate(31)

# Print the first few elements
# Note that sequence1 will be a Tensor[bool] but it can be easily converted to
# Tensor[int] for more concise printing
echo sequence1.asType(int)[_..11]
# Tensor[system.int] of shape "[12]" on backend "Cpu"
#     1     0     0     0     0     1     0     0     1     0     1     1

# The generator can be reset to start over
fibonacci_lfsr.reset()
let sequence2 = fibonacci_lfsr.generate(31)
doAssert sequence1 == sequence2
```

Galois style LFSRs are also supported and it is also possible to set a custom
start state as a Tensor[bool]:

```nim
var galois_lfsr = initLFSR(
  # note how the 0 exponent can be included and taps can be a Tensor as well
  taps = [5, 3, 0].toTensor,
  state = [true, true, true, true, true].toTensor, # this is equivalent to `all_true`
  conf = galois
)

# Generate the first 8 values
let sequence3a = galois_lfsr.generate(8)
echo sequence3a.asType(int)
# Tensor[system.int] of shape "[8]" on backend "Cpu"
#     1     1     1     1     0     0     0     1

# Generate a few more values
let sequence3b = galois_lfsr.generate(10)
echo sequence3b.asType(int)
# Tensor[system.int] of shape "[10]" on backend "Cpu"
#     1     0     1     1     1     0     1     0     1     0

galois_lfsr.reset()
echo galois_lfsr.generate(18)
# Tensor[system.int] of shape "[18]" on backend "Cpu"
#     1     1     1     1     0     0     0     1     1     0     1     1     1     0     1     0     1     0
```

### Maximal LFSR tap examples

As a convenience, a `tap_examples` function is provided. This function takes a
`size` and returns one example (out of many) sequence of taps that generates a
"maximal" LSFR sequence.

### LFSR efficiency

The LFSR module implementation is favors simplicity over speed. As of 2024, it
is able to generate 2^24 values in less than 1 minute on a mid-range laptop.

## Signal

The Signal modulei mplements several signal processing and related functions, such as a Kaiser window and a firls FIR filter design function. See the documentation of those functions for more details.

## License

Licensed and distributed under either of

* MIT license: [LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT

or

* Apache License, Version 2.0, ([LICENSE-APACHEv2](LICENSE-APACHEv2) or http://www.apache.org/licenses/LICENSE-2.0)

at your option. These files may not be copied, modified, or distributed except according to those terms.
