# Impulse

Impulse will be a collection of signal processing primitives for Nim.

## FFT

Currently this library only consists of an FFT module, which wraps
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


## License

Licensed and distributed under either of

* MIT license: [LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT

or

* Apache License, Version 2.0, ([LICENSE-APACHEv2](LICENSE-APACHEv2) or http://www.apache.org/licenses/LICENSE-2.0)

at your option. These files may not be copied, modified, or distributed except according to those terms.
