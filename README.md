# Impulse

Impulse will be a collection of signal processing primitives for Nim.

## FFT

Currently this library only consists of an FFT module, which wraps
[PocketFFT](https://gitlab.mpcdf.mpg.de/mtr/pocketfft) in form of a
header-only C++ version, https://github.com/mreineck/pocketfft

To use it, please import the submodule directly. For example:

```nim
import impulse/fft/pocketfft
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
