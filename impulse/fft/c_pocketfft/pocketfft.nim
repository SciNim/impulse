## This is a wrapper of the C library for PocketFFT
## https://gitlab.mpcdf.mpg.de/mtr/pocketfft/
##
## The file `pocketfft_src.h` is a header only version of the combined
## `pocketfft.h` and `pocketfft.c` files so that we

import strutils, os
import complex, math
const
  pocketFFTPath = currentSourcePath.parentDir

{.pragma: pocket, header: pocketFFTPath / "pocketfft.h".}
# Make sure to compile the C file!
{.compile: pocketFFTPath / "pocketfft.c".}

## These are all the relevant types. Most of them are not actually needed. We could also
## just use opaque types for `rfft_plan` and `cfft_plan` and call it a day.
## Note that we depend on the fact that the `cmplx` Complex data type used in
## PocketFFT is binary compatible with Nim's stdlib `Complex64` type (i.e.
## a flat struct of `(64-bit real part, 64-bit imaginary part)`.
const NFCT = 25
type
  rfftp_fctdata = object
    fct: csize_t
    tw: ptr float
    tws: ptr float

  rfftp_plan_i {.importc: "rfftp_plan", pocket.} = object
    length: csize_t
    nfct: csize_t
    mem: ptr float
    fct: array[NFCT, rfftp_fctdata]

  cfftp_fctdata = object
    fct: csize_t
    tw: ptr Complex64
    tws: ptr Complex64

  cfftp_plan_i {.importc: "cfftp_plan", pocket.} = object
    length: csize_t
    nfct: csize_t
    mem: ptr Complex64
    fct: array[NFCT, cfftp_fctdata]

  cfftp_plan {.importc: "cfftp_plan", pocket.} = ptr cfftp_plan_i

  rfftp_plan {.importc: "rfftp_plan", pocket.} = ptr rfftp_plan_i

  fftblue_plan_i {.importc: "fftblue_plan_i", pocket.} = object
    n: csize_t
    n2: csize_t
    plan: cfftp_plan
    mem: ptr float
    bk: ptr float
    pkf: ptr float

  fftblue_plan {.importc: "fftblue_plan", pocket.} = ptr fftblue_plan_i

  rfft_plan_i = object
    packplan: rfftp_plan
    blueblan: fftblue_plan

  cfft_plan_i = object
    packplan: cfftp_plan
    blueplan: fftblue_plan

  cfft_plan {.importc: "cfft_plan", pocket.} = ptr cfft_plan_i

  rfft_plan {.importc: "rfft_plan", pocket.} = ptr rfft_plan_i

proc make_cfft_plan*(length: csize_t): cfft_plan {.importc: "make_cfft_plan", pocket.}
proc destroy_cfft_plan*(plan: cfft_plan) {.importc: "destroy_cfft_plan", pocket.}
proc cfft_backward*(plan: cfft_plan; c: ptr Complex64; fct: cdouble): cint {.importc: "cfft_backward", pocket.}
proc cfft_forward*(plan: cfft_plan; c: ptr Complex64; fct: cdouble): cint {.importc: "cfft_forward", pocket.}
proc cfft_length*(plan: cfft_plan): csize_t {.importc: "cfft_length", pocket.}

proc make_rfft_plan*(length: csize_t): rfft_plan  {.importc: "make_rfft_plan", pocket, cdecl.}
proc destroy_rfft_plan*(plan: rfft_plan)  {.importc: "destroy_rfft_plan", pocket.}
proc rfft_backward*(plan: rfft_plan; c: ptr cdouble; fct: cdouble): cint {.importc: "rfft_backward", pocket.}
proc rfft_forward*(plan: rfft_plan; c: ptr cdouble; fct: cdouble): cint {.importc: "rfft_forward", pocket.}
proc rfft_length*(plan: rfft_plan): csize_t {.importc: "rfft_length", pocket.}

## Non user facing helper types to wrap the make / call / destroy logic.
type
  FFTPlanReal* = object
    pocket*: rfft_plan
    length*: int

  FFTPlanComplex* = object
    pocket*: cfft_plan
    length*: int

proc `=destroy`*(plan: FFTPlanReal) =
  ## Frees the `rfft_plan`
  destroy_rfft_plan(plan.pocket)

proc `=destroy`(plan: FFTPlanComplex) =
  ## Frees the `cfft_plan`
  destroy_cfft_plan(plan.pocket)

proc init*(_: typedesc[FFTPlanReal], length: int): FFTPlanReal =
  result = FFTPlanReal(pocket: make_rfft_plan(length.csize_t))

proc init*(_: typedesc[FFTPlanComplex], length: int): FFTPlanComplex =
  result = FFTPlanComplex(pocket: make_cfft_plan(length.csize_t))

func isOdd*(i: int): bool = (i and 1) == 1

type MemoryView*[T] = ptr UncheckedArray[T]
func toPtr*[T](ar: openArray[T]): MemoryView[T] = cast[ptr UncheckedArray[T]](ar[0].addr)

proc unpackFFT*(data: MemoryView[float]; outDat: MemoryView[Complex64], inLen: int) =
  ## 'Unpacks' a given FFT result from a call to `rfft_forward/backward` as returned
  ## by PocketFFT.
  ##
  ## That is we construct `Complex64` values from the interleaved `Re, Im` values
  ## and recover the `0` complex parts of the packed values.
  ##
  ## Note that it *does not* recover the hermitian conjugate terms of the FFT
  ## result. If you wish the entire FFT result as if you had called `cfft` (or the
  ## regular user facing `fft` procedure), call `symmetrize` instead / in addition.
  let lenOdd = isOdd inLen
  var k = 0
  var cmpl = complex(data[0]) # first element `Re(y[0]), Im(y[0]) == 0`
  outDat[k] = cmpl
  inc k
  for i in 1 ..< inLen: # from `i = 1` odd elements are `Re(y[i])` and even `Im(y[i])`
    if isOdd i: # set the real part
      cmpl.re = data[i]
    elif i > 0: # set the complex part and add
      cmpl.im = data[i]
      outDat[k] = cmpl
      inc k
  if not lenOdd: # In case input is even length, `data.high` is odd, thus need to add last
    cmpl.im = 0.0
    outDat[k] = cmpl

proc unpackFFT*(data: openArray[float]): seq[Complex64] =
  ## Out of place version of the above so that `symmetrize` can call the inplace
  ## version without reallocating again.
  let outLen = if isOdd data.len: (data.len + 1) div 2 # odd, recover 1 value
               else: (data.len + 2) div 2              # recover 2 values
  result = newSeq[Complex64](outLen)
  unpackFFT(toPtr data, toPtr result, data.len)

proc symmetrize*[T: float | Complex64](data: MemoryView[T], res: MemoryView[Complex64], inLen, outLen: int) =
  ## Recovers the latter half of the FFT terms, i.e. the hermitian conjugate
  ## terms from N/2 to N-1.
  ## If the input is `float` data, first `unpackFFTs` it.
  when T is float: # unpack data
    unpackFFT(data, res, inLen)
  else: # copy all the existing data
    copyMem(res, data, inLen * sizeof(Complex64))
  var k = outLen - 1 # fill result from end
  for i in 1 ..< ceilDiv(outLen, 2):
    res[k] = conjugate res[i]
    dec k

proc symmTargetSize*[T: float | Complex64](lastVal: T, length: int): int =
  ## Returns the correct target size needed in `symmetrize` for a given input data type
  when T is float:
    result = length
  else: # copy all the existing data
    let sub = if lastVal.im == 0.0: 2 ## Corresponds to even input, two zeroes recovered
              else: 1                 ## uneven input, one zero recovered
    result = (length * 2 - sub)

proc symmTargetSize*[T: float | Complex64](data: openArray[T]): int =
  result = symmTargetSize(data[^1], data.len)

proc symmetrize*[T: float | Complex64](data: openArray[T]): seq[Complex64] =
  result = newSeq[Complex64](symmTargetSize(data))
  symmetrize(toPtr data, toPtr result, data.len, result.len)

type
  ## `nkBackward`: Normalization by `1` in FFT and `1/N` in inverse FFT
  ## `nkForward`: Normalization by `1/N` in FFT and `1` in inverse FFT
  ## `nkOrtho`: Normalization by `1/√N` for both directions
  ## `nkCustom`: Custom normalization value given
  NormalizeKind* = enum nkBackward, nkOrtho, nkForward, nkCustom
  Normalize* = object
    kind*: NormalizeKind
    value*: float ## Actual normalization value

proc initNormalize*(kind: NormalizeKind, forward: bool, value: float, length: int): Normalize =
  case kind
  of nkBackward: result = Normalize(kind: nkBackward, value: if forward: 1.0 else: 1.0 / length.float)
  of nkForward:  result = Normalize(kind: nkForward,  value: if forward: 1.0 / length.float else: 1.0)
  of nkOrtho:    result = Normalize(kind: nkOrtho,    value: 1.0 / sqrt(length.float))
  of nkCustom:   result = Normalize(kind: nkCustom,   value: value)

template callFFT(plan, fwd, bck, data, length, forward, norm: untyped): untyped =
  if forward:
    let err = fwd(plan.pocket, data[0].addr, norm.value)
    if err != 0:
      raise newException(Exception, "Forward FFT calculation failed.")
  else:
    let err = bck(plan.pocket, data[0].addr, norm.value)
    if err != 0:
      raise newException(Exception, "Backward FFT calculation failed.")

proc rfft*(data: MemoryView[float], length: int, forward: bool, normalize = nkBackward, normValue = Inf) =
  ## Performs an FFT of purely real input `data`. The calculation happens inplace. The array
  ## must be of length `length`.
  ## `forward` determines if it's the forward or inverse FFT.
  ##
  ## This procedure is not intended as a main user facing proc. It is implementation
  ## specific to how PocketFFT performs the FFT for real inputs. It makes use of two symmetries
  ## in order to store the complex result in the input `data` array of length `length`.
  ##
  ## This means if you use this inplace procedure for performance reasons, be aware that the
  ## `data` array will contain the FFT result `y[k]` as follows.
  ##
  ## If `length` is even:
  ##
  ## `data = [Re(y[0]), Re(y[1]), Im(y[1]), ..., Re(y[N/2])]`
  ##
  ## If `length` is odd:
  ##
  ## `data = [Re(y[0]), Re(y[1]), Im(y[1]), ..., Re(y[N/2]), Im(y[N/2])]`
  ##
  ## Note that the imaginary part of `y[0]` is dropped in both cases and for even length inputs
  ## the imaginary part of the element `y[N/2]` is also dropped. These are both always zero
  ## for these two cases and thus redundant.
  ## All terms from `N/2+1` to `N-1` are the hermitian conjugate of the first `1` to `N/2` terms.
  ##
  ## Feel free to call `unpackFFT` on the `data` array to produce a `seq/array/Tensor` of
  ## the input, which is in `Complex64` and exactly `N/2` in length.
  ##
  ## See the explanation below for why these symmetries hold.
  ##
  ## Those two symmetries are:
  ## 1. For a real input array, the result of the FFT will always be 'symmetric'.
  ##    Given notation as x = [x_0, x_1, ..., x_N] our input and y = [y_0, ..., y_N] our FFT output.
  ##    If all x_i are real, then the only complex contributions come from the exp(-2πi kn/N) term.
  ##    With n running from 0 to N-1,
  ##
  ##    `y[k] = Σ_{i = 0}^{N-1} exp( -2πi kn / N ) x[n]`
  ##
  ##    we can see that the second half are the hermitian conjugate terms, because the `N - j`-th
  ##    term is always h.c. to the `j`-th term.
  ##
  ##    See by, for a fixed integer `n` and `k ∈ {0, 1, ..., N-1}`:
  ##
  ##    `k = N-1: exp(-2πi (N-1) n / N) = exp(-2πi Nn / N + 2πi n / N) = exp(-2πi n) + exp(2πi n / N)`,
  ##              with `exp(-2πi n) = 1`, due to n being an integer
  ##    `k = 1  : exp(-2πi n / N)`
  ##
  ##    so `k = 1` is the h.c. of `N-1`.
  ##
  ##    With that knowledge for the real FFT we can drop all the hermitian conjugate terms.
  ##
  ## 2. Then PocketFFT goes one step further by realizing that
  ##
  ##    `y[0] = Σ_i x[i]`, all exponents are 0 and thus `e^-... = 1`
  ##
  ##    allowing us to drop the imaginary part of the first `y[0]` and secondly for N even elements,
  ##    we can also drop the imaginary part of the 'middle' term (for `N = even` due to the `0`-th term
  ##    not contributing to the exponentials, the effective number of complex terms is odd, thus there
  ##    is one middle term; for `N = odd` no such term exists and thus cannot be dropped). For said middle
  ##    term the imaginary part is also always exactly `0`, because all exponents are exactly multiples
  ##    of `πi`.
  ##
  ##    `k = N/2: exp(-2πi N/2 n / N) = exp(-2πi n/2) = exp(-πi n)`
  ##
  ##    So that way PocketFFT can always return data into an output array of N elements for an input
  ##    of `N` elements.
  ##
  ## If the above is hard to follow, take a pen and paper and write out a discrete fourier transform by hand.
  let norm = initNormalize(normalize, forward, normValue, length)
  let plan = FFTPlanReal.init(length)
  let fwd = rfft_forward
  let bck = rfft_backward
  callFFT(plan, fwd, bck, data, length, forward, norm)

proc fft_impl*[T: float | Complex64](data: MemoryView[T], length: int, forward: bool, normalize = nkBackward, normValue = Inf) =
  ## Performs an FFT of input `data`. The calculation happens inplace. The array
  ## must be of length `length`.
  ## `forward` determines if it's the forward or inverse FFT.
  let norm = initNormalize(normalize, forward, normValue, length)
  when T is float:
    let plan = FFTPlanReal.init(length)
    let fwd = rfft_forward
    let bck = rfft_backward
  else:
    let plan = FFTPlanComplex.init(length)
    let fwd = cfft_forward
    let bck = cfft_backward
  callFFT(plan, fwd, bck, data, length, forward, norm)

proc fft*[T: float | Complex64](data: var openArray[T], forward = true, normalize = nkBackward, normValue = Inf) =
  ## Performs an FFT of input `data`. The calculation happens inplace. This
  ## means for real input data, the result is stored as packed `float` data (see `rfft` above).
  ##
  ## `forward` determines if it's the forward or inverse FFT.
  fft_impl(toPtr data, data.len, forward)

proc rfft_packed*(data: openArray[float], forward = true, normalize = nkBackward, normValue = Inf): seq[float] =
  ## Performs an FFT of input real `data`. The result is returned as a `seq`. The array
  ## must be of length `length`. The returned data is in maximally packed form as `float`
  ## data. See the `rfft` overload above.
  ##
  ## `forward` determines if it's the forward or inverse FFT.
  result = @data
  rfft(toPtr(result), data.len, forward)

proc rfft*(data: openArray[float], forward: bool = true, normalize = nkBackward, normValue = Inf): seq[Complex64] =
  ## Performs an FFT of input real `data`. The result is returned as a `seq[Complex64]`.
  ## The returned data only contains the non redundant N/2 first terms of the resulting
  ## FFT. Call `symmetrize` on the result to compute the (symmetric) hermitian conjugate
  ## terms from `N/2` to `N-1` (`N == data.len`).
  ##
  ## Alternatively, simply call `fft` above, which handles this for you.
  ##
  ## `forward` determines if it's the forward or inverse FFT.
  var res = @data
  fft(res, forward) # `rfft` result as a `seq[float]`
  result = unpackFFT res

proc fft*[T: float | Complex64](data: openArray[T], forward = true, normalize = nkBackward, normValue = Inf): seq[Complex64] =
  ## Performs an FFT of input `data`. The result is returned as a `seq`. The array
  ## must be of length `length`.
  ##
  ## `forward` determines if it's the forward or inverse FFT.
  ##
  ## For the real -> complex transform, this is 2 allocations:
  ## - 1 copy of the input data
  ## - 1 allocation for the output array
  ## For complex -> complex we get away with a single clone of the input.
  when T is float:
    result = symmetrize rfft_packed(data, forward, normalize, normValue)
  else:
    result = @data
    fft(result, forward, normalize, normValue)
