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
  FFTPlanReal = object
    pocket: rfft_plan
    length: int

proc `=destroy`(plan: FFTPlanReal) =
  ## Frees the `rfft_plan`
  destroy_rfft_plan(plan.pocket)

proc init(_: typedesc[FFTPlanReal], length: int): FFTPlanReal =
  result = FFTPlanReal(pocket: make_rfft_plan(length.csize_t))

type
  FFTPlanComplex = object
    pocket: cfft_plan
    length: int

proc `=destroy`(plan: FFTPlanComplex) =
  ## Frees the `rfft_plan`
  destroy_cfft_plan(plan.pocket)

proc init(_: typedesc[FFTPlanComplex], length: int): FFTPlanComplex =
  result = FFTPlanComplex(pocket: make_cfft_plan(length.csize_t))

proc fft*[T: float | Complex64](data: ptr UncheckedArray[T], length: int, forward: bool) =
  ## Performs an FFT of input `data`. The calculation happens inplace. The array
  ## must be of length `length`.
  ## `forward` determines if it's the forward or inverse FFT.
  when T is float:
    let plan = FFTPlanReal.init(length)
    let fwd = rfft_forward
    let bck = rfft_backward
  else:
    let plan = FFTPlanComplex.init(length)
    let fwd = cfft_forward
    let bck = cfft_backward

  if forward:
    let err = fwd(plan.pocket, data[0].addr, 1.0)
    if err != 0:
      raise newException(Exception, "Forward FFT calculation failed.")
  else:
    let err = bck(plan.pocket, data[0].addr, 1.0 / length.float)
    if err != 0:
      raise newException(Exception, "Forward FFT calculation failed.")

proc fft*[T: float | Complex64](data: var openArray[T], forward: bool = true) =
  ## Performs an FFT of input `data`. The calculation happens inplace. The array
  ## must be of length `length`.
  ## `forward` determines if it's the forward or inverse FFT.
  fft(cast[ptr UncheckedArray[T]](data[0].addr), data.len, forward)

proc fft*[T: float | Complex64](data: openArray[T], forward: bool = true): seq[T] =
  ## Performs an FFT of input `data`. The result is returned as a `seq`. The array
  ## must be of length `length`.
  ## `forward` determines if it's the forward or inverse FFT.
  result = @data
  result.fft(forward)
