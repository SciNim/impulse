# Impulse
# Copyright (c) 2020-Present The SciNim Project
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/[strutils, os, complex],
  ./std_cpp

static: doAssert defined(cpp), "This module requires compilation in C++ mode"

# ############################################################
#
#                   PocketFFT wrapper
#
# ############################################################

const
  pocketFFTPath = currentSourcePath.rsplit(DirSep, 1)[0]


{.pragma: pocket, header: pocketFFTPath / "pocketfft_hdronly.h".}

when compileOption("threads"):
  {.localPassC: "-I" & pocketFFTPath.}
else:
  {.localPassC: "-DPOCKETFFT_NO_MULTITHREADING -I" & pocketFFTPath.}

type
  Shape{.importcpp: "pocketfft::shape_t", pocket.} = CppVector[uint]
  Stride{.importcpp: "pocketfft::stride_t", pocket.} = CppVector[int]

proc c2c[T: SomeFloat](
    shape: Shape,
    strideIn: Stride,
    strideOut: Stride,
    axes: Shape,
    forward: bool,
    dataIn: ptr UncheckedArray[CppComplex[T]],
    dataOut: ptr UncheckedArray[CppComplex[T]],
    fct: T,
    nthreads: uint = 1
  ) {.importcpp: "pocketfft::c2c(@)", pocket.}

proc r2c[T: SomeFloat](
    shape: Shape,
    strideIn: Stride,
    strideOut: Stride,
    axes: Shape,
    forward: bool,
    dataIn: ptr T or ptr UncheckedArray[T],
    dataOut: ptr UncheckedArray[CppComplex[T]],
    fct: T,
    nthreads: uint = 1
  ) {.importcpp: "pocketfft::r2c(@)", pocket.}

proc c2r[T: SomeFloat](
    shape: Shape,
    strideIn: Stride,
    strideOut: Stride,
    axes: Shape,
    forward: bool,
    dataIn: ptr UncheckedArray[CppComplex[T]],
    dataOut: ptr T or ptr UncheckedArray[T],
    fct: T,
    nthreads: uint = 1
  ) {.importcpp: "pocketfft::c2r(@)", pocket.}

proc r2r_fftpack[T: SomeFloat](
    shape: Shape,
    strideIn: Stride,
    strideOut: Stride,
    axes: Shape,
    real2hermitian: bool,
    forward: bool,
    dataIn: ptr T or ptr UncheckedArray[T],
    dataOut: ptr T or ptr UncheckedArray[T],
    fct: T,
    nthreads: uint = 1
  ) {.importcpp: "pocketfft::r2r_fftpack(@)", pocket.}

proc r2r_separable_hartley[T: SomeFloat](
    shape: Shape,
    strideIn: Stride,
    strideOut: Stride,
    axes: Shape,
    forward: bool,
    dataIn: ptr T or ptr UncheckedArray[T],
    dataOut: ptr T or ptr UncheckedArray[T],
    fct: T,
    nthreads: uint = 1
  ) {.importcpp: "pocketfft::r2r_separable_hartley(@)", pocket.}

proc r2r_genuine_hartley[T: SomeFloat](
    shape: Shape,
    strideIn: Stride,
    strideOut: Stride,
    axes: Shape,
    forward: bool,
    dataIn: ptr T or ptr UncheckedArray[T],
    dataOut: ptr T or ptr UncheckedArray[T],
    fct: T,
    nthreads: uint = 1
  ) {.importcpp: "pocketfft::r2r_genuine_hartley(@)", pocket.}

proc dct[T: SomeFloat](
    shape: Shape,
    strideIn: Stride,
    strideOut: Stride,
    axes: Shape,
    dctType: range[1'i32..4'i32],
    dataIn: ptr T or ptr UncheckedArray[T],
    dataOut: ptr T or ptr UncheckedArray[T],
    fct: T,
    ortho: bool,
    nthreads: uint = 1
  ) {.importcpp: "pocketfft::dct(@)", pocket.}

proc dst[T: SomeFloat](
    shape: Shape,
    strideIn: Stride,
    strideOut: Stride,
    axes: Shape,
    dstType: range[1'i32..4'i32],
    dataIn: ptr T or ptr UncheckedArray[T],
    dataOut: ptr T or ptr UncheckedArray[T],
    fct: T,
    ortho: bool,
    nthreads: uint = 1
  ) {.importcpp: "pocketfft::dst(@)", pocket.}

# High-level API
# ------------------------------------------------------

type
  FFTDataDescriptor*[T] = object
    ## Descriptor of the data used in or out of the FFT
    shape: Shape
    stride: Stride
    buf: ptr UncheckedArray[T]

  FFTDescriptor[T] = object
    ## Descriptor of the FFT
    axes: Shape
    scalingFactor: T
    nthreads: uint
    forward: bool

func init*[T](_: type FFTDataDescriptor[T],
              buffer: ptr T or ptr UncheckedArray[T],
              shape, stride: distinct openArray[SomeInteger]
             ): FFTDataDescriptor[T] =
  ## Initialize a description of the input or output data
  ## from the `shape`, `stride` and `buffer` passed in.
  assert shape.len = stride.len
  assert not buffer.isNil

  result.shape = newCppVector[uint](shape.len)
  result.stride = newCppVector[int](stride.len)

  for i in 0 ..< shape.len:
    result.shape[i] = uint(shape[i])
    result.stride[i] = int(stride[i])

  result.buf = cast[ptr UncheckedArray[T]](buffer)

func init*[T](_: type FFTDataDescriptor[T],
              buffer: ptr T or ptr UncheckedArray[T],
              shape: openArray[SomeInteger],
             ): FFTDataDescriptor[T] =
  ## Initialize a description of the input or output data
  ## from the `shape` and `buffer` passed in.
  ## The data is assumed to be stored in
  ## C-contiguous (row-major) format
  assert not buffer.isNil

  result.shape = newCppVector[uint](shape.len)
  result.stride = newCppVector[int](shape.len)

  var accum = 1
  for i in countdown(shape.len-1, 0):
    result.stride[i] = int(accum)
    accum *= shape[i]

  for i in 0 ..< shape.len:
    result.shape[i] = uint(shape[i])

  result.buf = cast[ptr UncheckedArray[T]](buffer)

func init*[T](_: type FFTDescriptor[T],
              axes: varargs[int],
              forward: bool,
              scalingFactor = T(1),
              nthreads = 1
             ): FFTDescriptor[T] =
  ## Initialize a description of the FFT
  ## Set `nthreads` to 0 to use all your available cores
  ## Multithreading is only implemented for multidimensional FFT
  result.axes = newCppVector[uint](axes.len)
  for i in 0 ..< axes.len:
    result.axes[i] = uint axes[i]
  result.scalingFactor = scalingFactor
  result.forward = forward
  result.nthreads = uint nthreads

func apply*[In, Out](
      fft: FFTDescriptor,
      descOut: var FFTDataDescriptor[Out],
      descIn: FFTDataDescriptor[In],
     ) =
  when In is Complex and Out is Complex:
    c2c(
      descIn.shape,
      descIn.stride,
      descOut.stride,
      fft.axes,
      fft.forward,
      cast[ptr UncheckedArray[CppComplex[In.T]]](descIn.buf),
      cast[ptr UncheckedArray[CppComplex[Out.T]]](descOut.buf),
      fft.scalingFactor,
      fft.nthreads
    )
  elif Out is Complex:
    r2c(
      descIn.shape,
      descIn.stride,
      descOut.stride,
      fft.axes,
      fft.forward,
      descIn.buf,
      cast[ptr UncheckedArray[CppComplex[Out.T]]](descOut.buf),
      fft.scalingFactor,
      fft.nthreads
    )
  else:
    {.error: "Not implemented".}

# Sanity checks
# ------------------------------------------------------

when isMainModule:

  let dIn = @[1.0, 2.0, 1.0, -1.0, 1.5]
  var dOut = newSeq[Complex[float64]](dIn.len)

  let dInDesc = FFTDataDescriptor[float64].init(
    dIn[0].unsafeAddr, [dIn.len]
  )
  var dOutDesc = FFTDataDescriptor[Complex[float64]].init(
    dOut[0].addr, [dOut.len]
  )

  let fft = FFTDescriptor[float64].init(
    axes = [0],
    forward = true
  )

  fft.apply(dOutDesc, dInDesc)
  echo dIn
  echo dOut
