# Impulse
# Copyright (c) 2020-Present The SciNim Project
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  # 3rd party
  weave,
  ./filter_common

# Benchmark multiple implementations of separable 2D filters
#
# i.e. a blur filter
#
# | 1 1 1 |
# | 1 1 1 |
# | 1 1 1 |
#
#                               | 1 |
# can be applied as | 1 1 1 | * | 1 |
#                               | 1 |
# saving a significant amount of compute
#
# similarly the following binomial filter (a.k.a the 3-tap kernel)
#
# | 1 2 1 |
# | 2 4 2 |
# | 1 2 1 |
#
#                               | 1 |
# can be applied as | 1 2 1 | * | 2 |
#                               | 1 |
#
# Reference paper: https://hal.inria.fr/hal-01094906/document

proc filter2D_reference[T](imOut: var Image[T], imIn: Image[T]) =
  # Naive hardcoded application of
  # | 1 2 1 |
  # | 2 4 2 |
  # | 1 2 1 |
  assert imOut.height == imIn.height
  assert imOut.width == imIn.width

  let normalization = 1.T / 16.T

  # TODO: there is an extra pointer indirection here
  let pImgIn = imIn.unsafeAddr
  let pImgOut = imOut.addr

  parallelFor i in 1 ..< imOut.height-1:
    captures: {pImgIn, pImgOut, normalization}
    parallelFor j in 1 ..< pImgOut.width-1: # Note: the paper is skipping every 3 columns
      captures: {i, pImgIn, pImgOut, normalization}
      let a0 = pImgIn[i-1, j-1]; let b0 = pImgIn[i-1, j  ]; let c0 = pImgIn[i-1, j+1]
      let a1 = pImgIn[i  , j-1]; let b1 = pImgIn[i  , j  ]; let c1 = pImgIn[i  , j+1]
      let a2 = pImgIn[i+1, j-1]; let b2 = pImgIn[i+1, j  ]; let c2 = pImgIn[i+1, j+1]

      let s = 1.T * a0 + 2.T * b0 + 1.T * c0 +
              2.T * a1 + 4.T * b1 + 2.T * c1 +
              1.T * a2 + 2.T * b2 + 1.T * c2

      pImgOut[i, j] = s * normalization

  syncRoot(Weave)

  # Perf note:
  # - The RGB layout makes it hard to use more than 3 out of 4 vector lanes (for SSE/Neon) or 3 out of 8 vector lanes (for AVX)
  # - `s` computation is serialized unless using fast-math because floating-point addition is not asociative
  # - Along `i` data is loaded multiple times, if width is large it's flushed and reloaded from cache and is inefficient

proc filter2D_reg_rotation[T](imOut: var Image[T], imIn: Image[T]) =
  # Naive hardcoded application of
  # | 1 2 1 |
  # | 2 4 2 |
  # | 1 2 1 |
  assert imOut.height == imIn.height
  assert imOut.width == imIn.width

  let normalization = 1.T / 16.T

  # TODO: there is an extra pointer indirection here
  let pImgIn = imIn.unsafeAddr
  let pImgOut = imOut.addr

  parallelFor i in 1 ..< imOut.height-1:
    captures: {pImgIn, pImgOut, normalization}
    let j = 1
    var a0 = pImgIn[i-1, j-1]; var b0 = pImgIn[i-1, j]
    var a1 = pImgIn[i  , j-1]; var b1 = pImgIn[i  , j]
    var a2 = pImgIn[i+1, j-1]; var b2 = pImgIn[i+1, j]

    for j in 1 ..< pImgOut.width-1: # Note: the paper is skipping every 3 columns
      loadBalance(Weave)

      let c0 = pImgIn[i-1, j+1]
      let c1 = pImgIn[i  , j+1]
      let c2 = pImgIn[i+1, j+1]

      let s = 1.T * a0 + 2.T * b0 + 1.T * c0 +
              2.T * a1 + 4.T * b1 + 2.T * c1 +
              1.T * a2 + 2.T * b2 + 1.T * c2
      pImgOut[i, j] = s * normalization

      a0 = b0; b0 = c0 # Rotation
      a1 = b1; b1 = c1 # Rotation
      a2 = b2; b2 = c2 # Rotation

  syncRoot(Weave)

proc filter2D_rotation_separable[T](imOut: var Image[T], imIn: Image[T], kernel: static array[3, T]) =
  #                               | 1 |
  # can be applied as | 1 2 1 | * | 2 |
  #                               | 1 |

  var normalization = 0.0
  for val0 in kernel:
    for val1 in kernel:
      normalization += val0 * val1
  normalization = 1.0.T / normalization

  # TODO: there is an extra pointer indirection here
  let pImgIn = imIn.unsafeAddr
  let pImgOut = imOut.addr

  parallelFor i in 1 ..< imOut.height-1:
    captures: {pImgIn, pImgOut, normalization}
    let j = 1
    var a0 = pImgIn[i-1, j-1]; var b0 = pImgIn[i-1, j]
    var a1 = pImgIn[i  , j-1]; var b1 = pImgIn[i  , j]
    var a2 = pImgIn[i+1, j-1]; var b2 = pImgIn[i+1, j]
    var ra = kernel[0] * a0 + kernel[1] * a1 + kernel[2] * a2
    var rb = kernel[0] * b0 + kernel[1] * b1 + kernel[2] * b2

    for j in 1 ..< pImgOut.width-1:
      loadBalance(Weave)

      let c0 = pImgIn[i-1, j+1]
      let c1 = pImgIn[i  , j+1]
      let c2 = pImgIn[i+1, j+1]
      let rc = kernel[0] * c0 + kernel[1] * c1 + kernel[2] * c2

      let s = kernel[0] * ra + kernel[1] * rb + kernel[2] * rc
      pImgOut[i, j] = s * normalization

      ra = rb; rb = rc # rotation

  syncRoot(Weave)

proc filter2D_tiled[T](imOut: var Image[T], imIn: Image[T]) =
  # Naive hardcoded application of
  # | 1 2 1 |
  # | 2 4 2 |
  # | 1 2 1 |
  assert imOut.height == imIn.height
  assert imOut.width == imIn.width

  let normalization = 1.T / 16.T

  const TileH = 32
  const TileW = 32

  # TODO: there is an extra pointer indirection here
  let pImgIn = imIn.unsafeAddr
  let pImgOut = imOut.addr

  parallelForStrided ii in 1 ..< imOut.height-1, stride = TileH:
    captures: {pImgIn, pImgOut, normalization}
    parallelForStrided jj in 1 ..< pImgOut.width-1, stride = TileW: # This iterates 96 elements due to RGB
      captures: {ii, pImgIn, pImgOut, normalization}
      for i in ii ..< min(ii+TileH, pImgOut.height-1):
        for j in jj ..< min(jj+TileW, pImgOut.width-1):
          loadBalance(Weave)

          let a0 = pImgIn[i-1, j-1]; let b0 = pImgIn[i-1, j  ]; let c0 = pImgIn[i-1, j+1]
          let a1 = pImgIn[i  , j-1]; let b1 = pImgIn[i  , j  ]; let c1 = pImgIn[i  , j+1]
          let a2 = pImgIn[i+1, j-1]; let b2 = pImgIn[i+1, j  ]; let c2 = pImgIn[i+1, j+1]

          let s = 1.T * a0 + 2.T * b0 + 1.T * c0 +
                  2.T * a1 + 4.T * b1 + 2.T * c1 +
                  1.T * a2 + 2.T * b2 + 1.T * c2

          pImgOut[i, j] = s * normalization

  syncRoot(Weave)

proc filter2D_tiled_separable_rot[T](imOut: var Image[T], imIn: Image[T], kernel: static array[3, T]) =
  # Naive hardcoded application of
  # | 1 2 1 |
  # | 2 4 2 |
  # | 1 2 1 |
  assert imOut.height == imIn.height
  assert imOut.width == imIn.width

  let normalization = 1.T / 16.T

  const TileH = 32
  const TileW = 32

  # TODO: there is an extra pointer indirection here
  let pImgIn = imIn.unsafeAddr
  let pImgOut = imOut.addr

  parallelForStrided ii in 1 ..< imOut.height-1, stride = TileH:
    captures: {pImgIn, pImgOut, normalization}
    parallelForStrided jj in 1 ..< pImgOut.width-1, stride = TileW: # This iterates 96 elements due to RGB
      captures: {ii, pImgIn, pImgOut, normalization}
      for i in ii ..< min(ii+TileH, pImgOut.height-1):

        let j = jj
        var a0 = pImgIn[i-1, j-1]; var b0 = pImgIn[i-1, j]
        var a1 = pImgIn[i  , j-1]; var b1 = pImgIn[i  , j]
        var a2 = pImgIn[i+1, j-1]; var b2 = pImgIn[i+1, j]
        var ra = kernel[0] * a0 + kernel[1] * a1 + kernel[2] * a2
        var rb = kernel[0] * b0 + kernel[1] * b1 + kernel[2] * b2

        for j in jj ..< min(jj+TileW, pImgOut.width-1):
          loadBalance(Weave)

          let c0 = pImgIn[i-1, j+1]
          let c1 = pImgIn[i  , j+1]
          let c2 = pImgIn[i+1, j+1]
          let rc = kernel[0] * c0 + kernel[1] * c1 + kernel[2] * c2

          let s = kernel[0] * ra + kernel[1] * rb + kernel[2] * rc
          pImgOut[i, j] = s * normalization

          ra = rb; rb = rc # rotation

  syncRoot(Weave)

proc main() =
  const
    Width = 1280
    Height = 720
    Samples = 30

  let image = newRandomImage(Width, Height, float32)

  echo "Note: we have an extra pointer indirection that we can avoid by not using sequences in Images"
  echo "Very important: ensure you kill nimsuggest before benchmarking, it can reduce Weave performance by 20x"
  header()
  separator()

  init(Weave)

  var outRef = newImage(Width, Height, float32)
  bench("Reference", Samples):
    outRef.filter2D_reference(image)

  var outRot = newImage(Width, Height, float32)
  bench("Register Rotation", Samples):
    outRot.filter2D_reg_rotation(image)

  var outRotSep = newImage(Width, Height, float32)
  bench("Register Rotation + Separable", Samples):
    outRotSep.filter2D_rotation_separable(image, [float32 1.0, 2.0, 1.0])

  var outTiled = newImage(Width, Height, float32)
  bench("Tiled", Samples):
    outTiled.filter2D_tiled(image)

  var outTiledSepRot = newImage(Width, Height, float32)
  bench("Tiled Separable Rot", Samples):
    outTiledSepRot.filter2D_tiled_separable_rot(image, [float32 1.0, 2.0, 1.0])

  exit(Weave)

  doAssert mean_relative_error(outRot, outRef) < 1e-4
  doAssert mean_relative_error(outRotSep, outRef) < 1e-4
  doAssert mean_relative_error(outTiled, outRef) < 1e-4
  doAssert mean_relative_error(outTiledSepRot, outRef) < 1e-4

main()
