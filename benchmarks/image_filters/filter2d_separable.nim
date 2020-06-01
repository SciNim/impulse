# Impulse
# Copyright (c) 2020-Present The SciNim Project
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
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

func filter2D_reference[T](imOut: var Image[T], imIn: Image[T]) =
  # Naive hardcoded application of
  # | 1 2 1 |
  # | 2 4 2 |
  # | 1 2 1 |
  assert imOut.height == imIn.height
  assert imOut.width == imIn.width

  let normalization = 1.T / 16.T

  for i in 1 ..< imOut.height-1:
    for j in 1 ..< imOut.width-1: # Note: the paper is skipping every 3 columns
      let a0 = imIn[i-1, j-1]; let b0 = imIn[i-1, j  ]; let c0 = imIn[i-1, j+1]
      let a1 = imIn[i  , j-1]; let b1 = imIn[i  , j  ]; let c1 = imIn[i  , j+1]
      let a2 = imIn[i+1, j-1]; let b2 = imIn[i+1, j  ]; let c2 = imIn[i+1, j+1]

      let s = 1.T * a0 + 2.T * b0 + 1.T * c0 +
              2.T * a1 + 4.T * b1 + 2.T * c1 +
              1.T * a2 + 2.T * b2 + 1.T * c2

      imOut[i, j] = s * normalization

  # Perf note:
  # - The RGB layout makes it hard to use more than 3 out of 4 vector lanes (for SSE/Neon) or 3 out of 8 vector lanes (for AVX)
  # - `s` computation is serialized unless using fast-math because floating-point addition is not asociative
  # - Along `i` data is loaded multiple times, if width is large it's flushed and reloaded from cache and is inefficient

func filter2D_reg_rotation[T](imOut: var Image[T], imIn: Image[T]) =
  # Naive hardcoded application of
  # | 1 2 1 |
  # | 2 4 2 |
  # | 1 2 1 |
  assert imOut.height == imIn.height
  assert imOut.width == imIn.width

  let normalization = 1.T / 16.T

  for i in 1 ..< imOut.height-1:
    let j = 1
    var a0 = imIn[i-1, j-1]; var b0 = imIn[i-1, j]
    var a1 = imIn[i  , j-1]; var b1 = imIn[i  , j]
    var a2 = imIn[i+1, j-1]; var b2 = imIn[i+1, j]

    for j in 1 ..< imOut.width-1: # Note: the paper is skipping every 3 columns
      let c0 = imIn[i-1, j+1]
      let c1 = imIn[i  , j+1]
      let c2 = imIn[i+1, j+1]

      let s = 1.T * a0 + 2.T * b0 + 1.T * c0 +
              2.T * a1 + 4.T * b1 + 2.T * c1 +
              1.T * a2 + 2.T * b2 + 1.T * c2
      imOut[i, j] = s * normalization

      a0 = b0; b0 = c0 # Rotation
      a1 = b1; b1 = c1 # Rotation
      a2 = b2; b2 = c2 # Rotation

func filter2D_rotation_separable[T](imOut: var Image[T], imIn: Image[T], kernel: array[3, T]) =
  #                               | 1 |
  # can be applied as | 1 2 1 | * | 2 |
  #                               | 1 |

  var normalization = 0.0
  for val0 in kernel:
    for val1 in kernel:
      normalization += val0 * val1
  normalization = 1.0.T / normalization


  for i in 1 ..< imOut.height-1:
    let j = 1
    var a0 = imIn[i-1, j-1]; var b0 = imIn[i-1, j]
    var a1 = imIn[i  , j-1]; var b1 = imIn[i  , j]
    var a2 = imIn[i+1, j-1]; var b2 = imIn[i+1, j]
    var ra = kernel[0] * a0 + kernel[1] * a1 + kernel[2] * a2
    var rb = kernel[0] * b0 + kernel[1] * b1 + kernel[2] * b2

    for j in 1 ..< imOut.width-1:
      let c0 = imIn[i-1, j+1]
      let c1 = imIn[i  , j+1]
      let c2 = imIn[i+1, j+1]
      let rc = kernel[0] * c0 + kernel[1] * c1 + kernel[2] * c2

      let s = kernel[0] * ra + kernel[1] * rb + kernel[2] * rc
      imOut[i, j] = s * normalization

      ra = rb; rb = rc # rotation

func filter2D_tiled[T](imOut: var Image[T], imIn: Image[T]) =
  # Naive hardcoded application of
  # | 1 2 1 |
  # | 2 4 2 |
  # | 1 2 1 |
  assert imOut.height == imIn.height
  assert imOut.width == imIn.width

  let normalization = 1.T / 16.T

  for ii in countup(1, imOut.height-2, 32):
    for jj in countup(1, imOut.width-2, 32): # Due to RGB this is actually 96 elements
      for i in ii ..< min(ii+32, imOut.height-1):
        for j in jj ..< min(jj+16, imOut.width-1):
          let a0 = imIn[i-1, j-1]; let b0 = imIn[i-1, j  ]; let c0 = imIn[i-1, j+1]
          let a1 = imIn[i  , j-1]; let b1 = imIn[i  , j  ]; let c1 = imIn[i  , j+1]
          let a2 = imIn[i+1, j-1]; let b2 = imIn[i+1, j  ]; let c2 = imIn[i+1, j+1]

          let s = 1.T * a0 + 2.T * b0 + 1.T * c0 +
                  2.T * a1 + 4.T * b1 + 2.T * c1 +
                  1.T * a2 + 2.T * b2 + 1.T * c2

          imOut[i, j] = s * normalization

func filter2D_tiled_separable_rot[T](imOut: var Image[T], imIn: Image[T], kernel: array[3, T]) =
  # Naive hardcoded application of
  # | 1 2 1 |
  # | 2 4 2 |
  # | 1 2 1 |
  assert imOut.height == imIn.height
  assert imOut.width == imIn.width

  let normalization = 1.T / 16.T

  for ii in countup(1, imOut.height-2, 32):
    for jj in countup(1, imOut.width-2, 32): # Due to RGB this is actually 96 elements
      for i in ii ..< min(ii+32, imOut.height-1):

        let j = jj
        var a0 = imIn[i-1, j-1]; var b0 = imIn[i-1, j]
        var a1 = imIn[i  , j-1]; var b1 = imIn[i  , j]
        var a2 = imIn[i+1, j-1]; var b2 = imIn[i+1, j]
        var ra = kernel[0] * a0 + kernel[1] * a1 + kernel[2] * a2
        var rb = kernel[0] * b0 + kernel[1] * b1 + kernel[2] * b2

        for j in jj ..< min(jj+16, imOut.width-1):
          let c0 = imIn[i-1, j+1]
          let c1 = imIn[i  , j+1]
          let c2 = imIn[i+1, j+1]
          let rc = kernel[0] * c0 + kernel[1] * c1 + kernel[2] * c2

          let s = kernel[0] * ra + kernel[1] * rb + kernel[2] * rc
          imOut[i, j] = s * normalization

          ra = rb; rb = rc # rotation

proc main() =
  const
    Width = 1280
    Height = 720
    Samples = 1

  let image = newRandomImage(Width, Height, float32)

  separator()

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

  doAssert mean_relative_error(outRot, outRef) < 1e-4
  doAssert mean_relative_error(outRotSep, outRef) < 1e-4
  doAssert mean_relative_error(outTiled, outRef) < 1e-4
  doAssert mean_relative_error(outTiledSepRot, outRef) < 1e-4

main()
