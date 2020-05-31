# Impulse
# Copyright (c) 2020-Present The SciNim Project
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#                        Roots of unity
#
# ############################################################

import
  ../complex, math

type
  TwiddleGenerator*[T] = object
    ## We use float64 precision for the roots of unity
    v1, v2: seq[Complex[float64]]
    n: int
    mask, shift: int8

func nthRootOfUnity(x, n: int): Complex[float64] =
  ## Compute the nth-root of unity

  # Divide the circle in 8
  let angle = 0.25*PI/n.float64
  var x = x shl 3  # Multiply by 8

  if x < 4*n:      # First half
    if x < 2*n:    # First quadrant
      if x < n:    # First octant
        complex(cos(x.float64 * angle), sin(x.float64 * angle))
      else:
        complex(sin(float64(2*n-x) * angle), cos(float64(2*n-x) * angle))
    else:          # Second quadrant
      x -= 2*n
      if x < n:    # third octant
        complex(-sin(x.float64 * angle), cos(x.float64 * angle))
      else:
        complex(-cos(float64(2*n-x) * angle), sin(float64(2*n-x) * angle))
  else:            # Second half
    x = 8*n-x
    if x < 2*n:    # Third quadrant
      if x < n:    # Fifth octant
        complex(cos(x.float64 * angle), -sin(x.float64 * angle))
      else:
        complex(sin(float64(2*n-x) * angle), -cos(float64(2*n-x) * angle))
    else:          # Fourth quadrant
      x -= 2*n
      if x<n:      # seventh octant
        complex(-sin(x.float64 * angle), -cos(x.float64 * angle))
      else:
        complex(-cos(float64(2*n-x) * angle), -sin(float64(2*n-x) * angle))

func newTwiddleGenerator*[T](n: int): TwiddleGenerator[T] =
  result.n = n

  let nvals = (n+2) shr 1
  result.shift = 1
  while true:
    let shifted = 1 shl result.shift
    let shifted2 = shifted * shifted
    if shifted2 < nvals:
      inc result.shift
    else:
      break

  let shifted = 1'i8 shl result.shift
  result.mask = shifted - 1

  result.v1.setLen(shifted)
  result.v1[0] = complex(1.0, 0.0)
  for i in 1 ..< result.v1.len:
    result.v1[i] = nthRootOfUnity(i, n)

  result.v2.setLen((nvals+result.mask) div shifted)
  result.v2[0] = complex(1.0, 0.0)
  for i in 1 ..< result.v2.len:
    result.v2[i] = nthRootOfUnity(i*shifted, n)

func `[]`*[T](gen: TwiddleGenerator[T], idx: int): Complex[T] =
  var x1 {.noInit.}, x2 {.noInit.}: Complex[float64]
  if 2*idx <= gen.n:
    let x1 = gen.v1[idx and gen.mask]
    let x2 = gen.v2[idx shr gen.shift]
    result = complex(
      re =  T(x1.re*x2.re - x1.im*x2.im),
      im =  T(x1.re*x2.im + x1.im*x2.re)
    )
  else:
    let idx = gen.n-idx
    let x1 = gen.v1[idx and gen.mask]
    let x2 = gen.v2[idx shr gen.shift]
    result = complex(
      re =  T(x1.re*x2.re - x1.im*x2.im),
      im = -T(x1.re*x2.im + x1.im*x2.re)
    )

# Sanity checks
# -------------------------------------------------------------------------------

when isMainModule:
  import strformat

  # Visual check
  for n in 3 ..< 10:
    for x in 1 ..< 10:
      let root = nthRootOfUnity(x, n)
      echo &"nthRoot({x}, n={n}) = [{root.re:5.4f}, {root.im:5.4f}]"
      doAssert abs(1.0 - (root.re*root.re + root.im*root.im)) < 1e-12
