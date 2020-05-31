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

func nthRoot(x, n: int): Complex[float64] =
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

# Sanity checks
# -------------------------------------------------------------------------------

when isMainModule:
  import strformat

  # Visual check
  for n in 3 ..< 10:
    for x in 1 ..< 10:
      let root = nthRoot(x, n)
      echo &"nthRoot({x}, n={n}) = [{root.re:5.4f}, {root.im:5.4f}]"
      doAssert abs(1.0 - (root.re*root.re + root.im*root.im)) < 1e-12
