# Impulse
# Copyright (c) 2020-Present The SciNim Project
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#                       Complex
#
# ############################################################

type
  Complex*[T] = object
    ## Complex type which can also be built on top of SIMD
    re*, im*: T

{.push inline, noInit.}

func complex*[T](re, im: T): Complex[T] =
  result.re = re
  result.im = im

func assign*[T](a: var Complex[T], re, im: T) =
  a.re = re
  a.im = im

func assign*[T](a: var Complex[T], re: T) =
  a.re = re
  a.im = default(T)

func `+=`*(a: var Complex, b: Complex) =
  a.re += b.re
  a.im += b.im

func `-=`*(a: var Complex, b: Complex) =
  a.re -= b.re
  a.im -= b.im

func `*=`*[T](a: var Complex[T], scalar: T) =
  a.re *= scalar
  a.im *= scalar

func `*`*(a, b: Complex): Complex =
  result.re = a.re*b.re - a.im*b.im
  result.im = a.re*b.im + a.im*b.re

func `*=`*(a: var Complex, b: Complex) =
  a = a * b

func conj*(a: Complex): Complex =
  result.re = a.re
  result.im = -a.im
