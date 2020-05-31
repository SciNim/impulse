# Impulse
# Copyright (c) 2020-Present The SciNim Project
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#                     Real-valued FFT
#
# ############################################################

import
  ../dynamic_stack_arrays,
  ./twiddle_factors,
  ./factorization

type
  TwiddleFactor[T] = object
    twiddle: ptr UncheckedArray[T]
    twiddleg: ptr UncheckedArray[T]
    factor: int # Technically we could use int8/uint8 but due to alignment
                # we might as well use the word-size int

  RealFFTPlan*[T] = object
    ## FFT planning along a single dimension
    w: DynamicStackArray[TwiddleFactor[T]]
    mem: seq[T] # Buffer to avoid many TwiddleFactor alloc
    dimSize: int32

func addFactor[T](self: var RealFFTPlan[T], f: int32) {.inline.} =
  self.w.add TwiddleFactor[T](factor: f)

func factorize(self: var RealFFTPlan) =
  ## Divide the space into relatively prime factors
  ## We have special code path for 2, 3, 4, 5, 7, 11
  ## So we include 4
  var n = self.dimSize
  while (n and 3) == 0: # multiple of 4
    self.addFactor 4
    n = n shr 2         # div by 4
  if (n and 1) == 0:    # multiple of 2
    n = n shr 1
    self.addFactor 2
    # Put 2 in front of factor list
    swap(self.w[0], self.w[^1])

  # TODO: evaluate special path for 6 and 9 as well
  #       this would reduce stack allocated memory
  #       6 is detailed in [FFT Algorithm, Brian Gough, 1999]

  # Prepare wheel factorization with {2, 3, 5, 7}
  # The gaps take 48 bytes and reduce the number of checks to 48/2*3*5*7 = 48/210 = 22.9%
  # A {2, 3, 5, 7, 11} wheel would take 480 bytes and reduce the number of checks to 480/2310 = 20.8%
  for prime in n.listFactorize([int32 3, 5, 7]):
    self.addFactor prime

  const wheel = wheel([int32 2, 3, 5, 7])

  for prime in n.wheelFactorize(firstNextPrime = 11, wheel):
    self.addFactor prime

func requiredTwiddleMemSize(self: RealFFTPlan): int =
  result = 0
  var l1 = 1
  for k in 0 ..< self.w.len:
    let ip = self.w[k].factor
    let ido = self.dimSize div (l1 * ip)
    result += (ip-1) * (ido-1)

    # Generic twiddle factors
    if ip > 5:
      result += 2*ip
    l1 *= ip

template offset*[T](p: ptr UncheckedArray[T], offset: int): ptr =
  cast[typeof(p)](cast[ByteAddress](p) +% sizeof(T)*offset)

func computeTwiddleFactors(self: var RealFFTPlan) =
  let twid = newTwiddleGenerator[self.T](self.dimSize)
  var l1 = 1
  let pmem = cast[ptr UncheckedArray[self.T]](self.mem[0].addr)
  var offset = 0

  for k in 0 ..< self.w.len:
    let ip = self.w[k].factor
    let ido = self.dimSize div (l1 * ip)
    if k < self.w.len - 1: # The last factor doesn't need twiddle factors
      self.w[k].twiddle = pmem.offset(offset)
      offset += (ip-1) * (ido-1)
      for j in 1 ..< ip:
        for i in 1 .. ido shr 1:
          let tw = twid[j*l1*i]
          self.w[k].twiddle[(j-1)*(ido-1)+2*i-2] = tw.re
          self.w[k].twiddle[(j-1)*(ido-1)+2*i-1] = tw.im
    if ip > 5: # For the generic kernels
      self.w[k].twiddleg = pmem.offset(offset)
      offset += 2*ip
      self.w[k].twiddleg[0] = 1.0
      self.w[k].twiddleg[1] = 0.0

      var
        i = 2
        ic = 2*ip-2
      while i <= ic:
        let tw = twid[(i shr 1) * self.dimSize div ip]
        self.w[k].twiddleg[i  ] =   tw.re
        self.w[k].twiddleg[i+1] =   tw.im
        self.w[k].twiddleg[ic  ] =  tw.re
        self.w[k].twiddleg[ic+1] = -tw.im

        i += 2
        ic -= 2

    l1 *= ip

func newRealFFTPlan(dimSize: SomeInteger, T: typedesc[SomeFloat]): RealFFTPlan[T] =
  result.dimSize = dimSize.int32
  doAssert dimSize > 0
  if dimSize == 1:
    return
  result.factorize()
  let reqMem = result.requiredTwiddleMemSize()
  result.mem.setLen reqMem
  result.computeTwiddleFactors()

# Sanity checks
# -------------------------------------------------------------------------------

when isMainModule:
  import os, strformat

  let plan = newRealFFTPlan(123, float64)
  for factor in plan.mem:
    stdout.write &"{factor:<7.6f}\n"
