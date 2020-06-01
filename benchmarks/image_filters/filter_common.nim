# Impulse
# Copyright (c) 2020-Present The SciNim Project
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard library
  std/[random, monotimes, times, strformat, strutils, macros],
  # Helpers
  ../../helpers/error_functions

type
  Color*[T: SomeFloat] = object
    r*, g*, b*: T

  Image*[T: SomeFloat] = object
    width*, height*: int
    buf*: seq[Color[T]]

# Image utilities
# -----------------------------------------------------

proc newImage*(width, height: int, T: typedesc[SomeFloat]): Image[T] =
  result.width = width
  result.height = height
  result.buf.newSeq(width*height)

proc newRandomImage*(width, height: int, T: typedesc[SomeFloat]): Image[T] =
  var rng = initRand(1234)

  result.width = width
  result.height = height
  result.buf.newSeq(width*height)

  for i in 0 ..< width*height:
    result.buf[i].r = T(rng.rand(1.0))
    result.buf[i].g = T(rng.rand(1.0))
    result.buf[i].b = T(rng.rand(1.0))

{.push inline, noInit.}

func `[]`*[T](image: Image[T] or ptr Image[T], h, w: int): Color[T] =
  result = image.buf[h*image.width + w]

func `[]=`*[T](image: var Image[T], h, w: int, c: Color[T]) =
  image.buf[h*image.width + w] = c


func `[]=`*[T](image: ptr Image[T], h, w: int, c: Color[T]) =
  image.buf[h*image.width + w] = c

func `+`*[T](c, d: Color[T]): Color[T] =
  result = Color[T](
    r: c.r + d.r,
    g: c.g + d.g,
    b: c.b + d.b
  )

func `*`*[T](c, d: Color[T]): Color[T] =
  result = Color[T](
    r: c.r * d.r,
    g: c.g * d.g,
    b: c.b * d.b
  )

func `+=`*[T](c: var Color[T], d: Color[T]) =
  c.r += d.r
  c.g += d.g
  c.b += d.b

func `*=`*[T](c: var Color[T], scalar: T) =
  c.r *= scalar
  c.g *= scalar
  c.b *= scalar

func `*`*[T](c: Color[T], scalar: T): Color[T] =
  result = Color[T](
    r: c.r * scalar,
    g: c.g * scalar,
    b: c.b * scalar
  )

func `*`*[T](scalar: T, c: Color[T]): Color[T] =
  result = Color[T](
    r: c.r * scalar,
    g: c.g * scalar,
    b: c.b * scalar
  )

{.pop.} # inline, noInit

func mean_relative_error*[T](image, imageRef: Image[T]): T =
  doAssert image.width == imageRef.width
  doAssert image.height == imageRef.height

  result = 0.T
  for i in 0 ..< image.buf.len:
    result += relative_error(image.buf[i].r, imageRef.buf[i].r)
    result += relative_error(image.buf[i].g, imageRef.buf[i].g)
    result += relative_error(image.buf[i].b, imageRef.buf[i].b)
  result = result / (image.buf.len.T * 3)

# Benchmark utilities
# -----------------------------------------------------

# warmup
proc warmup*() =
  # Warmup - make sure cpu is on max perf
  let start = cpuTime()
  var foo = 123
  for i in 0 ..< 300_000_000:
    foo += i*i mod 456
    foo = foo mod 789

  # Compiler shouldn't optimize away the results as cpuTime rely on sideeffects
  let stop = cpuTime()
  echo &"Warmup: {stop - start:>4.4f} s, result {foo} (displayed to avoid compiler optimizing warmup away)\n"

warmup()

when defined(gcc):
  echo "\nCompiled with GCC"
elif defined(clang):
  echo "\nCompiled with Clang"
elif defined(vcc):
  echo "\nCompiled with MSVC"
elif defined(icc):
  echo "\nCompiled with ICC"
else:
  echo "\nCompiled with an unknown compiler"

echo "Optimization level => no optimization: ", not defined(release), " | release: ", defined(release), " | danger: ", defined(danger)

proc header*() =
  echo "\n\n"
  echo &"""{"Name":<30} {"throughput":>15} (ops/s)       {"average time":>9} (ms/op)"""

proc separator*() =
  echo "-".repeat(110)

proc report(op: string, start, stop: MonoTime, iters: int) =
  let us = inMicroseconds((stop-start) div iters)
  let throughput = 1e6 / float64(us)
  echo &"{op:<30} {throughput:>15.3f} ops/s         {us.float64 * 1e-3:>9} ms/op"

template bench*(op: string, iters: int, body: untyped): untyped =
  let start = getMonotime()
  for _ in 0 ..< iters:
    body
  let stop = getMonotime()

  report(op, start, stop, iters)
