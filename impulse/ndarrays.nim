# Impulse
# Copyright (c) 2020-Present The SciNim Project
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#              Multidimensional Arrays
#
# ############################################################

import macros

type
  NdArray*[N: static int32, T] = object
    ## Descriptor of a multidimensional array
    # int32 to save on size and
    # make all fit in a single cache line
    # do we need over 2 billions elements?
    shape: array[N, int32]
    strides: array[N, int32]
    buf: ptr UncheckedArray[T]

macro staticFor*(idx: untyped{nkIdent}, start, stopExclusive: static int, body: untyped): untyped =
  result = newStmtList()
  for i in start ..< stopEx:
    result.add nnkBlockStmt.newTree(
      ident("unrolledIter_" & $idx & $i),
      body.replaceNodes(idx, newLit i)
    )

func `[]`*[N, T](a: NdArray[N, T], coord: varargs[SomeInteger]): T {.inline.} =
  # TODO: verify codegen of varargs
  assert coord.len == N
  var offset = 0
  staticFor i in 0 ..< N:
    offset += a.strides[i] * coord[i]
  result = a.buf[offset]

func `[]=`*[N, T](a: var NdArray[N, T], coord: varargs[SomeInteger], val: T) {.inline.} =
  # TODO: verify codegen of varargs
  assert coord.len == N
  var offset = 0
  staticFor i in 0 ..< N:
    offset += a.strides[i] * coord[i]
  a.buf[offset] = val
