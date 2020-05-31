# Impulse
# Copyright (c) 2020-Present The SciNim Project
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#                   C++ standard types wrapper
#
# ############################################################

type
  CppUniquePtr* {.importcpp: "std::unique_ptr", header: "<memory>", byref.} [T] = object
  CppVector* {.importcpp"std::vector", header: "<vector>", byref.} [T] = object
  CppString* {.importcpp: "std::string", header: "<string>", byref.} = object
  CppComplex* {.importcpp: "std::complex", header: "<complex>", bycopy.} [T] = object

proc newCppVector*[T](): CppVector[T] {.importcpp: "std::vector<'*0>()", header: "<vector>", constructor.}
proc newCppVector*[T](size: int): CppVector[T] {.importcpp: "std::vector<'*0>(#)", header: "<vector>", constructor.}
proc len*(v: CppVector): int {.importcpp: "#.size()", header: "<vector>".}
proc add*[T](v: var CppVector[T], elem: T){.importcpp: "#.push_back(#)", header: "<vector>".}
proc `[]`*[T](v: CppVector[T], idx: int): T{.importcpp: "#[#]", header: "<vector>".}
proc `[]`*[T](v: var CppVector[T], idx: int): var T{.importcpp: "#[#]", header: "<vector>".}
proc `[]=`*[T](v: var CppVector[T], idx: int, value: T) {.importcpp: "#[#]=#", header: "<vector>".}
