# Package

version       = "0.1.0"
author        = "SciNim"
description   = "Signal processing primitives (FFT, ...)"
license       = "MIT"


# Dependencies

requires "nim >= 1.6.0"
requires "arraymancer >= 0.7.28"

task test, "Run standard tests":
  exec "nim c -r tests/test_fft.nim"
