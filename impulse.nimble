# Package

version       = "0.1.1"
author        = "SciNim"
description   = "Signal processing primitives (FFT, filtering, ...)"
license       = "MIT"


# Dependencies

requires "nim >= 1.6.0"
requires "arraymancer >= 0.7.30"

task test, "Run standard tests":
  exec "nim c -r tests/test_fft.nim"
  exec "nim c -r tests/test_fft2.nim"
  exec "nim c -r tests/test_signal.nim"
