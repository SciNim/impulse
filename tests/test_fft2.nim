import ../impulse/fft
import unittest

suite "FFT":
  test "Misc tests":
    # Written by @AngelEzquerra
    let expected = complex([6, -2, -2, -2].toTensor.asType(float), [0, 2, 0, -2].toTensor.asType(float))
    check fft(arange(4.0).asType(Complex64)) == expected
    check fft(arange(4.0).asType(float)) == expected
    check fft(arange(4.0).asType(float), forward = true) == expected
    check ifft(fft(arange(4.0))).real == arange(4.0)
    check ifft(fft(arange(4.0))).imag == [0.0, 0.0, 0.0, 0.0].toTensor
    check fft(arange(4.0), normalize=nkForward) == expected /. complex(expected.len.float)
    check fft(arange(4.0), normalize=nkOrtho) == expected /. complex(sqrt(expected.len.float))
    check ifft(arange(4.0)) == fft(arange(4.0), forward=false)
