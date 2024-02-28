import pocketfft
import arraymancer / tensor
export tensor

proc fft*[T: float | Complex64](data: var Tensor[T], forward: bool = true) =
  ## Performs an FFT of input `data`. The calculation happens inplace. The tensor
  ## must be of length `length`.
  ## `forward` determines if it's the forward or inverse FFT.
  fft(data.toUnsafeView, data.size.int, forward)

proc fft*[T: float | Complex64](data: Tensor[T], forward: bool = true): Tensor[T] =
  ## Performs an FFT of input `data`. A new tensor is returned. The tensor
  ## must be of length `length`.
  ## `forward` determines if it's the forward or inverse FFT.
  result = data.clone()
  fft(result.toUnsafeView, data.size.int, forward)
