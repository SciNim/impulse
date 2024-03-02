import pocketfft
import arraymancer / tensor
export tensor

## Unfortunately we need to duplicate some code for Arraymancer here. The issue is that
## there is no working way to define e.g. a concept `ArrayLike[T]` that allows to
## formalize the internal procs like `unpackFFT` or `symmetrize`. Any such attempts
## leads to internal compiler errors.
##
## Something like:
##
## type
##   ArrayLike*[T] = concept x
##     x.len is int
##     x.high is int
##     x[0] is T
##     x[^1] is T
##     toPtr(x) is MemoryView[T]
##
## Used in the main file crashes the compiler with
## Error: internal error: openArrayLoc: ArrayLike[Complex, ArrayLike]

func toPtr*[T](ar: Tensor[T]): MemoryView[T] = cast[ptr UncheckedArray[T]](ar.toUnsafeView)

proc unpackFFT*(data: Tensor[float]): Tensor[Complex64] =
  ## Out of place version of the above so that `symmetrize` can call the inplace
  ## version without reallocating again.
  let outLen = if isOdd data.len: (data.len + 1) div 2 # odd, recover 1 value
               else: (data.len + 2) div 2              # recover 2 values
  result = newTensorUninit[Complex64](outLen)
  unpackFFT(toPtr data, toPtr result, data.len)

proc fft*[T: float | Complex64](data: var Tensor[T], forward: bool = true, normalize = nkAuto, normValue = Inf) =
  ## Performs an FFT of input `data`. The calculation happens inplace. The tensor
  ## must be of length `length`.
  ## `forward` determines if it's the forward or inverse FFT.
  fft_impl(toPtr data, data.size.int, forward, normalize, normValue)

proc rfft_packed*(data: Tensor[float], forward: bool = true, normalize = nkAuto, normValue = Inf): Tensor[float] =
  ## Performs an FFT of input real `data`. The result is returned as a `Tensor`. The array
  ## must be of length `length`. The returned data is in maximally packed form as `float`
  ## data. See the `rfft` overload above.
  ##
  ## `forward` determines if it's the forward or inverse FFT.
  result = data.clone()
  rfft(toPtr(result), data.len, forward, normalize, normValue)

proc rfft*(data: Tensor[float], forward: bool = true, normalize = nkAuto, normValue = Inf): Tensor[Complex64] =
  ## Performs an FFT of input real `data`. The result is returned as a `Tensor[Complex64]`.
  ## The returned data only contains the non redundant N/2 first terms of the resulting
  ## FFT. Call `symmetrize` on the result to compute the (symmetric) hermitian conjugate
  ## terms from `N/2` to `N-1` (`N == data.len`).
  ##
  ## Alternatively, simply call `fft` above, which handles this for you.
  ##
  ## `forward` determines if it's the forward or inverse FFT.
  var res = data.clone()
  fft(res, forward, normalize, normValue) # `rfft` result as a `Tensor[float]`
  result = unpackFFT res

proc symmTargetSize*[T: float | Complex64](data: Tensor[T]): int =
  result = symmTargetSize(data[data.len - 1], data.len)

proc symmetrize*[T: float | Complex64](data: Tensor[T]): Tensor[Complex64] =
  result = newTensorUninit[Complex64](symmTargetSize(data))
  symmetrize(toPtr data, toPtr result, data.len, result.len)

proc fft*[T: float | Complex64](data: Tensor[T], forward = true, normalize = nkAuto, normValue = Inf): Tensor[Complex64] =
  ## Performs an FFT of input `data`. The result is returned as a `Tensor`. The array
  ## must be of length `length`.
  ##
  ## `forward` determines if it's the forward or inverse FFT.
  ##
  ## For the real -> complex transform, this is 2 allocations:
  ## - 1 copy of the input data
  ## - 1 allocation for the output array
  ## For complex -> complex we get away with a single clone of the input.
  when T is float:
    result = symmetrize rfft_packed(data, forward, normalize, normValue)
  else:
    result = data.clone()
    fft(result, forward, normalize, normValue)
