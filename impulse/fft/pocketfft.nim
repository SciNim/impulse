when defined(cpp):
  ## When building on the C++ backend, uses the header only C++ library, which
  ## has some added features.
  import cpp_pocketfft/pocketfft
  export pocketfft
else:
  ## On the C backend uses the regular C PocketFFT implemention. Comes with
  ## arraymancer support.
  import c_pocketfft/pocketfft
  export pocketfft
  import c_pocketfft/pocketfft_arraymancer
  export pocketfft_arraymancer
