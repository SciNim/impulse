# FFT overview

This implements the Fast Fourier Transform to compute the Discret Fourier Transform,
Discrete Cosine Transform and Discrete Sine Transform for both 1D and multidimensional problems.

Note for real-time audio: at the moment, computation is optimized for throughput not latency
and in particular uses dynamic memory allocation

## Research

- http://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/
- https://infoscience.epfl.ch/record/59946
- http://www.netlib.org/fftpack/
- https://infoscience.epfl.ch/record/59946
- http://www.fftw.org/pldi99.pdf
- http://cbrc3.cbrc.jp/~tominaga/translations/gsl/gsl-1.6/fftalgorithms.pdf
- https://math.mit.edu/~stevenj/18.335/FFTW-Alan-2008.pdf
- https://cnx.org/contents/ulXtQbN7@15/Implementing-FFTs-in-Practice
