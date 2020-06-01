# FFT overview

This implements the Fast Fourier Transform to compute the Discret Fourier Transform,
Discrete Cosine Transform and Discrete Sine Transform for both 1D and multidimensional problems.

Note for real-time audio: at the moment, computation is optimized for throughput not latency
and in particular uses dynamic memory allocation

We currently use PocketFFT.

Commit 49b813232507470a047727712acda105b84c7815
has an initial pure Nim implementation that follows
the PocketFFT algorithms which in turn follow
FFTPACK which is described in "fftalgorithm.pdf" and the non-available papers from Clive Tamperton.

In the future once/if we get a compiler for high-performance computing
this will be reimplemented using that compiler.

## High-level

- http://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/
- https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-973-communication-system-design-spring-2006/lecture-notes/lecture_8.pdf
- https://infoscience.epfl.ch/record/59946
- http://cbrc3.cbrc.jp/~tominaga/translations/gsl/gsl-1.6/fftalgorithms.pdf


## Implementations

- http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.472.5037&rep=rep1&type=pdf
- http://www.netlib.org/fftpack/
- http://www.fftw.org/pldi99.pdf
- http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.472.5037&rep=rep1&type=pdf
- https://math.mit.edu/~stevenj/18.335/FFTW-Alan-2008.pdf
- https://cnx.org/contents/ulXtQbN7@15/Implementing-FFTs-in-Practice
- https://cnx.org/exports/82e6ba6f-b828-42ef-9db1-8de4b448b869@22.1.pdf/fast-fourier-transforms-22.1.pdf
- https://gitlab.mpcdf.mpg.de/mtr/pocketfft

Via the Halide compiler:
- https://github.com/halide/Halide/tree/master/apps/fft

### Non-available PDFs
- https://www.researchgate.net/publication/222446662_Fast_mixed-radix_real_Fourier_transforms
- https://doi.org/10.1137/0912043
