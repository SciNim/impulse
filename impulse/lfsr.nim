## LFSR module which implements a Linear Feedback Shift Register that can be
## used to generate pseudo-random boolean sequences. It supports both Fibonacci
## and Galois LFSRs.
##
## LFSRs used in many digital communication systems (including, for example LTE
## and 5GNR). For more information see:
## https://simple.wikipedia.org/wiki/Linear-feedback_shift_register
##
## Notes:
## - This code is heavily based on Nikesh Bajaj's pylfsr, which can be
## found in https://pylfsr.github.io
## - This implementation is not optimized for performance, but for simplicity.
##   It would be relatively trivial to implement a much faster version by
##   operating on the bits directly, rather than using tensors.

import arraymancer
import std / [algorithm, strformat]

type LFSR_TYPE* = enum
  ## LFSR types (see https://simple.wikipedia.org/wiki/Linear-feedback_shift_register)
  fibonacci, galois

type LFSR_INIT_STATE_TYPE* = enum
  ## Common LFSR Initial States
  ## While it is possible to set any LFSR initial `state` by passing a
  ## Tensor[bool] to `initLFSR`, for convenience it is also possible
  ## to pass one of these enum values as the initial `state`:
  ## - `single_true`: all bits set to `false` except the LSB (i.e. the last one)
  ## - `all_true`: all bits set to `true`
  single_true, all_true

type LFSR* = object
  ## LFSR object used to generate fibonacci or galois pseudo random sequences
  ## To use it first create the object using `initLFSR` and then call either
  ## the `next` procedure to get the sequence values one by one or `generate`
  ## to generate multiple values in one go.
  taps*: Tensor[int]
  conf*: LFSR_TYPE
  state*: Tensor[bool]
  init_state: Tensor[bool]
  verbose*: bool
  counter_starts_at_zero*: bool
  outbit*: bool
  count*: int
  seq_bit_index: int
  sequence: Tensor[bool]
  feedbackbit: bool

proc initLFSR*(taps: Tensor[int] | seq[int],
    conf = fibonacci,
    state: Tensor[bool],
    verbose = false,
    counter_starts_at_zero = true): LFSR =
  ## Initialize LFSR with given feedback polynomial and initial state
  ##
  ## Inputs:
  ## - taps: Feedback polynomial taps as a Tensor[int] or seq[int] of exponents
  ##         in descending order. Exponent 0 can be omitted, since it is always
  ##         implicitly added. For example, to use the `x^5 + x^3 + 1`
  ##         polynomial you can set taps to `[5, 3]` or to `[5, 3, 0]` (or to
  ##         `[5, 3].toTensor` or `[5, 3, 0].toTensor`).
  ## - state: Initial state of the LFSR as a Tensor[bool] of size equal to the
  ##          highest exponent in taps (which must be its first element)
  ## - conf: LFSR type as an enum (`fibonacci` or `galois`)
  ## - verbose: Enable it to print additional logs
  ## - counter_starts_at_zero: Start the count from 0 or 1 (defaults to `true`)
  ##
  ## Return:
  ## - Ready to use LFSR object

  # Remove the last value from taps if it is a zero
  var taps = when typeof(taps) is Tensor: taps else: taps.toTensor
  if taps[taps.size - 1] == 0:
    taps = taps[_..^2]
  if taps.size > 1 and not taps.toSeq1D.isSorted(order = SortOrder.Descending):
    raise newException(ValueError,
      &"The LFSR polynomial must be ordered in descending exponent order, but it is not:\n{taps=}")
  if state.size != taps.max():
    raise newException(ValueError,
      &"The LFSR state size is {state.size} but must be {taps.max()} because that is the highest taps exponent ({taps=})")
  result = LFSR(
    taps: taps,
    conf: conf,
    state: state,
    init_state: state,
    verbose: verbose,
    counter_starts_at_zero: counter_starts_at_zero,
    outbit: false,
    count: 0,
    seq_bit_index: state.size - 1,
    sequence: newTensor[bool](0),
    feedbackbit: false
  )

proc initLFSR*(taps: Tensor[int] | seq[int],
    conf = fibonacci,
    state = single_true,
    verbose = false, counter_starts_at_zero = true): LFSR =
  ## Overload of initLFSR that takes an enum for the initial state
  ##
  ## Inputs:
  ## - taps: Feedback polynomial taps as a Tensor[int] or seq[int] of exponents
  ##         in descending order. Exponent 0 can be omitted, since it is always
  ##         implicitly added. For example, to use the `x^5 + x^3 + 1`
  ##         polynomial you can set taps to `[5, 3]` or to `[5, 3, 0]` (or to
  ##         `[5, 3].toTensor` or `[5, 3, 0].toTensor`).
  ## - state: Initial state of the LFSR as an enum value (`single_true` or
  ##          `all_true`). Defaults to `single_true`.
  ## - verbose: Enable it to print additional logs
  ## - counter_starts_at_zero: Start the count from 0 or 1 (defaults to `true`)
  ##
  ## Return:
  ## - Ready to use LFSR object
  when typeof(taps) is not Tensor:
    let taps = taps.toTensor
  let init_state = if state == all_true:
    arraymancer.ones[bool](taps.max())
  else:
    zeros[bool](taps.max() - 1).append(true)
  initLFSR(taps, conf = conf, state = init_state,
    verbose = verbose, counter_starts_at_zero = counter_starts_at_zero)

proc reset*(self: var LFSR) =
  ## Reset the LFSR `state` and `count` to their initial values
  self.state = self.init_state
  self.count = 0

proc next*(self: var LFSR, verbose = false,
    store_sequence: static bool = false) : bool =
  ## Run one cycle on LFSR with given feedback polynomial and
  ## update the count, state, feedback bit, output bit and sequence
  ##
  ## Inputs:
  ## - Preconfigured LFSR object
  ## - verbose: Print additional logs even if LSRF.verbose is disabled
  ## - store_sequence: static bool that enables saving the generated sequence
  ##
  ## Return:
  ## - bool output bit
  if self.verbose or verbose:
    echo "State: ", self.state

  if self.counter_starts_at_zero:
    result = self.state[self.seq_bit_index]
    when store_sequence:
      self.sequence = self.sequence.append(result)

  if self.conf == fibonacci:
    var b = self.state[self.taps[0] - 1] xor self.state[self.taps[1] - 1]
    if self.taps.size > 2:
      for coeff in self.taps[2.._]:
        b = self.state[coeff - 1] xor b

    self.state = self.state.roll(1)
    self.feedbackbit = b
    self.state[0] = self.feedbackbit
  else: # galois
    self.feedbackbit = self.state[0]
    self.state = self.state.roll(-1)
    for k in self.taps[1.._]:
      self.state[k-1] = self.state[k-1] xor self.feedbackbit

  if not self.counter_starts_at_zero:
    result = self.state[self.seq_bit_index]
    when store_sequence:
      self.sequence = self.sequence.append(result)

  self.count += 1
  self.outbit = result

iterator items*(lfsr: var LFSR,
    n: int = -1,
    store_sequence: static bool = false,
    stop_on_reset = false): bool =
  ## Iterator that will generate a random sequence of length `n`
  ##
  ## Inputs:
  ## - Preconfigured LFSR object
  ## - n: Number of random values to generate (-1 to generate indefinitely,
  ##      which is the default)
  ## - store_sequence: Static bool that enables saving the generated sequence
  ## - stop_on_reset: Stop the iteration if the LFSR has been reset
  ##
  ## Return:
  ## - Generated boolean values
  var generated_count = 0
  while (n < 0 or generated_count < n) and
      not (stop_on_reset and generated_count > 0 and lfsr.count == 0):
    # Generate until the target number of items is reached
    # or the LFSR is reset (i.e. if the count goes back to 0)
    yield lfsr.next(store_sequence = store_sequence)
    generated_count += 1

proc generate*(lfsr: var LFSR,
    n: int,
    store_sequence: static bool = false): Tensor[bool] {.noinit.} =
  ## Generate a random sequence of length `n`
  ##
  ## Inputs:
  ## - Preconfigured LFSR object
  ## - n: Number of random values to generate
  ## - store_sequence: Static bool that enables saving the generated sequence
  ##
  ## Return:
  ## - `Tensor[bool]` of size `n` containing the generated values
  result = newTensor[bool](n)
  for i in 0 ..< n:
    result[i] = lfsr.next(store_sequence = store_sequence)

func lfsr_tap_example*(size: int): seq[int] =
  ## Get an example "maximal" LSFR tap sequence for a given size (up to 24)
  ##
  ## This is a convenience function which can be used to select a set of taps
  ## that generates a "maximal" sequence of the given size.
  ## Note that there are many more tap sequences that generate maximal
  ## sequences and that these examples are have been taken from wikipedia.
  doAssert size >= 2 and size <= 24,
    "LSFR tap examples are only available for sizes between 2 and 24"
  let examples = @[
    @[2, 1],
    @[3, 2],
    @[4, 3],
    @[5, 3],
    @[6, 5],
    @[7, 6],
    @[8, 6, 5, 4],
    @[9, 5],
    @[10, 7],
    @[11, 9],
    @[12, 11, 10, 4],
    @[13, 12, 11, 8],
    @[14, 13, 12, 2],
    @[15, 14],
    @[16, 15, 13, 4],
    @[17, 14],
    @[18, 11],
    @[19, 18, 17, 14],
    @[20, 17],
    @[21, 19],
    @[22, 21],
    @[23, 18],
    @[24, 23, 22, 17]
  ]
  examples[size-2]
