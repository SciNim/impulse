import ../impulse/lfsr
import arraymancer / tensor

proc test_fibonacci(): bool =
  var lfsr1 = initLFSR(
    taps = [5, 3].toTensor,
    state = all_true,
    conf = fibonacci
  )
  let sequence1 = lfsr1.generate(31).asType(int)

  var lfsr2 = initLFSR(
    taps = @[5, 3],
    state = all_true,
    conf = fibonacci
  )
  let sequence2 = lfsr2.generate(31).asType(int)
  let expected_sequence = [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0,
                          1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0].toTensor
  return sequence1 == expected_sequence and sequence1 == sequence2

proc test_galois(): bool =
  var lfsr = initLFSR(
    taps = [5, 3].toTensor,
    state = all_true,
    conf = galois
  )

  let sequence = lfsr.generate(31).asType(int)
  let expected_sequence = [1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1,
                          0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1].toTensor
  return sequence == expected_sequence

proc test_init(): bool =
  # The taps must be sorted in descending order
  try:
    var lfsr0 = initLFSR(
      taps = [3, 5, 2].toTensor,
      state = all_true
    )
    echo "[ERROR] initLFSR should have raised an error due to unsorted taps"
    return false
  except ValueError:
    # This is expected
    discard

  # Test different ways to initialize and equivalent LFSR
  var lfsr1 = initLFSR(
    taps = [5, 3].toTensor,
    state = all_true
  )
  var lfsr2 = initLFSR(
    taps = [5, 3].toTensor,
    state = all_true,
    conf = fibonacci
  )
  var lfsr3 = initLFSR(
    taps = [5, 3].toTensor,
    state = ones[bool](5),
    conf = fibonacci
  )
  let s1 = lfsr1.generate(31)
  let s2 = lfsr2.generate(31)
  let s3 = lfsr3.generate(31)
  return s1 == s2 and s2 == s3

proc test_reset(): bool =
  var lfsr = initLFSR(
    taps = [5, 3].toTensor,
    state = all_true
  )
  let s1 = lfsr.generate(31)
  lfsr.reset()
  let s2 = lfsr.generate(31)
  return s1 == s2

doAssert test_fibonacci()
doAssert test_galois()
doAssert test_init()
doAssert test_reset()
