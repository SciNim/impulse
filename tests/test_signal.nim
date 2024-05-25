import ../impulse/signal
import arraymancer


proc test_kaiser(): bool =
  ## Test the `kaiser` window function
  let kw1 = kaiser(4)
  let kw2 = kaiser(4, 5.0)
  let kw3 = kaiser(4, 8.0)
  let expected_kw1 = [0.03671089, 0.7753221 , 0.7753221 , 0.03671089].toTensor
  let expected_kw3 = [0.00233883, 0.65247867, 0.65247867, 0.00233883].toTensor

  result = kw1.mean_absolute_error(expected_kw1) < 1e-8 and
    kw1.mean_absolute_error(kw2) < 1e-8 and
    kw3.mean_absolute_error(expected_kw3) < 1e-8

proc test_firls(): bool =
  ## Test the `firls` function
  block:
    let expected = [0.1264964, 0.278552488, 0.34506779765, 0.278552488, 0.1264964].toTensor
    if expected.mean_absolute_error(
      firls(4,
        [[0.0, 0.3], [0.4, 1.0]].toTensor,
        [[1.0, 1.0], [0.0, 0.0]].toTensor)) > 1e-8:
          return false

    # Same filter as above, but using rank-1, even length tensors as inputs
    if expected.mean_absolute_error(
      firls(4,
        [0.0, 0.3, 0.4, 1.0].toTensor,
        [1.0, 1.0, 0.0, 0.0].toTensor)) > 1e-8:
          return false

    # Same filter as above, but using unnormalized frequencies and a sampling
    # frequency of 10.0 Hz (note how all the frequencies are 5 times greater
    # because the default fs is 2.0):
    if expected.mean_absolute_error(
      firls(4,
        [[0.0, 1.5], [2.0, 5.0]].toTensor,
        [[1.0, 1.0], [0.0, 0.0]].toTensor,
        fs = 10.0)) > 1e-8:
          return false

  block:
    # Same kind of filter, but give more weight to the pass-through constraint
    let expected = [0.110353444, 0.28447325, 0.36086805, 0.28447325, 0.110353444].toTensor
    if expected.mean_absolute_error(
      firls(4,
        [[0.0, 0.3], [0.4, 1.0]].toTensor,
        [[1.0, 1.0], [0.0, 0.0]].toTensor,
        weights = [1.0, 0.5].toTensor)) > 1e-8:
          return false

  block:
    # A more complex, order 5, Type II, low-pass filter
    let expected = [-0.05603328, 0.146107441, 0.43071645, 0.43071645, 0.146107441, -0.05603328].toTensor
    if expected.mean_absolute_error(
      firls(5,
        [[0.0, 0.3], [0.3, 0.6], [0.6, 1.0]].toTensor,
        [[1.0, 1.0], [1.0, 0.2], [0.0, 0.0]].toTensor)) > 1e-8:
          return false

  block:
    # Example of an order 6 Type IV high-pass FIR filter:
    let expected = [-0.13944975, 0.2851858, -0.25859575, 0.0, 0.25859575, -0.2851858, 0.13944975].toTensor
    if expected.mean_absolute_error(firls(6,
      [[0.0, 0.4], [0.6, 1.0]].toTensor, [[0.0, 0.0], [0.9, 1.0]].toTensor,
      symmetric = false)) > 1e-8:
        return false

  return true

# Run the tests
doAssert test_kaiser()
doAssert test_firls()
