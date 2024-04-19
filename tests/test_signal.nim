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

doAssert test_kaiser()
