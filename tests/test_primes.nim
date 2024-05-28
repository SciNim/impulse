import ../impulse/primes
import arraymancer
import std / unittest

proc test_primes() =
  ## Test the `primes` function
  test "Prime number generation":
    check: primes(0).len == 0
    check: primes(1).len == 0
    check: primes(2) == [2].toTensor
    check: primes(3) == [2, 3].toTensor
    check: primes(4) == [2, 3].toTensor
    check: primes(11) == [2, 3, 5, 7, 11].toTensor
    check: primes(12) == [2, 3, 5, 7, 11].toTensor
    check: primes(19) == [2, 3, 5, 7, 11, 13, 17, 19].toTensor
    check: primes(20) == [2, 3, 5, 7, 11, 13, 17, 19].toTensor
    check: primes(22) == [2, 3, 5, 7, 11, 13, 17, 19].toTensor
    check: primes(100000).len == 9592
    check: primes(100003).len == 9593
    check: primes(100000)[^1].item == 99991

# Run the tests
suite "Primes":
  test_primes()
