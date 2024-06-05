import ../impulse/signal
import arraymancer
import std / unittest

proc test_kaiser() =
  ## Test the `kaiser` window function
  test "Kaiser window":
    let kw1 = kaiser(4)
    let kw2 = kaiser(4, 5.0)
    let kw3 = kaiser(4, 8.0)
    let expected_kw1 = [0.03671089, 0.7753221 , 0.7753221 , 0.03671089].toTensor
    let expected_kw3 = [0.00233883, 0.65247867, 0.65247867, 0.00233883].toTensor

    check: kw1.mean_absolute_error(expected_kw1) < 1e-8
    check: kw1.mean_absolute_error(kw2) < 1e-8
    check: kw3.mean_absolute_error(expected_kw3) < 1e-8

proc test_firls() =
  ## Test the `firls` function
  test "firls":
    block:
      let expected = [0.1264964, 0.278552488, 0.34506779765, 0.278552488, 0.1264964].toTensor
      check: expected.mean_absolute_error(
        firls(4,
          [[0.0, 0.3], [0.4, 1.0]].toTensor,
          [[1.0, 1.0], [0.0, 0.0]].toTensor)) < 1e-8

      # Same filter as above, but using rank-1, even length tensors as inputs
      check: expected.mean_absolute_error(
        firls(4,
          [0.0, 0.3, 0.4, 1.0].toTensor,
          [1.0, 1.0, 0.0, 0.0].toTensor)) < 1e-8

      # Same filter as above, but using unnormalized frequencies and a sampling
      # frequency of 10.0 Hz (note how all the frequencies are 5 times greater
      # because the default fs is 2.0):
      check: expected.mean_absolute_error(
        firls(4,
          [[0.0, 1.5], [2.0, 5.0]].toTensor,
          [[1.0, 1.0], [0.0, 0.0]].toTensor,
          fs = 10.0)) < 1e-8

    block:
      # Same kind of filter, but give more weight to the pass-through constraint
      let expected = [0.110353444, 0.28447325, 0.36086805, 0.28447325, 0.110353444].toTensor
      check: expected.mean_absolute_error(
        firls(4,
          [[0.0, 0.3], [0.4, 1.0]].toTensor,
          [[1.0, 1.0], [0.0, 0.0]].toTensor,
          weights = [1.0, 0.5].toTensor)) < 1e-8

    block:
      # A more complex, order 5, Type II, low-pass filter
      let expected = [-0.05603328, 0.146107441, 0.43071645, 0.43071645, 0.146107441, -0.05603328].toTensor
      check: expected.mean_absolute_error(
        firls(5,
          [[0.0, 0.3], [0.3, 0.6], [0.6, 1.0]].toTensor,
          [[1.0, 1.0], [1.0, 0.2], [0.0, 0.0]].toTensor)) < 1e-8

    block:
      # Example of an order 6 Type IV high-pass FIR filter:
      let expected = [-0.13944975, 0.2851858, -0.25859575, 0.0, 0.25859575, -0.2851858, 0.13944975].toTensor
      check: expected.mean_absolute_error(firls(6,
        [[0.0, 0.4], [0.6, 1.0]].toTensor, [[0.0, 0.0], [0.9, 1.0]].toTensor,
        symmetric = false)) < 1e-8

proc test_upfirdn() =
  test "upfirdn":
    # FIR filter
    check: upfirdn([1, 1, 1].toTensor, [1, 1, 1].toTensor) == [1, 2, 3, 2, 1].toTensor

    # Upsampling with zero insertion
    check: upfirdn([1, 2, 3].toTensor, [1].toTensor, 3) == [1, 0, 0, 2, 0, 0, 3].toTensor

    # Upsampling with sample-and-hold
    check: upfirdn([1, 2, 3].toTensor, [1, 1, 1].toTensor, 3) == [1, 1, 1, 2, 2, 2, 3, 3, 3].toTensor

    # Linear interpolation
    check: upfirdn([1.0, 1.0, 1.0].toTensor, [0.5, 1.0, 0.5].toTensor, 2) == [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5].toTensor

    # Decimation by 3
    check: upfirdn(arange(10), [1].toTensor, 1, 3) == [0, 3, 6, 9].toTensor

    # Linear interp, rate 2/3
    check: upfirdn(arange(10.0), [0.5, 1.0, 0.5].toTensor, 2, 3) == [0.0, 1.0, 2.5, 4, 5.5, 7.0, 8.5].toTensor

proc test_resample() =
  test "resample":
    let t = [1.0 , 1.2, -0.3, -1.0, 2.0].toTensor

    block: # Resample with upsampling only
      let resampled = resample(t, up = 3)
      let expected = [1.0006061736,
        1.1652623795, 1.2269863677, 1.2007274083, 1.0003755977,
        0.4963084977, -0.3001818521, -1.1022817902, -1.4444776098,
        -1.0006061736, 0.1147783685, 1.3383072463, 2.0012123471,
        1.7824018717, 0.9175789829].toTensor
      check: resampled.mean_absolute_error(expected) < 1e-8

    block: # Resample with downsampling only
      let resampled = resample(t, down = 2)
      let expected = [0.9812293475, -0.0867375515, 0.5627008880].toTensor
      check: resampled.mean_absolute_error(expected) < 1e-8

    block: # Resample with upsample and downsample
      let resampled = resample(t, up = 3, down = 2)
      let expected = [1.0006061736, 1.2269863677, 1.0003755977, -0.3001818521,
        -1.4444776098, 0.1147783685, 2.0012123471, 0.9175789829].toTensor
      check: resampled.mean_absolute_error(expected) < 1e-8

      # Doubling _both_ sampling rates should give the same result
      let filtered_double_ratios = resample(t, up = 6, down = 4)
      check: resampled == filtered_double_ratios

    block: # Resample with specific filter coefficients
      var (up, down) = (6, 4)
      (up, down) = reduce_resampling_rates(up, down)
      check: (up, down) == (3, 2)

      let h = generate_resampling_filter[float](up, down)
      let resampled = resample(t, h, up = up, down = down)
      let expected = [1.0006061736, 1.2269863677, 1.0003755977, -0.3001818521,
        -1.4444776098, 0.1147783685, 2.0012123471, 0.9175789829].toTensor
      check: resampled.mean_absolute_error(expected) < 1e-8

    block: # Resample with a specific FIR order factor and beta
      let resampled = resample(t, up = 3, down = 2,
        fir_order_factor = 30, beta=6.0)
      let expected = [1.0000890770, 1.1907035514, 1.0289575738, -0.3000267231,
        -1.4576402433, 0.1203579771, 2.0001781539, 0.9282917306].toTensor
      check: resampled.mean_absolute_error(expected) < 1e-8

    block: # Resample with a specific 0 window length (i.e. step upsampling)
      let resampled = resample(t, up = 3, down = 2, fir_order_factor = 0)
      let expected = [1.0, 1.2, 1.2, -0.3, -1, -1, 2, 0].toTensor
      check: resampled.mean_absolute_error(expected) < 1e-8

    block: # Resample a complex signal
      let tc = complex(t, 2.0 *. t)
      let resampled = resample(tc, up = 3, down = 2)
      let expected = complex(
        [1.0006061736, 1.2269863677, 1.0003755977, -0.3001818521,
        -1.4444776098, 0.1147783685, 2.0012123471, 0.9175789829].toTensor,
        [2.0012123471, 2.4539727354, 2.0007511954, -0.6003637041,
        -2.8889552195, 0.2295567369, 4.0024246942, 1.8351579659].toTensor
      )
      check: resampled.mean_absolute_error(expected) < 1e-8

    block:
      # Check that a wide set of combinations of input lengths and up and down
      # ratios work fine (i.e. no exceptions raised and the output lenghts are
      # what is expected)
      let expected_result_lengths = @[
        100, 67, 50, 40, 34, 29, 25, 23, 150, 100, 75, 60, 50, 43, 38, 34,
        200, 134, 100, 80, 67, 58, 50, 45, 250, 167, 125, 100, 84, 72, 63, 56,
        300, 200, 150, 120, 100, 86, 75, 67, 350, 234, 175, 140, 117, 100, 88,
        78, 400, 267, 200, 160, 134, 115, 100, 89, 450, 300, 225, 180, 150,
        129, 113, 100, 101, 68, 51, 41, 34, 29, 26, 23, 152, 101, 76, 61, 51,
        44, 38, 34, 202, 135, 101, 81, 68, 58, 51, 45, 253, 169, 127, 101, 85,
        73, 64, 57, 303, 202, 152, 122, 101, 87, 76, 68, 354, 236, 177, 142,
        118, 101, 89, 79, 404, 270, 202, 162, 135, 116, 101, 90, 455, 303, 228,
        182, 152, 130, 114, 101, 102, 68, 51, 41, 34, 30, 26, 23, 153, 102,
        77, 62, 51, 44, 39, 34, 204, 136, 102, 82, 68, 59, 51, 46, 255, 170,
        128, 102, 85, 73, 64, 57, 306, 204, 153, 123, 102, 88, 77, 68, 357,
        238, 179, 143, 119, 102, 90, 80, 408, 272, 204, 164, 136, 117, 102,
        91, 459, 306, 230, 184, 153, 132, 115, 102, 103, 69, 52, 42, 35, 30,
        26, 23, 155, 103, 78, 62, 52, 45, 39, 35, 206, 138, 103, 83, 69, 59,
        52, 46, 258, 172, 129, 103, 86, 74, 65, 58, 309, 206, 155, 124, 103,
        89, 78, 69, 361, 241, 181, 145, 121, 103, 91, 81, 412, 275, 206, 165,
        138, 118, 103, 92, 464, 309, 232, 186, 155, 133, 116, 103, 104, 70,
        52, 42, 35, 30, 26, 24, 156, 104, 78, 63, 52, 45, 39, 35, 208, 139,
        104, 84, 70, 60, 52, 47, 260, 174, 130, 104, 87, 75, 65, 58, 312, 208,
        156, 125, 104, 90, 78, 70, 364, 243, 182, 146, 122, 104, 91, 81, 416,
        278, 208, 167, 139, 119, 104, 93, 468, 312, 234, 188, 156, 134, 117,
        104, 105, 70, 53, 42, 35, 30, 27, 24, 158, 105, 79, 63, 53, 45, 40,
        35, 210, 140, 105, 84, 70, 60, 53, 47, 263, 175, 132, 105, 88, 75, 66,
        59, 315, 210, 158, 126, 105, 90, 79, 70, 368, 245, 184, 147, 123, 105,
        92, 82, 420, 280, 210, 168, 140, 120, 105, 94, 473, 315, 237, 189, 158,
        135, 119, 105, 106, 71, 53, 43, 36, 31, 27, 24, 159, 106, 80, 64, 53,
        46, 40, 36, 212, 142, 106, 85, 71, 61, 53, 48, 265, 177, 133, 106, 89,
        76, 67, 59, 318, 212, 159, 128, 106, 91, 80, 71, 371, 248, 186, 149,
        124, 106, 93, 83, 424, 283, 212, 170, 142, 122, 106, 95, 477, 318, 239,
        191, 159, 137, 120, 106, 107, 72, 54, 43, 36, 31, 27, 24, 161, 107,
        81, 65, 54, 46, 41, 36, 214, 143, 107, 86, 72, 62, 54, 48, 268, 179,
        134, 107, 90, 77, 67, 60, 321, 214, 161, 129, 107, 92, 81, 72, 375,
        250, 188, 150, 125, 107, 94, 84, 428, 286, 214, 172, 143, 123, 107,
        96, 482, 321, 241, 193, 161, 138, 121, 107, 108, 72, 54, 44, 36, 31,
        27, 24, 162, 108, 81, 65, 54, 47, 41, 36, 216, 144, 108, 87, 72, 62,
        54, 48, 270, 180, 135, 108, 90, 78, 68, 60, 324, 216, 162, 130, 108,
        93, 81, 72, 378, 252, 189, 152, 126, 108, 95, 84, 432, 288, 216, 173,
        144, 124, 108, 96, 486, 324, 243, 195, 162, 139, 122, 108, 109, 73,
        55, 44, 37, 32, 28, 25, 164, 109, 82, 66, 55, 47, 41, 37, 218, 146,
        109, 88, 73, 63, 55, 49, 273, 182, 137, 109, 91, 78, 69, 61, 327, 218,
        164, 131, 109, 94, 82, 73, 382, 255, 191, 153, 128, 109, 96, 85, 436,
        291, 218, 175, 146, 125, 109, 97, 491, 327, 246, 197, 164, 141, 123,
        109, 110, 74, 55, 44, 37, 32, 28, 25, 165, 110, 83, 66, 55, 48, 42,
        37, 220, 147, 110, 88, 74, 63, 55, 49, 275, 184, 138, 110, 92, 79, 69,
        62, 330, 220, 165, 132, 110, 95, 83, 74, 385, 257, 193, 154, 129, 110,
        97, 86, 440, 294, 220, 176, 147, 126, 110, 98, 495, 330, 248, 198, 165,
        142, 124, 110, 111, 74, 56, 45, 37, 32, 28, 25, 167, 111, 84, 67, 56,
        48, 42, 37, 222, 148, 111, 89, 74, 64, 56, 50, 278, 185, 139, 111, 93,
        80, 70, 62, 333, 222, 167, 134, 111, 96, 84, 74, 389, 259, 195, 156,
        130, 111, 98, 87, 444, 296, 222, 178, 148, 127, 111, 99, 500, 333, 250,
        200, 167, 143, 125, 111, 112, 75, 56, 45, 38, 32, 28, 25, 168, 112,
        84, 68, 56, 48, 42, 38, 224, 150, 112, 90, 75, 64, 56, 50, 280, 187,
        140, 112, 94, 80, 70, 63, 336, 224, 168, 135, 112, 96, 84, 75, 392,
        262, 196, 157, 131, 112, 98, 88, 448, 299, 224, 180, 150, 128, 112,
        100, 504, 336, 252, 202, 168, 144, 126, 112, 113, 76, 57, 46, 38, 33,
        29, 26, 170, 113, 85, 68, 57, 49, 43, 38, 226, 151, 113, 91, 76, 65,
        57, 51, 283, 189, 142, 113, 95, 81, 71, 63, 339, 226, 170, 136, 113,
        97, 85, 76, 396, 264, 198, 159, 132, 113, 99, 88, 452, 302, 226, 181,
        151, 130, 113, 101, 509, 339, 255, 204, 170, 146, 128, 113, 114, 76,
        57, 46, 38, 33, 29, 26, 171, 114, 86, 69, 57, 49, 43, 38, 228, 152,
        114, 92, 76, 66, 57, 51, 285, 190, 143, 114, 95, 82, 72, 64, 342, 228,
        171, 137, 114, 98, 86, 76, 399, 266, 200, 160, 133, 114, 100, 89, 456,
        304, 228, 183, 152, 131, 114, 102, 513, 342, 257, 206, 171, 147, 129,
        114, 115, 77, 58, 46, 39, 33, 29, 26, 173, 115, 87, 69, 58, 50, 44,
        39, 230, 154, 115, 92, 77, 66, 58, 52, 288, 192, 144, 115, 96, 83, 72,
        64, 345, 230, 173, 138, 115, 99, 87, 77, 403, 269, 202, 161, 135, 115,
        101, 90, 460, 307, 230, 184, 154, 132, 115, 103, 518, 345, 259, 207,
        173, 148, 130, 115, 116, 78, 58, 47, 39, 34, 29, 26, 174, 116, 87, 70,
        58, 50, 44, 39, 232, 155, 116, 93, 78, 67, 58, 52, 290, 194, 145, 116,
        97, 83, 73, 65, 348, 232, 174, 140, 116, 100, 87, 78, 406, 271, 203,
        163, 136, 116, 102, 91, 464, 310, 232, 186, 155, 133, 116, 104, 522,
        348, 261, 209, 174, 150, 131, 116, 117, 78, 59, 47, 39, 34, 30, 26,
        176, 117, 88, 71, 59, 51, 44, 39, 234, 156, 117, 94, 78, 67, 59, 52,
        293, 195, 147, 117, 98, 84, 74, 65, 351, 234, 176, 141, 117, 101, 88,
        78, 410, 273, 205, 164, 137, 117, 103, 91, 468, 312, 234, 188, 156,
        134, 117, 104, 527, 351, 264, 211, 176, 151, 132, 117, 118, 79, 59,
        48, 40, 34, 30, 27, 177, 118, 89, 71, 59, 51, 45, 40, 236, 158, 118,
        95, 79, 68, 59, 53, 295, 197, 148, 118, 99, 85, 74, 66, 354, 236, 177,
        142, 118, 102, 89, 79, 413, 276, 207, 166, 138, 118, 104, 92, 472, 315,
        236, 189, 158, 135, 118, 105, 531, 354, 266, 213, 177, 152, 133, 118,
        119, 80, 60, 48, 40, 34, 30, 27, 179, 119, 90, 72, 60, 51, 45, 40, 238,
        159, 119, 96, 80, 68, 60, 53, 298, 199, 149, 119, 100, 85, 75, 67, 357,
        238, 179, 143, 119, 102, 90, 80, 417, 278, 209, 167, 139, 119, 105,
        93, 476, 318, 238, 191, 159, 136, 119, 106, 536, 357, 268, 215, 179,
        153, 134, 119]

      let t = arange(-50.0, 100.0)
      var lengths = newSeq[int]()
      for n in arange(100, 120):
        for up in arange(2, 10):
          for down in arange(2, 10):
            let outp = resample(t[_..<n], up = up, down = down)
            lengths.add(outp.len)
      check: lengths == expected_result_lengths

# Run the tests
suite "Signal":
  test_kaiser()
  test_firls()
  test_upfirdn()
  test_resample()
