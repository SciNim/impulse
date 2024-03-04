import ../impulse/fft/pocketfft
import std / [random, math]

## The test case in this file is based on the tests part of PocketFFT

static: doAssert defined(c), "This test must be run on the C backend!"

const maxlen = 8192

proc fill_random(data: var openArray[float]) =
  for i in 0 ..< data.len:
    data[i] = rand(-0.5 .. 0.5)

proc errcalc(data, odata: openArray[float], length: int): float =
  var
    sum = 0.0
    errsum = 0.0
  for m in 0 ..< length:
    #echo "Data ", data[m], " vs ", odata[m]
    errsum += (data[m]-odata[m])*(data[m]-odata[m])
    sum += odata[m]*odata[m]
  result = sqrt(errsum / sum)

proc errcalc(data, odata: Tensor[float], length: int): float =
  result = errcalc(toOpenArray(data.toUnsafeView(), 0, length),
                   toOpenArray(odata.toUnsafeView(), 0, length),
                   length)

proc test_real(): int =
  var data: array[maxlen, float]
  var odata: array[maxlen, float]
  const epsilon = 2e-15
  var ret = 0
  fill_random(odata)
  odata[0] = 0.340188
  var errsum = 0.0
  for length in 1 ..< maxlen:
    copyMem(data[0].addr, odata[0].addr, length * sizeof(float))
    var plan = make_rfft_plan(length.csize_t)
    discard rfft_forward(plan, data[0].addr, 1.0);
    discard rfft_backward(plan, data[0].addr, 1.0 / length.float);
    destroy_rfft_plan(plan);
    var err = errcalc(data, odata, length)
    if err > epsilon:
      echo "problem at real length ", length, " : ", err
      ret = 1
    errsum += err;
  echo "errsum: ", errsum
  result = ret

proc test_real_hl(): int =
  var data: array[maxlen, float]
  var odata: array[maxlen, float]
  fill_random(odata)
  const epsilon = 2e-15
  var ret = 0
  var errsum = 0.0
  for length in 1 ..< maxlen:
    data = odata
    fft(toOpenArray(data, 0, length), forward = true)
    fft(toOpenArray(data, 0, length), forward = false)
    var err = errcalc(data, odata, length)
    if err > epsilon:
      echo "problem at real length ", length, " : ", err
      ret = 1
    errsum += err;
  echo "errsum: ", errsum
  result = ret

proc test_real_hl_seq(): int =
  const epsilon = 2e-15
  var ret = 0
  var errsum = 0.0
  for length in 1 ..< maxlen:
    var data: seq[float]
    var odata = newSeq[float](length)
    fill_random(odata)
    data = odata

    fft(data, forward = true)
    fft(data, forward = false)

    let err = errcalc(data, odata, length)
    if err > epsilon:
      echo "problem at real length ", length, " : ", err
      ret = 1
    errsum += err;
  echo "errsum: ", errsum
  result = ret

proc test_real_hl_tensor(): int =
  const epsilon = 2e-15
  var ret = 0
  var errsum = 0.0
  for length in 1 ..< maxlen:
    var data: Tensor[float]
    var odata = randomTensor[float](length, -0.5 .. 0.5)
    data = odata.clone()

    fft(data, forward = true)
    fft(data, forward = false)

    let err = errcalc(data, odata, length)
    if err > epsilon:
      echo "problem at real length ", length, " : ", err
      ret = 1
    errsum += err;
  echo "errsum: ", errsum
  result = ret

doAssert test_real() == 0
doAssert test_real_hl() == 0
doAssert test_real_hl_seq() == 0
doAssert test_real_hl_tensor() == 0
