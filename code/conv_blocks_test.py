from conv_blocks import *


def zero_pad_test():
  X = np.random.randn(4, 3, 3, 2)
  X_pad = zero_pad(X, 2)
  print("X shape = ", X.shape)
  print("X_pad shape = ", X_pad.shape)


def conv_single_step_test():
  np.random.seed(1)
  a_slice = np.random.randn(4, 4, 3)
  W = np.random.randn(4, 4, 3)
  b = np.random.randn(1, 1, 1)
  Z = conv_single_step(a_slice, W, b)
  print("Z = ", Z)


def conv_forward_test():
  np.random.seed(1)
  A_prev = np.random.randn(1000, 4, 4, 3)
  W = np.random.randn(2, 2, 3, 8)
  b = np.random.randn(1, 1, 1, 8)
  hyper_param = {"pad": 2,
                 "stride": 2}
  Z, cache_conv = conv_forward(A_prev, W, b, hyper_param)
  print("Z's mean = ", np.mean(Z))
  print("Z[3,2,1] = ", Z[3, 2, 1])
  print("cache_conv[0][1][2][3] = ", cache_conv[0][1][2][3])


def conv_forward_vect_test():
  np.random.seed(1)
  A_prev = np.random.randn(1000, 4, 4, 3)
  W = np.random.randn(2, 2, 3, 8)
  b = np.random.randn(1, 1, 1, 8)
  hyper_param = {"pad": 2,
                 "stride": 2}
  Z, cache_conv = conv_forward(A_prev, W, b, hyper_param)
  print("Z's mean = ", np.mean(Z))
  print("Z[3,2,1] = ", Z[3, 2, 1])
  print("cache_conv[0][1][2][3] = ", cache_conv[0][1][2][3])


def pool_forward_test():
  np.random.seed(1)
  A_prev = np.random.randn(2, 4, 4, 3)
  hyper_param = {"stride": 2, "f": 3}
  A, cache = pool_forward(A_prev, hyper_param, mode="average")
  print (" mode = average")
  print ("A = ", A)

def conv_backward_test():
  np.random.seed(1)
  A_prev = np.random.randn(10, 4, 4, 3)
  W = np.random.randn(2, 2, 3, 8)
  b = np.random.randn(1, 1, 1, 8)
  hyper_param = {"pad": 2,
                 "stride": 2}
  Z, cache_conv = conv_forward(A_prev, W, b, hyper_param)
  dA, dW, db = conv_backward(Z, cache_conv)

  print("dA_mean = ", np.mean(dA))
  print("dW_mean = ", np.mean(dW))
  print("db_mean = ", np.mean(db))



if __name__ == "__main__":
  # zero_pad_test()
  # conv_single_step_test()
  # conv_forward_test()
  # conv_forward_test()
  # conv_forward_vect_test()
  # pool_forward_test()
  conv_backward_test()
