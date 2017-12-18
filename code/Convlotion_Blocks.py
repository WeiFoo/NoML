import numpy as np
import h5py
import matplotlib.pyplot as plt
from im2col import  *

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)


def zero_pad(X, pad):
  """
  Zero padding X with pad number of zeros
  :param X: numpy array of shape(m, n_H, n_W, n_C) of m images
  :param pad: integer, amount of padding
  :return: X_pad
  """
  X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant',
                 constant_values=(0,))
  return X_pad


def conv_single_step(a_slice_prev, W, b):
  """
  Apply one filter by W on a single slice of a, which is the output of previous layer.

  :param a_slice_prev: slice of input, (f, f, n_c_prev)
  :param W: weights, (f, f, n_c_prev)
  :param b: bias, (1,1,1)
  :return Z: a scalar value, which is the result of convolution on a slice
  """
  s = a_slice_prev * W
  Z = np.sum(s, )
  Z = Z + float(b)  ## need to case b to float, so Z is scalar.
  return Z


def conv_forward(A_prev, W, b, hyper_param):
  """
  Implements the forward propagation for a convolution function.

  :param A_prev: output of previous layer, (n_H, n_W, n_prev_C)
  :param W: weights, (f, f, n_prev_C, n_C)
  :param b: bias, (1,1,1,n_C)
  :param hyper_param: dict containing "stride" and "pad"
  :return:
    Z: output, (m, n_H, n_W, n_C)
    cache: cache of values, parameters needed for the conv_backpropagation
  """
  (m, n_prev_H, n_prev_W, n_prev_C) = A_prev.shape
  (f, f, n_prev_C, n_C) = W.shape
  stride = hyper_param["stride"]
  pad = hyper_param["pad"]

  n_H = int((n_prev_H + 2 * pad - f) / stride) + 1  # new n_H after conv
  n_W = int((n_prev_W + 2 * pad - f) / stride) + 1  # new n_W after conv

  Z = np.zeros((m, n_H, n_W, n_C))

  A_prev_pad = zero_pad(A_prev, pad)

  for i in range(m):
    a_prev_pad = A_prev_pad[i, :, :, :]
    for h in range(n_H):
      for w in range(n_W):
        for c in range(n_C):
          y0 = h * stride
          y1 = y0 + f
          x0 = w * stride
          x1 = x0 + f
          a_slice_prev = a_prev_pad[y0:y1, x0:x1, :]
          Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c],
                                           b[:, :, :, c])

  assert (Z.shape == (m, n_H, n_W, n_C))
  cache = (A_prev, W, b, hyper_param)
  return Z, cache


def conv_forward_vect(A_prev, W, b, hyper_param):
  """
  Implements the forward propagation for a convolution function with im2col.

  :param A_prev: output of previous layer, (n_H, n_W, n_prev_C)
  :param W: weights, (f, f, n_prev_C, n_C)
  :param b: bias, (1,1,1,n_C)
  :param hyper_param: dict containing "stride" and "pad"
  :return:
    Z: output, (m, n_H, n_W, n_C)
    cache: cache of values, parameters needed for the conv_backpropagation
  """
  (m, n_prev_H, n_prev_W, n_prev_C) = A_prev.shape
  (f, f, n_prev_C, n_C) = W.shape
  stride = hyper_param["stride"]
  pad = hyper_param["pad"]

  n_H = int((n_prev_H + 2 * pad - f) / stride) + 1  # new n_H after conv
  n_W = int((n_prev_W + 2 * pad - f) / stride) + 1  # new n_W after conv

  X_col = im2col_indices(A_prev, f, f, padding=pad, stride = stride)
  W_col = W.reshape(n_C, -1)
  out = W_col* X_col + b
  out = out.reshape(n_C, n_H, n_W, m)
  Z = out.transpose(3,1,2,0)
  cache = (A_prev, W, b, hyper_param)
  return Z, cache



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
  A_prev = np.random.randn(1000,4,4,3)
  W = np.random.randn(2,2,3,8)
  b = np.random.randn(1,1,1,8)
  hyper_param = {"pad":2,
                 "stride":2}
  Z, cache_conv = conv_forward(A_prev, W, b, hyper_param)
  print("Z's mean = ", np.mean(Z))
  print("Z[3,2,1] = ", Z[3,2,1])
  print("cache_conv[0][1][2][3] = ", cache_conv[0][1][2][3])


def conv_forward_vect_test():
  np.random.seed(1)
  A_prev = np.random.randn(1000,4,4,3)
  W = np.random.randn(2,2,3,8)
  b = np.random.randn(1,1,1,8)
  hyper_param = {"pad":2,
                 "stride":2}
  Z, cache_conv = conv_forward(A_prev, W, b, hyper_param)
  print("Z's mean = ", np.mean(Z))
  print("Z[3,2,1] = ", Z[3,2,1])
  print("cache_conv[0][1][2][3] = ", cache_conv[0][1][2][3])

if __name__ == "__main__":
  # zero_pad_test()
  # conv_single_step_test()
  # conv_forward_test()
  conv_forward_test()
  # conv_forward_vect_test()