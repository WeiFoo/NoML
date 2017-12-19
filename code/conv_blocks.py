import numpy as np
import h5py
import matplotlib.pyplot as plt
from im2col import *

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

  X_col = im2col_indices(A_prev, f, f, padding=pad, stride=stride)
  W_col = W.reshape(n_C, -1)
  out = W_col * X_col + b
  out = out.reshape(n_C, n_H, n_W, m)
  Z = out.transpose(3, 1, 2, 0)
  cache = (A_prev, W, b, hyper_param)
  return Z, cache


def pool_forward(A_prev, hyper_param, mode="max"):
  """
   forward pass of the pooling layer

  :param A_prev: Input data, (m, n_prev_H, n_prev_W, n_prev_C)
  :param hyper_param: dict of "f" and "stride"
  :return:
    A: output of pooling
    cache: cache used in the back propagation
  """
  (m, n_prev_H, n_prev_W, n_prev_C) = A_prev.shape

  f = hyper_param["f"]
  stride = hyper_param["stride"]

  # dimension of the output
  n_H = int((n_prev_H - f) / stride) + 1  # new n_H after conv
  n_W = int((n_prev_W - f) / stride) + 1  # new n_W after conv
  n_C = n_prev_C

  # output
  A = np.zeros((m, n_H, n_W, n_C))

  for i in range(m):
    for h in range(n_H):
      for w in range(n_W):
        for c in range(n_C):
          y0 = h * stride
          y1 = y0 + f
          x0 = w * stride
          x1 = x0 + f

          a_slice_prev = A_prev[y0:y1, x0:x1, :]
          if mode == "max":
            A[i, h, w, c] = np.max(a_slice_prev)
          if mode == "average":
            A[i, h, w, c] = np.mean(a_slice_prev)

  cache = (A_prev, hyper_param)
  return A, cache
