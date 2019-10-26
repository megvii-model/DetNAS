import numpy as np
import os
import math
import unittest


class TestCase(unittest.TestCase):
    """docstring for TestCase"""
    def assertTensorClose(self, v0, v1, *, max_err=1e-6, name=None):
        v0 = np.ascontiguousarray(v0, dtype=np.float32)
        v1 = np.ascontiguousarray(v1, dtype=np.float32)

        assert np.isfinite(v0.sum()) and np.isfinite(v1.sum()), (v0, v1)

        self.assertEqual(v0.shape, v1.shape)
        vdiv = np.max([np.abs(v0), np.abs(v1), np.ones_like(v0)], axis=0)
        err = np.abs(v0 - v1) / vdiv
        check = err > max_err
        if check.sum():
            idx = tuple(i[0] for i in np.nonzero(check))
            if name is None:
                name = 'tensor'
            else:
                name = 'tensor {}'.format(name)
            raise AssertionError(
                '{} not equal: '
                'shape={} nonequal_idx={} v0={} v1={} err={}'.format(
                    name, v0.shape,
                    idx, v0[idx], v1[idx], err[idx]))
