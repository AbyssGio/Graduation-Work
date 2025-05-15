import jittor as jt
import numpy as np

def jt_finfo(dtype):
    if isinstance(dtype, jt.Var):
        dtype = dtype.dtype

    type_map = {
        jt.float32: np.float32,
        jt.float16: np.float16,
        jt.float64: np.float64,
    }

    return np.finfo(type_map.get(dtype, dtype))