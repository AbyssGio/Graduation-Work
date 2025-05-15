import jittor as jt
import numpy as np

if __name__ == '__main__':
    a = jt.array(np.zeros([16,50176]))
    b = jt.array(np.zeros([16,50176]))
    c = jt.array(np.zeros([16,50176]))
    d = jt.stack([a,b],2)
    print(jt.attrs(d))