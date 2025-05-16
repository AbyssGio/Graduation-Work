import jittor as jt

def Osqueeze(x):
    shape = list(x.shape)
    newshape = [s for s in shape if s > 1]
    return x.reshape(newshape)