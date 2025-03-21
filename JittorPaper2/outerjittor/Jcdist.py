import jittor as jt


def cdist(a, b):
    res = jt.zeros([a.shape[0],  b.shape[0]])
    for a_ix, a_item in enumerate(a):
        for b_ix, b_item in enumerate(b):
            res[a_ix][b_ix] = jt.norm((a_item - b_item), p=2)
    print(res)
