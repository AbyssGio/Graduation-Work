import jittor as jt


def masked_select(mat, mask):
    return mat[mask == 1]
