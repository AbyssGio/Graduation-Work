import jittor as jt

def diagonal(Mat):
    ret = []
    for i in range(0, Mat.shape[0]):
        tem = jt.diag(Mat[i])
        ret.append(tem)

    ret = jt.array(ret)
    return ret