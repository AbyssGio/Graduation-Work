import jittor as jt
import numpy as np

def segment_coo(src, index, out, reduce):
    ans = jt.zeros_like(out)
    if reduce == "sum" or reduce == "add":
        for s, i in src, index:
            for a in range(0,len(i)):
                ans[i[a]] += s[a]

    return ans