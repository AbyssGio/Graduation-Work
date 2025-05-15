import numpy as np


def cosine_similarity(a, b, dim=1):
    return np.dot(a[dim], b[dim]) / (np.linalg.norm(a[dim]) * np.linalg.norm(b[dim]))