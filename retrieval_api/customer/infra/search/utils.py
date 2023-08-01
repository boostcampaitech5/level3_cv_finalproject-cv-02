import numpy as np

# Good -> 2순위
def L2norm(a: np.ndarray, b=0):
    return np.linalg.norm(a - b)

# Good -> 1순위
def cos_sim(a: np.ndarray, b: np.ndarray):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Good
def pearson_sim(a: np.ndarray, b: np.ndarray):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return np.dot((a - np.mean(a)), (b - np.mean(b))) / ((np.linalg.norm(a - np.mean(a))) * (np.linalg.norm(b - np.mean(b))))

# Bad
def jaccard_sim(a: np.ndarray, b: np.ndarray):
    intersection = np.intersect1d(a, b)
    union = np.union1d(a, b)
    similarity = len(intersection) / len(union)
    return similarity

# Bad
def covariance(a, b):
    return np.cov(a, b, ddof=1)