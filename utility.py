import time
import numpy as np
from sklearn.metrics import f1_score


def disp_elapsed(t0):
    elapsed_seconds = time.time() - t0
    if elapsed_seconds < 60:
        print("Done: {0:.1f} seconds".format(elapsed_seconds))
    elif elapsed_seconds < 3600:
        print("Done: {0:.1f} minutes".format(elapsed_seconds / 60))
    else:
        print("Done: {0:.1f} hours".format(elapsed_seconds / (60 * 60)))


def f1_best(y, pred, thresh_s=None):
    if thresh_s is None:
        thresh_s = np.linspace(0.1, 0.6, 41)
    best_f1 = 0
    best_thresh = 0
    for thresh in thresh_s:
        f1 = f1_score(y, (pred > thresh).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_f1, best_thresh
