import time
from datetime import datetime
import numpy as np
from sklearn.metrics import f1_score


def disp_elapsed(t0):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_seconds = time.time() - t0
    if elapsed_seconds < 60:
        print(f"{now}  {elapsed_seconds:.1f} seconds")
    elif elapsed_seconds < 3600:
        print(f"{now}  {elapsed_seconds / 60:.1f} minutes")
    else:
        print(f"{now}  {elapsed_seconds / (60 * 60):.1f} hours")


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
