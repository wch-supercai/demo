import numpy as np

N=180011
thr1 = 0.3936 + 0.1829 * (np.log(N) / np.log(2))
print("minimaxi:",thr1)

thr2 = np.sqrt(2.0 * np.log(N))
print("固定阈值:",thr2)