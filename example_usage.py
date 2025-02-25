import numpy as np
from pca_nipals.pca import PcaClass as pca, pcaeval

Num_observation = 30
xvar = 4
Noutput = 2
Num_testing = 10
n_component = Noutput+1             # Number of PLS components (=Number of X Variables)

# Calibration Dataset
X = np.random.rand(Num_observation, xvar)
Beta = np.random.rand(xvar, Noutput) * 2 - 1  # np.array([3,2,1])
Y = (X @ Beta)

# Targeted Output (For which Null space is to be explored)
X_test = np.random.rand(Num_testing, xvar)
Y_test = (X_test @ Beta)

Cx, Sx = X.mean(axis=0), X.std(axis=0, ddof=1)
Cy, Sy = Y.mean(axis=0), Y.std(axis=0, ddof=1)


pca_model = pca().fit(X, n_component=n_component)
eval: pcaeval = pca_model.evaluation(xtest=X_test)
print(eval.xhat)
print(eval.tscore)
print(eval.HT2)
print(eval.spe)
pca_model.visual_plot(X_test=X_test)


pcamiss = pca().fit(X, n_component=xvar)
incom_data = X_test.copy()
n = np.size(incom_data)
portion_of_missed_data = 0.2
nanidx = np.random.choice(
    range(0, n), size=int(np.round(portion_of_missed_data * n)), replace=False)
incom_data.flat[nanidx] = np.nan
compl_data = pcamiss.MissEstimator(incom_data=incom_data)

print("Original Data")
print(X_test)
print("-----------------")

print("incomplete Data")
print(incom_data)
print("-----------------")

print("completed Data")
print(compl_data)
print("-----------------")
