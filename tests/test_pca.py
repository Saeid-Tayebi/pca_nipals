from pca_nipals.pca import PcaClass as pca, pcaeval
import numpy as np
import pytest
from tests.refrence_model.pca_sklear import pcaSkleanr

Num_observation = 30
xvar = 4
Noutput = 2
Num_testing = 1
n_component = Noutput+1             # Number of PLS components (=Number of X Variables)

# Calibration Dataset
X = np.random.rand(Num_observation, xvar)
Beta = np.random.rand(xvar, Noutput) * 2 - 1  # np.array([3,2,1])
Y = (X @ Beta)

# Targeted Output (For which Null space is to be explored)
X_test = np.random.rand(Num_testing, xvar)
Y_test = (X_test @ Beta)

# Center scalling X and Y
Cx, Sx = X.mean(axis=0), X.std(axis=0, ddof=1)
Cy, Sy = Y.mean(axis=0), Y.std(axis=0, ddof=1)

X, X_test = (X-Cx)/Sx, (X_test-Cx)/Sx
Y, Y_test = (Y-Cy)/Sy, (Y_test-Cy)/Sy


benchmarkpca = pcaSkleanr(X, n_component)
xhatb, Ttesb, T2tesb, SPEtesb, = benchmarkpca.eval(X_test)


mypca = pca().fit(X, n_component=n_component)
eval: pcaeval = mypca.evaluation(X_test)
xhat, Ttes, T2tes, SPEtes = eval.xhat, eval.tscore, eval.HT2, eval.spe


def test_xhat():
    pca2 = pca().fit(X, n_component=xvar)
    xhat: pcaeval = pca2.evaluation(X).xhat
    assert np.allclose(xhat, X, rtol=1e-3)


def test_num_com_setter():
    mypca1 = pca().fit(X[1:5, :], n_component=6)
    assert mypca1.n_component == 3

    mypca2 = pca().fit(X, n_component=xvar+1)
    assert mypca2.n_component == xvar


def test_P():
    assert np.allclose(abs(benchmarkpca.P), abs(mypca.P), atol=.01)


def test_xtes():

    threshold = .001
    assert np.allclose(abs(xhat), abs(xhatb), atol=threshold)
    assert np.allclose(abs(Ttes), abs(Ttesb), atol=threshold)
    assert np.allclose(abs(T2tes), abs(T2tesb), atol=threshold)
    assert np.allclose(abs(SPEtes), abs(SPEtesb), atol=threshold)


def test_missing_data():
    pcamiss = pca().fit(X, n_component=xvar)
    incom_data = X_test.copy()
    n = np.size(incom_data)
    portion_of_missed_data = np.random.uniform(0.1, 0.4)
    nanidx = np.random.choice(
        range(0, n), size=int(np.round(portion_of_missed_data * n)), replace=False)
    incom_data.flat[nanidx] = np.nan
    incom_data[-1, :] = np.nan

    compl_data = pcamiss.MissEstimator(incom_data=incom_data)
    not_a_nan_idx = np.where(~np.isnan(incom_data.reshape(-1)))[0]

    # Ensure no NaN values remain
    assert np.size(np.where(np.isnan(compl_data))[0]) == 0

    # Ensure known values are unchanged
    assert np.allclose(incom_data.flat[not_a_nan_idx], compl_data.flat[not_a_nan_idx], rtol=1e-3)

    # Test if completely empty rows are handled correctly
    assert np.all(compl_data[-1, :] == 0)

    # Test case where there are no missing values
    complete_data = pcamiss.MissEstimator(incom_data=X_test)
    assert np.allclose(X_test, complete_data, rtol=1e-3)


def test_remove_zero_rows():
    """Test if rows with all zeros in X or Y are removed correctly before fitting."""

    # Case 1: X has rows of all zeros
    X_case1 = np.array([[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6], [
                       7, 8, 9, 10, 11, 12], [11, 21, 31, 41, 51, 16]])
    pca_model = pca().fit(X_case1, n_component=2)
    assert pca_model.Xtrain_normal.shape[0] == 3

    # Case 2: No zero rows (control case)
    X_case4 = np.array([[1, 2, 3, 4, 5, 6], [71, 18, 19, 12, 13, 32], [51, 58, 15, 52, 53, 55]])
    pca_model = pca().fit(X_case4, n_component=2)
    assert pca_model.Xtrain_normal.shape[0] == 3

    # Case 3: X col with no var
    X_case1 = np.array([[1, 21, 31, 41],
                        [1, 15, 6, 17],
                        [1, 8, 19, 10]])  # First column has zero variance

    pca_model = pca().fit(X_case1, n_component=2)
    assert pca_model.Xtrain_normal.shape[1] == 3

    # Case 4: no variance at all
    with pytest.raises(ValueError, match="data does not have any variance"):
        X_case3 = np.array([[1, 1, 1],
                            [2, 2, 2],
                            [3, 3, 3]])

        pca_model = pca().fit(X_case3, n_component=2)
