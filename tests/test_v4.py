"""v4 组件的合成数据单测：理论预测在受控环境中必须成立。"""
import numpy as np
import pandas as pd


def test_tobit_em_beats_deletion_on_synthetic_censoring():
    """线性模型 + 右删失：Tobit-EM 系数误差 < 删除式 < ... 且方向正确。"""
    from sklearn.linear_model import Ridge
    from scipy.stats import norm
    rng = np.random.default_rng(0)
    n, beta, sigma = 4000, 2.0, 1.0
    x = rng.normal(0, 1, n)
    y_star = beta * x + rng.normal(0, sigma, n)
    L = 1.5
    y_obs = np.minimum(y_star, L)
    censored = y_star > L

    fit = lambda X, Y: Ridge(alpha=1e-6).fit(X.reshape(-1, 1), Y).coef_[0]
    b_mask = fit(x, y_obs)                       # 用删失观测
    b_del = fit(x[~censored], y_obs[~censored])  # 删除删失样本
    # Tobit-EM
    y_em = y_obs.copy()
    for _ in range(5):
        b = fit(x, y_em)
        mu = b * x
        resid_sd = np.std(y_em[~censored] - mu[~censored])
        a = (L - mu[censored]) / resid_sd
        y_em[censored] = mu[censored] + resid_sd * norm.pdf(a) / np.clip(1 - norm.cdf(a), 1e-9, None)
    b_em = fit(x, y_em)

    assert abs(b_em - beta) < abs(b_mask - beta), (b_em, b_mask)
    assert abs(b_em - beta) < abs(b_del - beta), (b_em, b_del)
    assert abs(b_em - beta) < 0.15


def test_tweedie_reduces_mse_on_synthetic_means():
    """z=theta+noise：Tweedie 校正后 MSE(theta) 必须低于原始 z。"""
    from engine.v4_components import tweedie_correct
    rng = np.random.default_rng(1)
    n = 2000
    theta = rng.normal(0, 1.0, n)
    s2 = np.full(n, 0.8 ** 2)
    z = theta + rng.normal(0, 0.8, n)
    corrected = tweedie_correct(pd.Series(z), pd.Series(s2)).to_numpy()
    mse_raw = np.mean((z - theta) ** 2)
    mse_cor = np.mean((corrected - theta) ** 2)
    assert mse_cor < mse_raw * 0.9, (mse_cor, mse_raw)


def test_tweedie_shrinks_extremes_hardest():
    from engine.v4_components import tweedie_correct
    rng = np.random.default_rng(2)
    z = pd.Series(rng.normal(0, 1, 1000))
    corrected = tweedie_correct(z, pd.Series(np.full(1000, 0.5)))
    shift = (corrected - z).to_numpy()
    zz = z.to_numpy()
    assert np.mean(shift[zz > np.quantile(zz, 0.9)]) < 0   # 高端向下拉
    assert np.mean(shift[zz < np.quantile(zz, 0.1)]) > 0   # 低端向上拉
