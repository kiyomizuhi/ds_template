import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.base import clone


def get_pipeline_model(alpha, random_state):
    """
    scikit-learnのPipelineのインスタンスを返すメソッド。
    StandardScalerとRidgeでpipelineが出来る。

    Args
    ----------
    alpha: float
        L2正則化項
    random_state: int
        ランダムシード

    Return
    ----------
    sklearn.pipeline.Pipeline
        未fitのscikit-learnのPipelineのインスタンス

    """
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha, fit_intercept=True, random_state=random_state)
        )


def clone_fit(pipe_model, X_train, y_train, random_state):
    """
    Pipelineのインスタンスをcloneして学習データでfitして返すメソッド。

    Args
    ----------
    pipe_model: sklearn.pipeline.Pipeline
        StandardScalerとRidgeで繋がったscikit-learnのpipelineのインスタンス
    X_train: pandas.DataFrame
        学習データの説明変数
    y_train: pandas.Series
        学習データの目的変数
    random_state: int
        ランダムシード

    Return
    ----------
    pipe_: sklearn.pipeline.Pipeline
        学習データでfit済みのPipelineのインスタンス

    """
    pipe_ = clone(pipe_model(random_state=random_state))
    pipe_.fit(X_train, y_train)
    return pipe_


def get_residues(pipe_model, X_train, X_test, y_train, y_test, random_state=0):
    """
    Pipelineのインスタンスを学習データでfitして
    テストデータのRMSE（平均二乗誤差）を返すメソッド

    Args
    ----------
    pipe_model: sklearn.pipeline.Pipeline
        StandardScalerとRidgeで繋がったscikit-learnのpipelineのインスタンス
    X_train: pandas.DataFrame
        学習データの説明変数
    y_train: pandas.Series
        学習データの目的変数
    X_test: pandas.DataFrame
        テストデータの説明変数
    y_test: pandas.Series
        テストデータの目的変数
    random_state: int
        ランダムシード

    Return
    ----------
    float
        学習データでfit済みのPipelineのインスタンス

    """
    pipe_ = clone_fit(pipe_model, X_train, y_train, random_state)
    return np.sqrt(mean_squared_error(pipe_.predict(X_test), y_test))


def get_coefs_via_bootstrap(pipe_model, X, y, cols, num_seeds):
    """
    Pipelineのインスタンスをbootstrapした学習データでfitして
    回帰係数を取得する。seedの数だけ繰り返す。

    Args
    ----------
    pipe_model: sklearn.pipeline.Pipeline
        StandardScalerとRidgeで繋がったscikit-learnのpipelineのインスタンス
    X: pandas.DataFrame
        学習データの説明変数
    y: pandas.Series
        学習データの目的変数
    cols: list
        説明変数のカラム名のlist
    num_seed: int
        ランダムシードの数

    Return
    ----------
    df_coefs_orig: pandas.DataFrame
        元の説明変数の回帰係数を格納したDataFrame
    df_coefs_stds: pandas.DataFrame
        正規化した説明変数の回帰係数を格納したDataFrame

    """
    df_coefs_orig = pd.DataFrame(None, columns=cols + ['intercept'])
    df_coefs_stds = pd.DataFrame(None, columns=cols + ['intercept'])
    for i in range(num_seeds):
        np.random.seed(i)
        idxs = np.random.choice(len(y), size=len(y), replace=True)
        pipe_ = clone_fit(pipe_model, X[idxs, :], y[idxs], i)

        # 元の特徴量に対するcoef_
        coefs_ = [
            c/s for c, s in zip(
                pipe_['ridge'].coef_,
                pipe_['standardscaler'].scale_)
            ]
        coefs_.append(pipe_['ridge'].intercept_)
        df_coefs_orig.loc[i] = coefs_

        # 標準化した特徴量に対するcoef_
        coefs_ = list(pipe_['ridge'].coef_)
        coefs_.append(pipe_['ridge'].intercept_)
        df_coefs_stds.loc[i] = coefs_
    return df_coefs_orig, df_coefs_stds
