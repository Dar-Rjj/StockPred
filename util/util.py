import pandas as pd
from typing import Literal


#### 缺失值填补

def fillna(df: pd.DataFrame, method: Literal['mean', 'median', 'pad', 'ffill', 'bfill', 'interpolate']) -> pd.DataFrame:
    """
    Wrapper function to fill NaN in data.
    :param df: a pandas DataFrame.
    :param method: the method to fill NaN.
    :return: a pandas DataFrame with all NaN filled by certain method.
    """
    if method in ['mean', 'median']:
        return df.fillna(getattr(df, method)())
    if method in ['pad']:
        return df.fillna(method=method)
    if method in ['interpolate', 'ffill', 'bfill']:
        return getattr(df, method)()
    
    raise ValueError('Method is not defined')


#### 归一化

def minmax_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Min-Max normalization for a DataFrame.
    """
    return (df - df.min()) / (df.max() - df.min())

def zscore_normalization(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std()


#### 异常极值去除

def mad_filter(df: pd.DataFrame, quantile, axis=0) -> pd.DataFrame:
    median = df.quantile(0.5)
    deviation_median = abs(df - median).quantile(0.5)
    interval = quantile * deviation_median

    return df.clip(median - interval, median + interval, axis=axis^1)

def three_sigma_filter(df: pd.DataFrame, n=3, axis=0) -> pd.DataFrame:
    mean = df.mean(axis=axis)
    interval = n * df.std(axis=axis)

    return df.clip(mean - interval, mean + interval, axis=axis^1)

def percentile_filter(df: pd.DataFrame, min, max, axis=0) -> pd.DataFrame:
    pos = df.quantile([min, max])

    return df.clip(pos.iloc[0], pos.iloc[1], axis=axis^1)