import pandas as pd


def is_nullable_integer_dtype(series: pd.Series) -> bool:
    """

    Args:
        series: The series

    Returns:
        bool: True if series contains nullable integer
    """

    return True if str(series.dtype)[0] == "I" else False


def to_nullable_integer_dtype(series: pd.Series) -> pd.Series:
    """ Convert an int series to a nullable integer dtype

    Args:
        series: The series

    Returns:
        pd.Series: The series with nullable dtype
    """

    return series.astype(str(series.dtype).replace("i", "I")) if is_nullable_integer_dtype(series) else series


def to_numpy_integer_dtype(series: pd.Series) -> pd.Series:
    """ Convert a nullable int series to a numpy integer dtype

    Args:
        series: The series

    Returns:
        pd.Series: The series with nullable dtype
    """

    return series.astype(str(series.dtype).replace("I", "i")) if is_nullable_integer_dtype(series) else series
