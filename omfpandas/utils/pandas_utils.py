import tokenize
from io import StringIO
from token import STRING

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


def parse_vars_from_expr(expr: str) -> list[str]:
    """ Parse variables from a pandas query expression string.

    Args:
        expr: The expression string

    Returns:
        list[str]: The list of variables
    """
    variables = set()
    tokens = tokenize.generate_tokens(StringIO(expr).readline)
    logical_operators = {'and', 'or', '&', '|'}
    inside_backticks = False
    current_var = []

    for token in tokens:
        if token.string == '`':
            if inside_backticks:
                # End of backtick-enclosed variable
                variables.add(' '.join(current_var))
                current_var = []
            inside_backticks = not inside_backticks
        elif inside_backticks:
            if token.type in {tokenize.NAME, STRING}:
                current_var.append(token.string)
        elif token.type == tokenize.NAME and token.string not in logical_operators:
            variables.add(token.string)

    return list(variables)
