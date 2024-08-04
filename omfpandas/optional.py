def _import_pandera_from_yaml():
    """Helper method to import pandera's from_yaml and handle ImportError."""
    try:
        from pandera.io import from_yaml
        return from_yaml
    except ImportError:
        raise ImportError("pandera is required to run this method.  "
                          "Please install it by running 'poetry install omfpandas --extras validate' "
                          "or 'pip install pandera[io]")


def _import_profilereport():
    """Helper method to import pandas_profiling's ProfileReport and handle ImportError."""
    try:
        from pandas_profiling import ProfileReport
        return ProfileReport
    except ImportError:
        raise ImportError("pandas-profiling is required to run this method.  "
                          "Please install it by running 'poetry install omfpandas --extras profile' "
                          "or 'pip install pandas-profiling'")
