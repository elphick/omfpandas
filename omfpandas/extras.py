_pa = None
_yp = None


def _import_pandera():
    """Helper method to import pandera and handle ImportError."""
    global _pa
    if _pa is None:
        try:
            import pandera as pa
            _pa = pa
        except ImportError:
            raise ImportError("pandera is required to run this method. "
                              "Please install it by running 'poetry install omfpandas --extras validate' "
                              "or 'pip install pandera[io]'")
    return _pa


def _import_ydata_profiling():
    """Helper method to import ydata_profiling and handle ImportError."""
    global _yp
    if _yp is None:
        try:
            import ydata_profiling as yp
            _yp = yp
        except ImportError:
            raise ImportError("ydata-profiling is required to run this method. "
                              "Please install it by running 'poetry install omfpandas --extras profile' "
                              "or 'pip install ydata-profiling'")
    return _yp
