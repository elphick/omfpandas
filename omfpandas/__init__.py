from omfpandas.reader import OMFPandasReader
from omfpandas.writer import OMFPandasWriter
from importlib import metadata

try:
    __version__ = metadata.version('omfpandas')
except metadata.PackageNotFoundError:
    # Package is not installed
    pass

try:
    omf_version = metadata.version('omf')
    if omf_version.startswith('1'):
        __omf_version__ = 'v1'
    elif omf_version.startswith('2'):
        __omf_version__ = 'v2'
    else:
        raise ValueError(f"Unsupported omf version: {omf_version}")
except metadata.PackageNotFoundError:
    # OMF package is not installed
    __omf_version__ = None
