from omfpandas.omfp import OMFPandas
import omfpandas.volume

from importlib import metadata

try:
    __version__ = metadata.version('omfpandas')
except metadata.PackageNotFoundError:
    # Package is not installed
    pass
