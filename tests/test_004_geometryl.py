from pathlib import Path

import yaml

from omfpandas import OMFPandasReader
from omfpandas.blockmodel import TensorGeometry


def test_write_geometry():
    # Create the object OMFPandas with the path to the OMF file.
    test_omf_path: Path = Path('../assets/v2/test_file.omf')
    omfp: OMFPandasReader = OMFPandasReader(filepath=test_omf_path)
    geom: TensorGeometry = omfp.get_bm_geometry(blockmodel_name='vol')
    d_geom: dict = geom.to_dict()

    print(geom)
    geom.to_yaml_file('./geom.yaml')
    geom.to_json_file('./geom.json')

