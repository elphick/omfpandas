import json
import tempfile
from pathlib import Path

import omf
import pytest
import numpy as np
import pandas as pd
from omf import Project

from omfpandas.blockmodels.geometry import RegularGeometry, TensorGeometry

from omfpandas import OMFPandasReader
from conftest import requires_omf_version, get_omf_file


def test_regular_geometry_initialization():
    corner = (0.0, 0.0, 0.0)
    axis_u = (1.0, 0.0, 0.0)
    axis_v = (0.0, 1.0, 0.0)
    axis_w = (0.0, 0.0, 1.0)
    block_size = (10.0, 10.0, 10.0)
    shape = (10, 10, 10)

    geometry = RegularGeometry(corner, axis_u, axis_v, axis_w, block_size, shape)

    assert geometry.corner == corner
    assert geometry.axis_u == axis_u
    assert geometry.axis_v == axis_v
    assert geometry.axis_w == axis_w
    assert geometry.block_size == block_size
    assert geometry.shape == shape


def test_regular_geometry_centroids():
    corner = (0.0, 0.0, 0.0)
    axis_u = (1.0, 0.0, 0.0)
    axis_v = (0.0, 1.0, 0.0)
    axis_w = (0.0, 0.0, 1.0)
    block_size = (10.0, 10.0, 10.0)
    shape = (10, 10, 10)

    geometry = RegularGeometry(corner, axis_u, axis_v, axis_w, block_size, shape)

    assert np.allclose(geometry.centroid_u, np.arange(5.0, 105.0, 10.0))
    assert np.allclose(geometry.centroid_v, np.arange(5.0, 105.0, 10.0))
    assert np.allclose(geometry.centroid_w, np.arange(5.0, 105.0, 10.0))


def test_regular_geometry_extents():
    corner = (0.0, 0.0, 0.0)
    axis_u = (1.0, 0.0, 0.0)
    axis_v = (0.0, 1.0, 0.0)
    axis_w = (0.0, 0.0, 1.0)
    block_size = (10.0, 10.0, 10.0)
    shape = (10, 10, 10)

    geometry = RegularGeometry(corner, axis_u, axis_v, axis_w, block_size, shape)

    expected_extents = ((0.0, 100.0), (0.0, 100.0), (0.0, 100.0))
    assert geometry.extents == expected_extents


@requires_omf_version('v2')
def test_tensor_geometry_initialization():
    corner = (0.0, 0.0, 0.0)
    axis_u = (1.0, 0.0, 0.0)
    axis_v = (0.0, 1.0, 0.0)
    axis_w = (0.0, 0.0, 1.0)
    tensor_u = np.full(10, 10.0)
    tensor_v = np.full(10, 10.0)
    tensor_w = np.full(10, 10.0)

    geometry = TensorGeometry(corner, axis_u, axis_v, axis_w, tensor_u, tensor_v, tensor_w)

    assert geometry.corner == corner
    assert geometry.axis_u == axis_u
    assert geometry.axis_v == axis_v
    assert geometry.axis_w == axis_w
    assert np.allclose(geometry.tensor_u, tensor_u)
    assert np.allclose(geometry.tensor_v, tensor_v)
    assert np.allclose(geometry.tensor_w, tensor_w)


@requires_omf_version('v2')
def test_tensor_geometry_centroids():
    corner = (0.0, 0.0, 0.0)
    axis_u = (1.0, 0.0, 0.0)
    axis_v = (0.0, 1.0, 0.0)
    axis_w = (0.0, 0.0, 1.0)
    tensor_u = np.full(10, 10.0)
    tensor_v = np.full(10, 10.0)
    tensor_w = np.full(10, 10.0)

    geometry = TensorGeometry(corner, axis_u, axis_v, axis_w, tensor_u, tensor_v, tensor_w)

    assert np.allclose(geometry.centroid_u, np.arange(5.0, 105.0, 10.0))
    assert np.allclose(geometry.centroid_v, np.arange(5.0, 105.0, 10.0))
    assert np.allclose(geometry.centroid_w, np.arange(5.0, 105.0, 10.0))


@requires_omf_version('v2')
def test_tensor_geometry_extents():
    corner = (0.0, 0.0, 0.0)
    axis_u = (1.0, 0.0, 0.0)
    axis_v = (0.0, 1.0, 0.0)
    axis_w = (0.0, 0.0, 1.0)
    tensor_u = np.full(10, 10.0)
    tensor_v = np.full(10, 10.0)
    tensor_w = np.full(10, 10.0)

    geometry = TensorGeometry(corner, axis_u, axis_v, axis_w, tensor_u, tensor_v, tensor_w)

    expected_extents = ((0.0, 100.0), (0.0, 100.0), (0.0, 100.0))
    assert geometry.extents == expected_extents


# @requires_omf_version('v1|v2')
def test_regular_geometry_from_multi_index():
    index = pd.MultiIndex.from_product([range(10), range(10), range(10)], names=['x', 'y', 'z'])
    geometry = RegularGeometry.from_multi_index(index)

    assert geometry.corner == (-0.5, -0.5, -0.5)
    assert geometry.block_size == (1.0, 1.0, 1.0)


@requires_omf_version('v2')
def test_tensor_geometry_from_multi_index():
    index = pd.MultiIndex.from_product([range(10), range(10), range(10), [1.0], [1.0], [1.0]],
                                       names=['x', 'y', 'z', 'dx', 'dy', 'dz'])
    geometry = TensorGeometry.from_multi_index(index)

    assert geometry.corner == (-0.5, -0.5, -0.5)
    assert np.allclose(geometry.tensor_u, np.full(10, 1.0))
    assert np.allclose(geometry.tensor_v, np.full(10, 1.0))
    assert np.allclose(geometry.tensor_w, np.full(10, 1.0))


@requires_omf_version('v1')
def test_regular_geometry_from_element_v1():
    # load the block model from an omf file
    omfp: OMFPandasReader = OMFPandasReader(filepath=get_omf_file())
    bm_element_name: str = 'Block Model'
    geometry = RegularGeometry.from_element(omfp.get_element_by_name(bm_element_name))

    assert np.allclose(geometry.corner, (444700.0, 492800.0, 2330.0))
    assert np.allclose(geometry.axis_u, (1.0, 0.0, 0.0))
    assert np.allclose(geometry.axis_v, (0.0, 1.0, 0.0))
    assert np.allclose(geometry.axis_w, (0.0, 0.0, 1.0))
    assert np.allclose(geometry.block_size, (10.0, 10.0, 10.0))
    assert np.allclose(geometry.shape, (110, 160, 96))


@requires_omf_version('v2')
def test_tensor_regular_geometry_conversion():
    from omf.blockmodel import TensorGridBlockModel, RegularBlockModel
    # load the block model from an omf file
    omfpr: OMFPandasReader = OMFPandasReader(filepath=get_omf_file())
    bm_element_name: str = 'tensor'

    # Get the tensor model
    tensor_model: TensorGridBlockModel = omfpr.get_element_by_name(bm_element_name)
    tensor_geometry: TensorGeometry = TensorGeometry.from_element(omfpr.get_element_by_name(bm_element_name))

    # Convert the tensor model to a regular block model
    regular_model = RegularBlockModel(name=f'{tensor_model.name}_regular',
                                      corner=tensor_model.corner,
                                      axis_u=tensor_model.axis_u,
                                      axis_v=tensor_model.axis_v,
                                      axis_w=tensor_model.axis_w,
                                      block_size=[tensor_model.tensor_u[0],
                                                  tensor_model.tensor_v[0],
                                                  tensor_model.tensor_w[0]],  # Knowing the tensor model is regular
                                      block_count=list(tensor_model.parent_block_count),
                                      cbc=[1] * tensor_model.num_cells
                                      )
    # add the data
    regular_model.attributes = tensor_model.attributes
    regular_model.validate()

    # Create a temporary file to store the regular block model
    with tempfile.NamedTemporaryFile(delete=False, suffix=".omf") as temp_file:
        omfp: Project = Project(name='test')
        omfp.elements.append(regular_model)
        omf.save(project=omfp, filename=temp_file.name, mode='w')

        # Load the regular block model from the temporary file
        omfp_temp: OMFPandasReader = OMFPandasReader(filepath=Path(temp_file.name))
        regular_geometry: RegularGeometry = RegularGeometry.from_element(
            omfp_temp.get_element_by_name(f'{bm_element_name}_regular'))

    assert np.allclose(tensor_geometry.corner, regular_geometry.corner)
    assert np.allclose(tensor_geometry.axis_u, regular_geometry.axis_u)
    assert np.allclose(tensor_geometry.axis_v, regular_geometry.axis_v)
    assert np.allclose(tensor_geometry.axis_w, regular_geometry.axis_w)
    assert np.allclose(tensor_geometry.tensor_u[0], regular_geometry.block_size[0])
    assert np.allclose(tensor_geometry.tensor_v[0], regular_geometry.block_size[1])
    assert np.allclose(tensor_geometry.tensor_w[0], regular_geometry.block_size[2])


@requires_omf_version('v2')
def test_tensor_geometry_from_element():
    # load the block model from an omf file
    omfp: OMFPandasReader = OMFPandasReader(filepath=get_omf_file())
    geometry = TensorGeometry.from_element(omfp.get_element_by_name(element_name='tensor'))

    assert np.allclose(geometry.corner, (10.0, 10.0, -10.0))
    assert np.allclose(geometry.axis_u, (1.0, 0.0, 0.0))
    assert np.allclose(geometry.axis_v, (0.0, 1.0, 0.0))
    assert np.allclose(geometry.axis_w, (0.0, 0.0, 1.0))
    assert np.allclose(geometry.tensor_u, np.full(10, 1.0))
    assert np.allclose(geometry.tensor_v, np.full(15, 1.0))
    assert np.allclose(geometry.tensor_w, np.full(20, 1.0))


@requires_omf_version('v2')
def test_tensor_geometry_json_round_trip():
    from omfpandas.blockmodels.geometry import TensorGeometry

    # Create the object OMFPandas with the path to the OMF file.
    test_omf_path: Path = get_omf_file()
    omfp: OMFPandasReader = OMFPandasReader(filepath=test_omf_path)
    geom: TensorGeometry = omfp.get_bm_geometry(blockmodel_name='tensor')

    # Serialize the geometry to JSON
    json_str = geom.to_json()

    # Deserialize the JSON back to a TensorGeometry object
    geom2 = TensorGeometry.from_json(json_str)

    # Verify that the deserialized object matches the original
    assert np.allclose(geom.corner, geom2.corner)
    assert np.allclose(geom.axis_u, geom2.axis_u)
    assert np.allclose(geom.axis_v, geom2.axis_v)
    assert np.allclose(geom.axis_w, geom2.axis_w)
    assert np.allclose(geom.tensor_u, geom2.tensor_u)
    assert np.allclose(geom.tensor_v, geom2.tensor_v)
    assert np.allclose(geom.tensor_w, geom2.tensor_w)


@requires_omf_version('v2')
def test_tensor_geometry_json_file_round_trip():
    from omfpandas.blockmodels.geometry import TensorGeometry

    # Create the object OMFPandas with the path to the OMF file
    omfp: OMFPandasReader = OMFPandasReader(filepath=get_omf_file())
    geom: TensorGeometry = omfp.get_bm_geometry(blockmodel_name='tensor')

    # Serialize the geometry to a JSON file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
        temp_file_path = Path(temp_file.name)
        geom.to_json_file(temp_file_path)

    try:
        # Deserialize the JSON file back to a TensorGeometry object
        with open(temp_file_path, 'r') as f:
            json_str = f.read()
        geom2 = TensorGeometry.from_json(json_str)

        # Verify that the deserialized object matches the original
        assert np.allclose(geom.corner, geom2.corner)
        assert np.allclose(geom.axis_u, geom2.axis_u)
        assert np.allclose(geom.axis_v, geom2.axis_v)
        assert np.allclose(geom.axis_w, geom2.axis_w)
        assert np.allclose(geom.tensor_u, geom2.tensor_u)
        assert np.allclose(geom.tensor_v, geom2.tensor_v)
        assert np.allclose(geom.tensor_w, geom2.tensor_w)

    finally:
        temp_file_path.unlink()  # Ensure the temporary file is deleted


def test_regular_geometry_json_round_trip():
    from omfpandas.blockmodels.geometry import RegularGeometry

    # Create the object OMFPandas with the path to the OMF file
    omfp: OMFPandasReader = OMFPandasReader(filepath=get_omf_file())
    geom: RegularGeometry = omfp.get_bm_geometry(blockmodel_name='regular')

    # Serialize the geometry to JSON
    json_str = geom.to_json()

    # Deserialize the JSON back to a RegularGeometry object
    geom2 = RegularGeometry.from_json(json_str)

    # Verify that the deserialized object matches the original
    assert np.allclose(geom.corner, geom2.corner)
    assert np.allclose(geom.axis_u, geom2.axis_u)
    assert np.allclose(geom.axis_v, geom2.axis_v)
    assert np.allclose(geom.axis_w, geom2.axis_w)
    assert np.allclose(geom.block_size, geom2.block_size)
    assert np.allclose(geom.shape, geom2.shape)


def test_regular_geometry_json_file_round_trip():
    from omfpandas.blockmodels.geometry import RegularGeometry

    # Create the object OMFPandas with the path to the OMF file.
    test_omf_path: Path = get_omf_file()
    omfp: OMFPandasReader = OMFPandasReader(filepath=test_omf_path)
    geom: RegularGeometry = omfp.get_bm_geometry(blockmodel_name='regular')

    # Serialize the geometry to a JSON file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
        temp_file_path = Path(temp_file.name)
        geom.to_json_file(temp_file_path)

    try:
        # Deserialize the JSON file back to a RegularGeometry object
        with open(temp_file_path, 'r') as f:
            json_str = f.read()
        geom2 = RegularGeometry.from_json(json_str)

        # Verify that the deserialized object matches the original
        assert np.allclose(geom.corner, geom2.corner)
        assert np.allclose(geom.axis_u, geom2.axis_u)
        assert np.allclose(geom.axis_v, geom2.axis_v)
        assert np.allclose(geom.axis_w, geom2.axis_w)
        assert np.allclose(geom.block_size, geom2.block_size)
        assert np.allclose(geom.shape, geom2.shape)

    finally:
        temp_file_path.unlink()  # Ensure the temporary file is deleted


@requires_omf_version('v2')
def test_centroid_lookup():
    from omfpandas.blockmodels.geometry import TensorGeometry

    # Create the object OMFPandas with the path to the OMF file.
    omfp: OMFPandasReader = OMFPandasReader(filepath=get_omf_file())
    geom: TensorGeometry = omfp.get_bm_geometry(blockmodel_name='tensor')

    # Test the lookup
    centroid = geom.nearest_centroid_lookup(0.3, 0.4, 0.6)
    assert centroid == (0.5, 0.5, 0.5)
