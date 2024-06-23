from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from omf import VolumeElement, VolumeGridGeometry, ScalarArray, ScalarData


def volume_to_df(volume: VolumeElement, variables: Optional[list[str]] = None,
                 with_geometry_index: bool = True) -> pd.DataFrame:
    """Convert volume to a DataFrame.

    Args:
        volume (VolumeElement): The VolumeElement to convert.
        variables (Optional[list[str]]): The variables to include in the DataFrame. If None, all variables are included.
        with_geometry_index (bool): If True, includes geometry index in the DataFrame. Default is True.

    Returns:
        pd.DataFrame: The DataFrame representing the VolumeElement.
    """
    # read the data
    df: pd.DataFrame = read_volume_variables(volume, variables=variables)
    if with_geometry_index:
        df.index = geometry_to_index(volume)
    return df


def df_to_volume(df: pd.DataFrame, volume_name: str) -> VolumeElement:
    """Write a DataFrame to a VolumeElement.

    Args:
        df (pd.DataFrame): The DataFrame to convert to a VolumeElement.
        volume_name (str): The name of the VolumeElement.

    Returns:
        VolumeElement: The VolumeElement representing the DataFrame.
    """
    # Get the original order
    original_order = df.index

    try:
        # Sort the dataframe to align with the omf spec
        df.sort_index(level=['z', 'y', 'x'], inplace=True)

        # Create the volume and geometry
        volume = VolumeElement(name=volume_name)
        geometry: VolumeGridGeometry = index_to_geometry(df.index)
        volume.geometry = geometry

        # add the data
        scalars: list[ScalarData] = []
        for variable in df.columns:
            sd: ScalarData = ScalarData()
            sd.array = df[variable].values
            sd.name = variable
            sd.location = 'cells'
            scalars.append(sd)
        volume.data = scalars

    finally:
        # Reset the index to the original order (to avoid side effects)
        df = df.reindex(original_order)
    return volume


def volume_to_parquet(volume: VolumeElement, out_path: Optional[Path] = None, variables: Optional[list[str]] = None,
                      with_geometry_index: bool = True, allow_overwrite: bool = False):
    """Convert volume to a Parquet file.

    Args:
        volume (VolumeElement): The VolumeElement to convert.
        out_path (Optional[Path]): The path to the Parquet file to write. If None, a file with the volume name is created.
        variables (Optional[list[str]]): The variables to include in the DataFrame. If None, all variables are included.
        with_geometry_index (bool): If True, includes geometry index in the DataFrame. Default is True.
        allow_overwrite (bool): If True, overwrite the existing Parquet file. Default is False.

    Raises:
        FileExistsError: If the file already exists and allow_overwrite is False.
    """
    if out_path is None:
        out_path = Path(f"{volume.name}.parquet")
    if out_path.exists() and not allow_overwrite:
        raise FileExistsError(f"File already exists: {out_path}. If you want to overwrite, set allow_overwrite=True.")
    df: pd.DataFrame = volume_to_df(volume, variables=variables, with_geometry_index=with_geometry_index)
    df.to_parquet(out_path)


def read_volume_variables(volume: VolumeElement, variables: list[str]) -> pd.DataFrame:
    """Read the variables from the VolumeElement.

    Args:
        volume (VolumeElement): The VolumeElement to read from.
        variables (list[str]): The variables to include in the DataFrame.

    Returns:
        pd.DataFrame: The DataFrame representing the variables in the VolumeElement.

    Raises:
        ValueError: If the variable is not found in the VolumeElement.
    """
    # identify 'cell' variables in the file
    variables = [v.name for v in volume.data if v.location == 'cells']

    # Loop over the variables
    chunks: list[np.ndarray] = []
    for variable in variables:
        # Check if the variable exists in the VolumeElement
        if variable not in variables:
            raise ValueError(f"Variable '{variable}' not found in the VolumeElement: {volume.name}")
        chunks.append(_get_variable_data_by_name(volume, variable).ravel())

    # Concatenate all chunks into a single DataFrame
    return pd.DataFrame(np.vstack(chunks), index=variables).T


def geometry_to_index(volume: VolumeElement) -> pd.MultiIndex:
    """Returns a pd.MultiIndex for the volume element.

    Args:
        volume (VolumeElement): The VolumeElement to get the index from.

    Returns:
        pd.MultiIndex: The MultiIndex representing the volume element geometry.
    """
    geometry: VolumeGridGeometry = volume.geometry
    ox, oy, oz = geometry.origin

    # Make coordinates (points) along each axis, i, j, k
    i = ox + np.cumsum(geometry.tensor_u)
    i = np.insert(i, 0, ox)
    j = oy + np.cumsum(geometry.tensor_v)
    j = np.insert(j, 0, oy)
    k = oz + np.cumsum(geometry.tensor_w)
    k = np.insert(k, 0, oz)

    # convert to centroids
    x, y, z = (i[1:] + i[:-1]) / 2, (j[1:] + j[:-1]) / 2, (k[1:] + k[:-1]) / 2
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    # Calculate dx, dy, dz
    dxx, dyy, dzz = np.meshgrid(geometry.tensor_u, geometry.tensor_v, geometry.tensor_w, indexing="ij")

    # TODO: consider rotation

    index = pd.MultiIndex.from_arrays([xx.ravel("F"), yy.ravel("F"), zz.ravel("F"),
                                       dxx.ravel("F"), dyy.ravel("F"), dzz.ravel("F")],
                                      names=['x', 'y', 'z', 'dx', 'dy', 'dz'])

    return index


def index_to_geometry(index: pd.MultiIndex) -> VolumeGridGeometry:
    """Convert a MultiIndex to a VolumeGridGeometry.

    Args:
        index (pd.MultiIndex): The MultiIndex to convert to a VolumeGridGeometry.

    Returns:
        VolumeGridGeometry: The VolumeGridGeometry representing the MultiIndex.
    """
    # check that the index contains the expected levels
    if not {'x', 'y', 'z', 'dx', 'dy', 'dz'}.issubset(index.names):
        raise ValueError("Index must contain the levels 'x', 'y', 'z', 'dx', 'dy', 'dz'.")

    x = index.get_level_values('x').unique()
    y = index.get_level_values('y').unique()
    z = index.get_level_values('z').unique()

    # Get the shape of the original 3D arrays
    shape = (len(x), len(y), len(z))

    # Reshape the ravelled index back into the original shapes
    tensor_u = index.get_level_values('dx').values.reshape(shape, order='F')[:, 0, 0]
    tensor_v = index.get_level_values('dy').values.reshape(shape, order='F')[0, :, 0]
    tensor_w = index.get_level_values('dz').values.reshape(shape, order='F')[0, 0, :]

    origin_x = x.min() - tensor_u[0] / 2
    origin_y = y.min() - tensor_v[0] / 2
    origin_z = z.min() - tensor_w[0] / 2

    # Create the volume
    geometry: VolumeGridGeometry = VolumeGridGeometry()
    geometry.origin = np.array([origin_x, origin_y, origin_z])
    geometry.axis_u, geometry.axis_v, geometry.axis_w = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    geometry.tensor_u = tensor_u
    geometry.tensor_v = tensor_v
    geometry.tensor_w = tensor_w

    return geometry


def _get_variable_data_by_name(volume: VolumeElement, variable_name: str) -> np.ndarray:
    """Get the variable data by its name from a VolumeElement.

    Args:
        volume (VolumeElement): The VolumeElement to get the data from.
        variable_name (str): The name of the variable to retrieve.

    Returns:
        np.ndarray: The data of the variable in the VolumeElement.

    Raises:
        ValueError: If the variable is not found as cell data in the VolumeElement or if multiple variables with the same name are found.
    """
    scalar_data = [sd for sd in volume.data if sd.location == 'cells' and sd.name == variable_name]
    if not scalar_data:
        raise ValueError(f"Variable '{variable_name}' not found as cell data in the VolumeElement: {volume}")
    elif len(scalar_data) > 1:
        raise ValueError(f"Multiple variables with the name '{variable_name}' found in the VolumeElement: {volume}")
    return scalar_data[0].array.array
