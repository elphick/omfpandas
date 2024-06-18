from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from omf import VolumeElement, OMFReader, VolumeGridGeometry


def volume_to_df(volume: VolumeElement, variables: Optional[list[str]] = None,
                 with_geometry_index: bool = True) -> pd.DataFrame:
    """Convert volume to a DataFrame."""
    # read the data
    df: pd.DataFrame = read_volume_variables(volume, variables=variables)
    if with_geometry_index:
        df.index = get_index(volume)
    return df


def volume_to_parquet(volume: VolumeElement, out_path: Optional[Path] = None, variables: Optional[list[str]] = None,
                      with_geometry_index: bool = True):
    """Convert volume to a Parquet file."""
    if out_path is None:
        out_path = Path(f"{volume.name}.parquet")
    df: pd.DataFrame = volume_to_df(volume, variables=variables, with_geometry_index=with_geometry_index)
    df.to_parquet(out_path)


def read_volume_variables(volume: VolumeElement, variables: list[str]) -> pd.DataFrame:
    """Read the variables from the VolumeElement."""
    # identify 'cell' variables in the file
    variables = [v.name for v in volume.data if v.location == 'cells']

    # Loop over the variables
    chunks: list[np.ndarray] = []
    for variable in variables:
        # Check if the variable exists in the VolumeElement
        if variable not in variables:
            raise ValueError(f"Variable '{variable}' not found in the VolumeElement: {element}")
        chunks.append(_get_variable_data_by_name(volume, variable).ravel())

    # Concatenate all chunks into a single DataFrame
    return pd.DataFrame(np.vstack(chunks), index=variables).T


def get_index(volume: VolumeElement) -> pd.MultiIndex:
    """Returns a pd.MultiIndex for the volume element."""
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


def _get_variable_data_by_name(volume: VolumeElement, variable_name: str) -> np.ndarray:
    # get the index of the variable in order to index into elements
    scalar_data = [sd for sd in volume.data if sd.location == 'cells' and sd.name == variable_name]
    if not scalar_data:
        raise ValueError(f"Variable '{variable_name}' not found as cell data in the VolumeElement: {volume}")
    elif len(scalar_data) > 1:
        raise ValueError(f"Multiple variables with the name '{variable_name}' found in the VolumeElement: {volume}")
    return scalar_data[0].array.array
