import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd

from omfpandas.blockmodels import multiindex_to_encoded_index

FloatArray = Union[np.ndarray, list[float], np.ndarray[float]]
Vector = Union[tuple[float, float, float], list[float, float, float]]
Point = Union[tuple[float, float, float], list[float, float, float]]
Triple = Union[tuple[float, float, float], list[float, float, float]]
MinMax = Union[tuple[float, float], list[float, float]]


# class CustomEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, Vector3):
#             return obj.to_dict()
#         return super().default(obj)


@dataclass
class Geometry(ABC):
    """Base class for geometry objects.

    The geometry associated with omf block models are not defined by block centroids, and vary by block model type.
    In the pandas representation, the geometry is defined by the block centroids, so this class is used to
    define the geometry in terms of block centroids.
    Additionally, other properties of the geometry are defined here, such as the shape of the geometry.

    Attributes (in omf and pyvista) are stored in Fortran 'F' order, meaning that the last index changes the fastest.
    Hence the MultiIndex levels need to be sorted by 'z', 'y', 'x', to align with the Fortran order.
    This has x changing fastest, z changing slowest.

    """

    corner: Point
    axis_u: Vector
    axis_v: Vector
    axis_w: Vector
    _centroid_u: Optional[FloatArray] = field(default=None, init=False, repr=False)
    _centroid_v: Optional[FloatArray] = field(default=None, init=False, repr=False)
    _centroid_w: Optional[FloatArray] = field(default=None, init=False, repr=False)
    _shape: Optional[Point] = field(default=None, init=False, repr=False)
    _is_regular: Optional[bool] = field(default=None, init=False, repr=False)

    def to_summary_json(self) -> str:
        """Convert the geometry to a JSON string.

        Returns:
            str: The JSON string representing the geometry.
        """
        return json.dumps(self.summary)

    def to_json_file(self, json_filepath: Path) -> Path:
        """Write the Geometry to a JSON file.

        Args:
            json_filepath (Path): The path to write the JSON file.

        Returns:
            Path to the json file.
        """
        json_filepath.write_text(self.to_json())
        return json_filepath

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @property
    @abstractmethod
    def is_regular(self) -> bool:
        pass

    @property
    @abstractmethod
    def centroid_u(self) -> np.ndarray[float]:
        pass

    @property
    @abstractmethod
    def centroid_v(self) -> np.ndarray[float]:
        pass

    @property
    @abstractmethod
    def centroid_w(self) -> np.ndarray[float]:
        pass

    @property
    def num_cells(self) -> int:
        return int(np.prod(self.shape))

    @property
    def shape(self) -> Triple:
        if self._shape is None:
            self._shape = (
                len(self.centroid_u),
                len(self.centroid_v),
                len(self.centroid_w),
            )
        return self._shape

    @property
    @abstractmethod
    def extents(self) -> tuple[MinMax, MinMax, MinMax]:
        pass

    @property
    def bounding_box(self) -> tuple[MinMax, MinMax]:
        return self.extents[0], self.extents[1]

    @property
    @abstractmethod
    def summary(self) -> dict:
        pass

    @classmethod
    @abstractmethod
    def from_element(cls, element) -> "Geometry":
        pass

    @classmethod
    @abstractmethod
    def from_multi_index(cls, index: pd.MultiIndex):
        pass

    @abstractmethod
    def to_multi_index(self) -> pd.MultiIndex:
        pass

    @abstractmethod
    def nearest_centroid_lookup(self, x: float, y: float, z: float) -> Point:
        pass


@dataclass
class RegularGeometry(Geometry):
    """Regular geometry data class.

    Regular Geometry applies to an omf.v1 VolumeElement or an omf.v2 RegularBlockModel.
    """

    block_size: Triple
    _shape: Triple = field(default=None, init=False, repr=False)

    def __init__(
        self,
        corner: Point,
        axis_u: Vector,
        axis_v: Vector,
        axis_w: Vector,
        block_size: Triple,
        shape: Triple = None,
    ):
        self.corner = corner
        self.axis_u = axis_u
        self.axis_v = axis_v
        self.axis_w = axis_w
        self.block_size = block_size
        self._shape = shape

    def __repr__(self):
        return f"RegularGeometry: {self.summary}"

    def __str__(self):
        return f"RegularGeometry: {self.summary}"

    @property
    def is_regular(self) -> bool:
        return True

    @property
    def centroid_u(self) -> np.ndarray[float]:
        if self._centroid_u is None:
            self._centroid_u = np.arange(
                self.corner[0] + self.block_size[0] / 2,
                self.corner[0] + self.block_size[0] * self.shape[0],
                self.block_size[0],
            )
        return self._centroid_u

    @property
    def centroid_v(self) -> np.ndarray[float]:
        if self._centroid_v is None:
            self._centroid_v = np.arange(
                self.corner[1] + self.block_size[1] / 2,
                self.corner[1] + self.block_size[1] * self.shape[1],
                self.block_size[1],
            )
        return self._centroid_v

    @property
    def centroid_w(self) -> np.ndarray[float]:
        if self._centroid_w is None:
            self._centroid_w = np.arange(
                self.corner[2] + self.block_size[2] / 2,
                self.corner[2] + self.block_size[2] * self.shape[2],
                self.block_size[2],
            )
        return self._centroid_w

    @property
    def shape(self) -> Triple:
        return self._shape

    @shape.setter
    def shape(self, value: Triple):
        self._shape = value

    @property
    def extents(self) -> tuple[MinMax, MinMax, MinMax]:
        return (
            (
                float(self.centroid_u[0] - self.block_size[0] / 2),
                float(self.centroid_u[-1] + self.block_size[0] / 2),
            ),
            (
                float(self.centroid_v[0] - self.block_size[1] / 2),
                float(self.centroid_v[-1] + self.block_size[1] / 2),
            ),
            (
                float(self.centroid_w[0] - self.block_size[2] / 2),
                float(self.centroid_w[-1] + self.block_size[2] / 2),
            ),
        )

    @property
    def summary(self) -> dict:
        return {
            "corner": tuple(self.corner),
            "axis_u": tuple(self.axis_u),
            "axis_v": tuple(self.axis_v),
            "axis_w": tuple(self.axis_w),
            "block_size": self.block_size,
            "shape": self.shape,
            "is_regular": self.is_regular,
            "extents": self.extents,
            "bounding_box": self.bounding_box,
        }

    @classmethod
    def from_element(cls, element) -> "RegularGeometry":
        from omf import RegularBlockModel

        if not isinstance(element, RegularBlockModel):
            raise ValueError("Element must be an instance of omf.RegularBlockModel")
        return cls(
            element.corner,
            element.axis_u,
            element.axis_v,
            element.axis_w,
            element.block_size,
            shape=element.block_count,
        )

    @classmethod
    def from_multi_index(
        cls,
        index: pd.MultiIndex,
        axis_u: Vector = (1, 0, 0),
        axis_v: Vector = (0, 1, 0),
        axis_w: Vector = (0, 0, 1),
    ) -> "RegularGeometry":

        # check that the index contains the expected levels
        if not {"x", "y", "z"}.issubset(index.names):
            raise ValueError("Index must contain the levels 'x', 'y', 'z'.")

        x = index.get_level_values("x").unique()
        y = index.get_level_values("y").unique()
        z = index.get_level_values("z").unique()

        # check the block sizes are unique
        dx = np.unique(np.diff(x))
        dy = np.unique(np.diff(y))
        dz = np.unique(np.diff(z))

        # Use the minimum difference as the block size
        block_size = float(dx.min()), float(dy.min()), float(dz.min())

        # Calculate the shape based on the full range of coordinates
        full_shape = (
            len(np.arange(x.min(), x.max() + block_size[0], block_size[0])),
            len(np.arange(y.min(), y.max() + block_size[1], block_size[1])),
            len(np.arange(z.min(), z.max() + block_size[2], block_size[2])),
        )

        corner_x = x.min() - block_size[0] / 2
        corner_y = y.min() - block_size[1] / 2
        corner_z = z.min() - block_size[2] / 2

        # Create the volume
        return cls(
            corner=(corner_x, corner_y, corner_z),
            axis_u=axis_u,
            axis_v=axis_v,
            axis_w=axis_w,
            block_size=block_size,
            shape=full_shape,
        )

    @classmethod
    def from_extents(
        cls,
        extents: tuple[MinMax, MinMax, MinMax],
        block_size: Triple,
        axis_u: Vector = (1, 0, 0),
        axis_v: Vector = (0, 1, 0),
        axis_w: Vector = (0, 0, 1),
    ) -> "RegularGeometry":
        """Create a RegularGeometry from extents."""
        min_x, max_x = extents[0]
        min_y, max_y = extents[1]
        min_z, max_z = extents[2]

        corner = (
            min_x - block_size[0] / 2,
            min_y - block_size[1] / 2,
            min_z - block_size[2] / 2,
        )
        shape = (
            int((max_x - min_x) / block_size[0]),
            int((max_y - min_y) / block_size[1]),
            int((max_z - min_z) / block_size[2]),
        )

        return cls(corner, axis_u, axis_v, axis_w, block_size, shape)

    def to_json(self) -> str:
        """Convert the full geometry to a JSON string."""
        data = {
            "corner": list(self.corner),
            "axis_u": list(self.axis_u),
            "axis_v": list(self.axis_v),
            "axis_w": list(self.axis_w),
            "block_size": list(self.block_size),
            "shape": list(self.shape),
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> "RegularGeometry":
        """Deserialize a JSON string to a full geometry object."""
        data = json.loads(json_str)
        return cls(
            corner=list(data["corner"]),
            axis_u=list(data["axis_u"]),
            axis_v=list(data["axis_v"]),
            axis_w=list(data["axis_w"]),
            block_size=list(data["block_size"]),
            shape=list(data["shape"]),
        )

    def to_multi_index(self) -> pd.MultiIndex:
        """Convert a RegularGeometry to a MultiIndex.

        The MultiIndex will have the following levels:
        - x: The x coordinates of the cell centres
        - y: The y coordinates of the cell centres
        - z: The z coordinates of the cell centres
        """

        """Returns a pd.MultiIndex for the regular blockmodel element, accounting for rotation.

        Args:
            blockmodel (BaseBlockModel): The regular BlockModel to get the index from.

        Returns:
            pd.MultiIndex: The MultiIndex representing the blockmodel element geometry.
        """
        ox, oy, oz = self.corner
        dx, dy, dz = self.block_size
        nx, ny, nz = self.shape

        # Calculate the coordinates of the block centers
        x = ox + (np.arange(nx) + 0.5) * dx
        y = oy + (np.arange(ny) + 0.5) * dy
        z = oz + (np.arange(nz) + 0.5) * dz

        # Create a grid of coordinates
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        # Flatten the grid
        centroids = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()])

        # Rotation axes
        u, v, w = self.axis_u, self.axis_v, self.axis_w

        # Create rotation matrix
        rotation_matrix = np.array([u, v, w]).T

        # Apply rotation
        rotated_centroids = rotation_matrix @ centroids

        # Create a MultiIndex
        index = pd.MultiIndex.from_arrays(
            [rotated_centroids[0], rotated_centroids[1], rotated_centroids[2]],
            names=["x", "y", "z"],
        )

        # Sort the MultiIndex by x, y, z levels
        return index.sortlevel(level=["x", "y", "z"])[0]

    def to_encoded_index(self) -> pd.Index:
        """Convert a RegularGeometry to an encoded integer index

        The integer index is encoded to preserve the spatial position.

        Use the coordinate_hashing.hashed_index_to_multiindex function to convert it back to x, y, z pd.MultiIndex

        Returns:

        """
        return multiindex_to_encoded_index(self.to_multi_index())

    def nearest_centroid_lookup(self, x: float, y: float, z: float) -> Point:
        """Find the nearest centroid for provided x, y, z points.

        Args:
            x (float): X coordinate.
            y (float): Y coordinate.
            z (float): Z coordinate.

        Returns:
            Point3: The coordinates of the nearest centroid.
        """

        reference_centroid: Point = (
            self.centroid_u[0],
            self.centroid_v[0],
            self.centroid_w[0],
        )
        dx, dy, dz = self.block_size
        ref_x, ref_y, ref_z = reference_centroid

        nearest_x = round((x - ref_x) / dx) * dx + ref_x
        nearest_y = round((y - ref_y) / dy) * dy + ref_y
        nearest_z = round((z - ref_z) / dz) * dz + ref_z

        return nearest_x, nearest_y, nearest_z


@dataclass
class TensorGeometry(Geometry):
    """Tensor geometry data class.

    Applicable for on omf.v2 TensorGridBlockModel.
    """

    tensor_u: FloatArray
    tensor_v: FloatArray
    tensor_w: FloatArray
    _block_sizes: Optional[Point] = field(default=None, init=False)

    def __repr__(self):
        return f"TensorGeometry: {self.summary}"

    def __str__(self):
        return f"TensorGeometry: {self.summary}"

    @property
    def is_regular(self) -> bool:
        return (
            np.allclose(self.tensor_u, self.tensor_u[0])
            and np.allclose(self.tensor_v, self.tensor_v[0])
            and np.allclose(self.tensor_w, self.tensor_w[0])
        )

    @property
    def centroid_u(self) -> np.ndarray[float]:
        if self._centroid_u is None:
            self._centroid_u = (
                self.corner[0] + np.cumsum(self.tensor_u) - self.tensor_u / 2
            )
        return self._centroid_u

    @property
    def centroid_v(self) -> np.ndarray[float]:
        if self._centroid_v is None:
            self._centroid_v = (
                self.corner[0] + np.cumsum(self.tensor_v) - self.tensor_v / 2
            )
        return self._centroid_v

    @property
    def centroid_w(self) -> np.ndarray[float]:
        if self._centroid_w is None:
            self._centroid_w = (
                self.corner[0] + np.cumsum(self.tensor_w) - self.tensor_w / 2
            )
        return self._centroid_w

    @property
    def block_sizes(self) -> Triple:
        if self._block_sizes is None:
            # Create a grid of all possible combinations
            grid = np.array(
                np.meshgrid(self.tensor_u, self.tensor_v, self.tensor_w)
            ).T.reshape(-1, 3)
            # Find unique combinations
            unique_block_sizes = np.unique(grid, axis=0)
            self._block_sizes = [tuple(map(float, size)) for size in unique_block_sizes]
        return self._block_sizes

    @property
    def extents(self) -> tuple[MinMax, MinMax, MinMax]:
        return (
            (
                float(self.centroid_u[0] - self.tensor_u[0] / 2),
                float(self.centroid_u[-1] + self.tensor_u[0] / 2),
            ),
            (
                float(self.centroid_v[0] - self.tensor_v[0] / 2),
                float(self.centroid_v[-1] + self.tensor_v[0] / 2),
            ),
            (
                float(self.centroid_w[0] - self.tensor_w[0] / 2),
                float(self.centroid_w[-1] + self.tensor_w[0] / 2),
            ),
        )

    @property
    def summary(self) -> dict:
        return {
            "corner": tuple(self.corner),
            "axis_u": tuple(self.axis_u),
            "axis_v": tuple(self.axis_v),
            "axis_w": tuple(self.axis_w),
            "block_sizes": self.block_sizes,
            "shape": self.shape,
            "is_regular": self.is_regular,
            "extents": self.extents,
            "bounding_box": self.bounding_box,
        }

    @classmethod
    def from_element(cls, element) -> "TensorGeometry":
        from omf import TensorGridBlockModel

        if not isinstance(element, TensorGridBlockModel):
            raise ValueError("Element must be an instance of omf.TensorGridBlockModel")

        return cls(
            element.corner,
            element.axis_u,
            element.axis_v,
            element.axis_w,
            element.tensor_u,
            element.tensor_v,
            element.tensor_w,
        )

    @classmethod
    def from_multi_index(cls, index: pd.MultiIndex) -> "TensorGeometry":
        # check that the index contains the expected levels
        level_names: list[str] = ["x", "y", "z", "dx", "dy", "dz"]
        if not set(level_names).issubset(index.names):
            raise ValueError(f"Index must contain the levels {level_names}.")

        x = index.get_level_values("x").unique()
        y = index.get_level_values("y").unique()
        z = index.get_level_values("z").unique()

        # Get the shape of the original 3D arrays
        shape = (len(x), len(y), len(z))

        # Reshape the ravelled index back into the original shapes
        tensor_u = index.get_level_values("dx").values.reshape(shape, order="F")[
            :, 0, 0
        ]
        tensor_v = index.get_level_values("dy").values.reshape(shape, order="F")[
            0, :, 0
        ]
        tensor_w = index.get_level_values("dz").values.reshape(shape, order="F")[
            0, 0, :
        ]

        origin_x = x.min() - tensor_u[0] / 2
        origin_y = y.min() - tensor_v[0] / 2
        origin_z = z.min() - tensor_w[0] / 2

        # Create the volume
        return cls(
            corner=(origin_x, origin_y, origin_z),
            axis_u=(1, 0, 0),
            axis_v=(0, 1, 0),
            axis_w=(0, 0, 1),
            tensor_u=tensor_u,
            tensor_v=tensor_v,
            tensor_w=tensor_w,
        )

    def to_json(self) -> str:
        """Convert the full geometry to a JSON string."""
        data = {
            "corner": list(self.corner),
            "axis_u": list(self.axis_u),
            "axis_v": list(self.axis_v),
            "axis_w": list(self.axis_w),
            "tensor_u": self.tensor_u.tolist(),
            "tensor_v": self.tensor_v.tolist(),
            "tensor_w": self.tensor_w.tolist(),
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> "TensorGeometry":
        """Deserialize a JSON string to a full geometry object."""
        data = json.loads(json_str)
        return cls(
            corner=list(data["corner"]),
            axis_u=list(data["axis_u"]),
            axis_v=list(data["axis_v"]),
            axis_w=list(data["axis_w"]),
            tensor_u=np.array(data["tensor_u"]),
            tensor_v=np.array(data["tensor_v"]),
            tensor_w=np.array(data["tensor_w"]),
        )

    def to_multi_index(self) -> pd.MultiIndex:
        """Convert a TensorGeometry to a MultiIndex.

        The MultiIndex will have the following levels:
        - x: The x coordinates of the cell centres
        - y: The y coordinates of the cell centres
        - z: The z coordinates of the cell centres
        - dx: The x cell sizes
        - dy: The y cell sizes
        - dz: The z cell sizes
        """

        """Returns a pd.MultiIndex for the tensor blockmodel element.

        Args:
            blockmodel (BlockModel): The tensor BlockModel to get the index from.

        Returns:
            pd.MultiIndex: The MultiIndex representing the blockmodel element geometry.
        """
        ox, oy, oz = self.corner

        # Make coordinates (points) along each axis, i, j, k
        i = ox + np.cumsum(self.tensor_u)
        i = np.insert(i, 0, ox)
        j = oy + np.cumsum(self.tensor_v)
        j = np.insert(j, 0, oy)
        k = oz + np.cumsum(self.tensor_w)
        k = np.insert(k, 0, oz)

        # convert to centroids
        x, y, z = (i[1:] + i[:-1]) / 2, (j[1:] + j[:-1]) / 2, (k[1:] + k[:-1]) / 2
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        # Calculate dx, dy, dz
        dxx, dyy, dzz = np.meshgrid(
            self.tensor_u, self.tensor_v, self.tensor_w, indexing="ij"
        )

        # TODO: consider rotation

        index: pd.MultiIndex = pd.MultiIndex.from_arrays(
            [xx.ravel(), yy.ravel(), zz.ravel(), dxx.ravel(), dyy.ravel(), dzz.ravel()],
            names=["x", "y", "z", "dx", "dy", "dz"],
        )

        # Sort the MultiIndex by x, y, z levels
        return index.sortlevel(level=["x", "y", "z"])[0]

    def nearest_centroid_lookup(self, x: float, y: float, z: float) -> Point:
        """Find the nearest centroid for provided x, y, z points.

        Args:
            x (float): X coordinate.
            y (float): Y coordinate.
            z (float): Z coordinate.

        Returns:
            Point3: The coordinates of the nearest centroid.
        """

        # This only works for regular geometries in a tensor format - needs to be extended.
        if not self.is_regular:
            raise ValueError(
                "TensorGeometry is not regular. Cannot perform nearest centroid lookup."
            )

        reference_centroid: Point = (
            self.centroid_u[0],
            self.centroid_v[0],
            self.centroid_w[0],
        )
        dx, dy, dz = self.block_sizes[0]
        ref_x, ref_y, ref_z = reference_centroid

        nearest_x = round((x - ref_x) / dx) * dx + ref_x
        nearest_y = round((y - ref_y) / dy) * dy + ref_y
        nearest_z = round((z - ref_z) / dz) * dz + ref_z

        return nearest_x, nearest_y, nearest_z
