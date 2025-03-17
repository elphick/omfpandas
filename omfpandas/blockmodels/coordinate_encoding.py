"""
Functions to support coordinate encoding/decoding for block models.
NOTE: The encoding scheme can only support x, y, z, given the 64 bit limit, so only regular block models are supported.
"""
import pandas as pd


def encode_coordinates(x: float, y: float, z: float) -> int:
    """Encode the coordinates into a 64-bit integer."""
    x_int = int(x * 10) & 0xFFFFFF
    y_int = int(y * 10) & 0xFFFFFF
    z_int = int(z * 10) & 0xFFFF
    encoded = (x_int << 40) | (y_int << 16) | z_int
    return encoded


def decode_coordinates(encoded: int) -> tuple:
    """Decode the 64-bit integer back to the original coordinates."""
    x_int = (encoded >> 40) & 0xFFFFFF
    y_int = (encoded >> 16) & 0xFFFFFF
    z_int = encoded & 0xFFFF
    x = x_int / 10.0
    y = y_int / 10.0
    z = z_int / 10.0
    return x, y, z


def multiindex_to_encoded_index(multi_index: pd.MultiIndex) -> pd.Index:
    """Convert a MultiIndex to an encoded integer Index."""
    encoded_indices = [
        encode_coordinates(x, y, z)
        for x, y, z in zip(
            multi_index.get_level_values("x"),
            multi_index.get_level_values("y"),
            multi_index.get_level_values("z"),
        )
    ]
    return pd.Index(encoded_indices, name='encoded_xyz')


def encoded_index_to_multiindex(encoded_index: pd.Index) -> pd.MultiIndex:
    """Convert an encoded integer Index back to a MultiIndex."""
    decoded_coords = [decode_coordinates(encoded) for encoded in encoded_index]
    x, y, z = zip(*decoded_coords)
    return pd.MultiIndex.from_arrays([x, y, z], names=["x", "y", "z"])