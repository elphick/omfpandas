def df_to_blockmodel_factory(is_tensor: bool = False):
    """
    Get the appropriate function to convert a DataFrame to a BlockModel.

    Args:
        is_tensor (bool): If True, returns the function for TensorGridBlockModel. If False, returns the function for RegularBlockModel.

    Returns:
        function: The function to convert a DataFrame to a BlockModel.
    """
    from omfpandas import __omf_version__

    if __omf_version__ == 'v1':
        from omfpandas.blockmodels.v1.volume import df_to_volume_bm as df_to_blockmodel
    elif __omf_version__ == 'v2':
        if is_tensor:
            from omfpandas.blockmodels.v2.tensor import df_to_tensor_bm as df_to_blockmodel
        else:
            from omfpandas.blockmodels.v2.regular import df_to_regular_bm as df_to_blockmodel
    else:
        raise ValueError(f"Unsupported omf version: {__omf_version__}")
    return df_to_blockmodel


def blockmodel_to_df_factory(is_tensor: bool = False):
    """Get the appropriate function to convert a block model to a DataFrame.

    Args:
        is_tensor (bool): If True, a TensorGridBlockModel function will be returned. If False, a VolumeBlockModel function will be returned.

    Returns:
        function: The function to convert a block model to a DataFrame.

    Raises:
        ValueError: If the OMF version is not supported.
    """
    from omfpandas import __omf_version__

    if __omf_version__ == 'v1':
        from omfpandas.blockmodels.v1.volume import volume_bm_to_df
        return volume_bm_to_df
    elif __omf_version__ == 'v2':
        if is_tensor:
            from omfpandas.blockmodels.v2.tensor import tensor_bm_to_df
            return tensor_bm_to_df
        else:
            from omfpandas.blockmodels.v2.regular import regular_bm_to_df
            return regular_bm_to_df
    else:
        raise ValueError(f"Unsupported omf version: {__omf_version__}")
