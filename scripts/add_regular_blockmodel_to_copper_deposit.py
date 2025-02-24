"""
Add Regular Model to OMF

The example file copper_deposit.omf contains a tensor block model. We will add a new regular block model to the file.
"""

from pathlib import Path

import omf
from omf import RegularBlockModel, TensorGridBlockModel, Project

# %%
# Load
# ----

omf1_path: Path = Path('./../assets/v2/copper_deposit.omf')

project: Project = omf.load(str(omf1_path))

# %%
# Create
# ------

# Create a new regular block model equivalent to the tensor block model

tensor_block_model: TensorGridBlockModel = project.elements[0]


regular_block_model: RegularBlockModel = RegularBlockModel.from_tensor(tensor_block_model)
