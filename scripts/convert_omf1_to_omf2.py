"""
OMF1 to OMF2
============

Convert a OMF1 file to OMF2 using the omf package dev branch

"""
from pathlib import Path

import omf
from omf import Project

# %%
# Convert
# -------

omf1_path: Path = Path('./../assets/v1/copper_deposit.omf')
omf2_path: Path = Path('./../assets/v2/copper_deposit.omf')

project: Project = omf.load(str(omf1_path))
omf.save(project, filename=str(omf2_path))
