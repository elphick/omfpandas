import logging
from abc import ABC
from pathlib import Path
from typing import Optional

import omf
from omf import Project


class OMFPandasBase(ABC):

    def __init__(self, filepath: Path):
        """Instantiate the OMFPandas object.

        Args:
            filepath (Path): Path to the OMF file.

        Raises:
            FileNotFoundError: If the OMF file does not exist.
            ValueError: If the file is not an OMF file.
        """
        self._logger = logging.getLogger(__class__.__name__)
        if not filepath.suffix == '.omf':
            raise ValueError(f'File is not an OMF file: {filepath}')
        self.filepath: Path = filepath
        self.project: Optional[Project] = None
        if filepath.exists():
            self.project = omf.load(str(filepath))
        self._elements = self.project.elements if self.project else []
        self.elements: dict[str, str] = {e.name: e.__class__.__name__ for e in self._elements}

    def get_element_by_name(self, element_name: str):
        """Get an element by its name.

        :param element_name: The name of the element to retrieve.
        :return:
        """
        element = [e for e in self._elements if e.name == element_name]
        if not element:
            raise ValueError(f"Element '{element_name}' not found in the OMF file: {self.filepath.name}. "
                             f"Available elements are: {list(self.elements.keys())}")
        elif len(element) > 1:
            raise ValueError(f"Multiple elements with the name '{element_name}' found in the OMF file: "
                             f"{self.filepath.name}")
        return element[0]
