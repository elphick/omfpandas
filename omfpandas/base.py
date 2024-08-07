import io
import json
import logging
import tempfile
import webbrowser
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
        self.element_attributes: dict[str, list[str]] = {e.name: [a.name for a in e.attributes] for e in self._elements}

    def __repr__(self):
        return f"OMFPandasBase({self.filepath})"

    def __str__(self):
        return f"OMFPandasBase object for OMF file: {self.filepath}"

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

    def get_element_attribute_names(self, element_name: str) -> list[str]:
        """Get the attribute names of an element.

        :param element_name: The name of the element to retrieve.
        :return:
        """
        element = self.get_element_by_name(element_name)
        return [attr.name for attr in element.attributes]

    def view_block_model_profile(self, blockmodel_name: str, query: Optional[str] = None):
        """View the profile of a BlockModel in the default web browser.

        Args:
            blockmodel_name (str): The name of the BlockModel to profile.
            query (str): A query defining the subset of the BlockModel.
        """

        el = self.get_element_by_name(blockmodel_name)
        filter_key: str = query if query else 'no_filter'

        if el.metadata.get('profile') is None:
            raise ValueError(f"BlockModel '{blockmodel_name}' has not been profiled.  "
                             f"Please run 'profile_blockmodel' first.")

        # JSON string containing the profile report
        profile_html: str = el.metadata['profile'][filter_key]

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
            temp_file.write(profile_html.encode('utf-8'))
            temp_file_path = temp_file.name

        # Open the temporary file in the default web browser
        webbrowser.open(f"file://{temp_file_path}")
