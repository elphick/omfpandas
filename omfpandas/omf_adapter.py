# omfpandas/omf_adapter.py
from omf import Project as ProjectV1

try:
    from omf_v2 import Project as ProjectV2
except ImportError:
    ProjectV2 = None


class OMFAdapter:
    def __init__(self, version):
        if version == 'v1':
            self.project = ProjectV1()
        elif version == 'v2':
            if ProjectV2 is None:
                raise ValueError("OMF v2 is not supported in this environment, "
                                 "please install the required dependencies")
            self.project = ProjectV2()
        else:
            raise ValueError("Unsupported OMF version")

    def load_project(self, filepath):
        return self.project.load(filepath)

    def save_project(self, project, filepath):
        return self.project.save(project, filepath)
