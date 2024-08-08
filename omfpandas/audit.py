import getpass
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Literal


@dataclass
class ChangeMessage:
    """Dataclass to mange auditability"""
    element: str
    action: Literal['create', 'update', 'delete']
    description: str
    user: str = field(default_factory=getpass.getuser)
    timestamp: str = field(default_factory=datetime.now)

    def __str__(self):
        return json.dumps(asdict(self), default=str)

