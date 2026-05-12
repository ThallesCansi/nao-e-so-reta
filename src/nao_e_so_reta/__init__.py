"""API pública do projeto Não é só reta."""

# ruff: noqa: E402,F403

import os
import tempfile
from pathlib import Path

_mpl_config_dir = Path(tempfile.gettempdir()) / "nao_e_so_reta_mpl"
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))

from .analysis import *
from .config import *
from .graph_io import *
from .norms import *
from .plots import *
from .projections import *
from .routing import *
from .sampling import *
