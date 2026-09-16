# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""Pytest bootstrap: expose the plugin package to test collection.

Most tests load modules via ``importlib.util.spec_from_file_location`` and do
not need this, but any test that imports ``vllm_external_executor`` directly
(e.g. ``test_storage_checkpoint_engine.py``) requires the package root on
``sys.path`` even when the plugin is not pip-installed.
"""

import sys
from pathlib import Path

_PKG_ROOT = str(Path(__file__).resolve().parent)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
