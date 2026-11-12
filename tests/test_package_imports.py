"""Catch broken cross-module imports, including less frequently used data APIs."""
import importlib
from pathlib import Path

import pytest
import vdc

PACKAGE = Path(vdc.__file__).resolve().parent
MODULES = sorted("vdc." + ".".join(p.relative_to(PACKAGE).with_suffix("").parts)
                 for p in PACKAGE.rglob("*.py") if p.name != "__init__.py")


@pytest.mark.parametrize("module", MODULES)
def test_package_module_imports(module):
    if module == "vdc.eval.visualize":
        pytest.importorskip("seaborn", reason="visualization extra is optional")
    importlib.import_module(module)
