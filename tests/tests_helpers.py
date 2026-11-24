"""Small utilities shared by the tests."""

import pytest
from conftest import ref_paths


def ref_paths_or_skip():
    p = ref_paths()
    if not p:
        pytest.skip("No reference photo")
    return p
