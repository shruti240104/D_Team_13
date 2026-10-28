"""Optional smoke test that starts the REAL Streamlit app headlessly (Member 4).

Skipped automatically when Streamlit is not installed.  It only checks that the
script runs without an exception in its initial (no upload) state.
"""
from pathlib import Path

import pytest


def test_app_starts_without_exception():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(app_path, default_timeout=60).run()
    assert not at.exception
