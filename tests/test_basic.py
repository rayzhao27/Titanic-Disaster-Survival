"""
Basic tests to ensure CI pipeline works
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that basic imports work"""
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        assert True
    except ImportError as e:
        assert False, f"Import failed: {e}"

def test_config_exists():
    """Test that config file exists"""
    config_path = Path(__file__).parent.parent / "config" / "config.py"
    assert config_path.exists(), "Config file should exist"

def test_main_exists():
    """Test that main.py exists"""
    main_path = Path(__file__).parent.parent / "main.py"
    assert main_path.exists(), "main.py should exist"

def test_api_exists():
    """Test that API file exists"""
    api_path = Path(__file__).parent.parent / "src" / "api" / "app.py"
    assert api_path.exists(), "API file should exist"

def test_models_directory():
    """Test that models directory exists"""
    models_path = Path(__file__).parent.parent / "models"
    assert models_path.exists(), "Models directory should exist"