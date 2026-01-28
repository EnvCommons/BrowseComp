"""
Path constants for BrowseComp environment.

Handles both production (/orwd_data) and local development paths.
"""

from pathlib import Path
import os


# Check if /orwd_data exists (production), otherwise use local directory (dev)
if os.path.exists("/orwd_data"):
    ENV_PATH = Path("/orwd_data")
else:
    ENV_PATH = Path(__file__).parent


# CSV file locations with fallback logic
BROWSECOMP_CSV_PROD = ENV_PATH / "browsecomp" / "browse_comp_test_set.csv"
BROWSECOMP_CSV_LOCAL = Path(__file__).parent / "browse_comp_test_set.csv"

# Try production path first, fall back to local
if BROWSECOMP_CSV_PROD.exists():
    BROWSECOMP_CSV = BROWSECOMP_CSV_PROD
else:
    BROWSECOMP_CSV = BROWSECOMP_CSV_LOCAL
