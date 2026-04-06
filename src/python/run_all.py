"""
Master orchestrator: runs the full data preparation pipeline.

Usage:
    cd src/python
    python run_all.py
"""

import sys
from pathlib import Path

# Ensure src/python is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prep_monthly_gdp import main as prep_gdp
from prep_narrative import main as prep_narrative
from prep_longterm import main as prep_longterm
from prep_hfi import main as prep_hfi


def main():
    print("\n" + "=" * 70)
    print("  FULL DATA PIPELINE")
    print("=" * 70)

    # Step 1: Monthly GDP (produces gdp_monthly_df.csv)
    prep_gdp()

    # Step 2: Narrative shocks (produces romer_china_data.csv)
    prep_narrative()

    # Step 3: Long-term projection data (produces china_longterm_data.csv)
    prep_longterm()

    # Step 4: HFI shocks (produces hfi_core_data.csv)
    prep_hfi()

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
