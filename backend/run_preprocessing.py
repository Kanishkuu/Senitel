"""
Run the CERT r4.2 preprocessing pipeline.

This script preprocesses the raw CSV files from CERT r4.2 and converts them
to normalized parquet files that can be used by the streamer or the ML pipeline.

Usage:
    python run_preprocessing.py              # Full preprocessing
    python run_preprocessing.py --sample     # Fast sample mode
    python run_preprocessing.py --check      # Check dataset availability
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "insider-threat-detection"))

import argparse
from src.pipeline.stage1_pipeline import run_stage1, check_dataset
from src.utils.config import PipelineConfig, get_config


def main():
    parser = argparse.ArgumentParser(description="Run CERT r4.2 preprocessing")
    parser.add_argument("--sample", action="store_true", help="Run in sample mode (fast)")
    parser.add_argument("--load-http", action="store_true", help="Include http.csv (14.5 GB)")
    parser.add_argument("--check", action="store_true", help="Check dataset availability")
    parser.add_argument("--config", type=str, default="configs/config.local.yaml",
                        help="Config file path (relative to insider-threat-detection/)")

    args = parser.parse_args()

    # Load config
    config_path = PROJECT_ROOT / "insider-threat-detection" / args.config

    if args.check:
        print("Checking dataset availability...")
        config = PipelineConfig.from_yaml(config_path)
        check_dataset(config)
        return

    # Determine settings
    sample = args.sample
    load_http = args.load_http

    if sample:
        load_http = True  # Sample mode includes HTTP sampling

    print("=" * 60)
    print("CERT r4.2 Preprocessing Pipeline")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Sample mode: {sample}")
    print(f"Load HTTP: {load_http}")
    print("=" * 60)

    # Run pipeline
    config = PipelineConfig.from_yaml(config_path)
    stats = run_stage1(
        config=config,
        load_http=load_http,
        http_sample=sample,
        output_dir=None
    )

    print("\n✓ Preprocessing complete!")
    print(f"Output directory: {config.output.base_dir / config.output.normalized_dir}")


if __name__ == "__main__":
    main()
