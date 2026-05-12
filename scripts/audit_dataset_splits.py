
import pandas as pd
import sys
import yaml
from pathlib import Path

def analyze_manifest(file_path):
    path = Path(file_path)
    if not path.exists():
        # Only print error if explicitly requested
        return False

    print(f"\n{'='*60}")
    print(f"AUDITING DATASET SPLIT: {path.name}")
    print(f"{'='*60}")

    df = pd.read_csv(path)
    
    total_samples = len(df)
    total_vids = df['video_id'].nunique()
    
    print(f"Total Samples: {total_samples}")
    print(f"Total Videos:  {total_vids}")
    
    if 'split' not in df.columns:
        print("Error: 'split' column not found in manifest.")
        return True

    splits = sorted(df['split'].dropna().unique())
    
    for split in splits:
        split_df = df[df['split'] == split]
        split_count = len(split_df)
        split_vids = split_df['video_id'].nunique()
        split_pct = (split_count / total_samples) * 100
        
        print(f"\n--- {split.upper()} SPLIT ---")
        print(f"  Samples: {split_count} ({split_pct:.1f}%)")
        print(f"  Videos:  {split_vids}")
        
        # Class distribution
        if 'punch_type' in df.columns:
            counts = split_df['punch_type'].value_counts()
            print("  Class Distribution:")
            for punch, count in counts.items():
                print(f"    - {punch:10}: {count}")
        
        # Video list
        vids = sorted(split_df['video_id'].unique().tolist())
        print(f"  Videos: {', '.join(vids)}")

    print(f"\n{'='*60}\n")
    return True

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "src" / "models" / "base" / "data_config.yaml"
    
    manifests_to_check = []
    
    # Load config to get manifest names
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            data_cfg = cfg.get("data", {})
            # Get filenames from config
            manifests_to_check.extend([
                data_cfg.get("manifest_file"),
                data_cfg.get("clip_manifest_file"),
                data_cfg.get("pose_manifest_file")
            ])
    
    # Clean up None values and convert to Paths
    manifest_paths = [project_root / m for m in manifests_to_check if m]

    if len(sys.argv) > 1:
        # If user passed a specific file, check only that
        analyze_manifest(sys.argv[1])
    else:
        found = False
        for m in manifest_paths:
            if analyze_manifest(m):
                found = True
        
        if not found:
            print("No manifests found based on data_config.yaml. Use: python scripts/audit_dataset_splits.py <file.csv>")
