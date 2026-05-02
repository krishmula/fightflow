import csv
from collections import Counter
from pathlib import Path

def check_annotations(csv_path):
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found.")
        return

    counts = {} # video_id -> {punch_type: count}
    total = 0
    unique_videos = set()
    all_classes = set()

    try:
        with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 3 or row[0] == 'video_id':
                    continue
                video_id = row[0]
                punch_type = row[2]
                
                if video_id not in counts:
                    counts[video_id] = Counter()
                
                counts[video_id][punch_type] += 1
                unique_videos.add(video_id)
                all_classes.add(punch_type)
                total += 1
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if total == 0:
        print("No annotations found in the CSV.")
        return

    sorted_classes = sorted(list(all_classes))
    
    print("\nAnnotation Summary (Per Video and Class)")
    print("=" * (45 + len(sorted_classes) * 12))
    
    # Header
    header = f"{'Video ID':<35} |"
    for cls in sorted_classes:
        header += f" {cls[:10]:<10} |"
    header += f" {'Total':<6}"
    print(header)
    print("-" * len(header))
    
    # Rows
    for video_id in sorted(counts.keys()):
        row_str = f"{video_id:<35} |"
        video_total = 0
        for cls in sorted_classes:
            count = counts[video_id][cls]
            row_str += f" {count:<10} |"
            video_total += count
        row_str += f" {video_total:<6}"
        print(row_str)
        
    print("=" * len(header))
    print(f"Total Labels: {total}")
    print(f"Unique Videos: {len(unique_videos)}")
    print(f"Classes Found: {', '.join(sorted_classes)}")

if __name__ == "__main__":
    CSV_PATH = "data/annotations.csv"
    check_annotations(CSV_PATH)
