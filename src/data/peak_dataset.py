import os
import av
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class PeakFrameDataset(Dataset):
    def __init__(self, csv_path, video_dir, video_ids, transform=None):
        self.video_dir = video_dir
        self.transform = transform
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['video_id'].isin(video_ids)].reset_index(drop=True)
        self.class_map = {'none': 0, 'jab': 1, 'cross': 2, 'hook': 3, 'uppercut': 4}

    def __len__(self):
        return len(self.df)

    def _extract_frame(self, video_path, frame_index):
        container = av.open(video_path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        
        # calculate approximate target PTS
        fps = float(stream.average_rate or stream.guessed_rate or 30.0)
        time_base = stream.time_base
        seconds = frame_index / fps
        target_pts = int(seconds / float(time_base))
        
        # fast seek backwards to the nearest keyframe
        container.seek(target_pts, any_frame=False, backward=True, stream=stream)
        
        last_frame = None
        for frame in container.decode(video=0):
            if frame.pts is None: continue
            
            t = float(frame.pts * time_base)
            current_idx = int(round(t * fps))
            
            # If we overshot, stop early
            if current_idx > frame_index:
                break
                
            last_frame = frame.to_ndarray(format="rgb24")
            
            # Exact match found, stop
            if current_idx == frame_index:
                break
                
        container.close()
        return last_frame

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = os.path.join(self.video_dir, row['video_id'])
        frame_idx = int(row['frame_index'])
        label = self.class_map.get(row['punch_type'], 0) # default 'none'

        try:
            frame_array = self._extract_frame(video_path, frame_idx)
        except Exception as e:
            print(f"Error reading frame {frame_idx} from {video_path}: {e}")
            frame_array = None

        if frame_array is None:
            # Fallback: a black image if video extraction failed completely
            frame_array = np.zeros((128, 128, 3), dtype=np.uint8)

        image = Image.fromarray(frame_array)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
