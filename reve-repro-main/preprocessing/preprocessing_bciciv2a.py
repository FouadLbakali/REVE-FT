"""
adapted from https://github.com/wjq-learning/CBraMod
"""

import os
import pickle

import lmdb
import mne
from scipy.signal import butter

from preprocessing.paths import PATHS


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype="band")


files_dict = {
    "train": [
        "A01E.gdf",
        "A01T.gdf",
        "A02E.gdf",
        "A02T.gdf",
        "A03E.gdf",
        "A03T.gdf",
        "A04E.gdf",
        "A04T.gdf",
        "A05E.gdf",
        "A05T.gdf",
    ],
    "val": ["A06E.gdf", "A06T.gdf", "A07E.gdf", "A07T.gdf"],
    "test": ["A08E.gdf", "A08T.gdf", "A09E.gdf", "A09T.gdf"],
}

classes = {
    769: 0,  # left hand
    770: 1,  # right hand
    771: 2,  # feet
    772: 3,  # tongue
}

dataset = {"train": [], "val": [], "test": []}
root_dir = PATHS["bciciv2a"]["raw"]
os.makedirs(PATHS["bciciv2a"]["processed"], exist_ok=True)
db = lmdb.open(PATHS["bciciv2a"]["processed"], map_size=1024**3)  # 1 GB

for split, files_list in files_dict.items():
    for file in files_list:
        print(f"Processing {file}...")
        raw = mne.io.read_raw_gdf(os.path.join(root_dir, file), preload=True)
        raw.pick_types(eeg=True)

        events, event_id = mne.events_from_annotations(raw)
        print(f"Events found: {events.shape[0]}")
        print(f"Event IDs: {event_id}")
        continue

        raw_data = raw.get_data()


txn = db.begin(write=True)
txn.put(key="__keys__".encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()
