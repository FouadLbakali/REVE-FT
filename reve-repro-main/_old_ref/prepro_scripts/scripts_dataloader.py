import ast
import random
from os.path import join as pjoin

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset, Sampler
import pickle
import gc

def calculate_bs(key, smallest_bs=128):
    C = 16 * smallest_bs  # Calculate the constant based on smallest_bs
    bs = max(1, int(C // key))  # Ensure batch size is at least 1
    return bs

class GroupedSampler(Sampler):
    def __init__(self, dataset, batch_size=32, drop_last=False, n_gpu=3):
        self.dataset = dataset
        self.segs = self.dataset.segs
        self.groups = self.dataset.groups
        self.bs = batch_size
        self.drop_last = 0 if drop_last else 1
        self.n_gpu = n_gpu
        self._bs_ = batch_size

    def __iter__(self):
        indices = []
        keys = list(self.groups.keys())
        keys = [k for k in keys if k > 6]
        for key in keys:
            self.bs =  calculate_bs(key, smallest_bs=self._bs_)
            group = self.groups[key]
            group_indices = sorted(group)
            random.shuffle(group_indices)
            indices += [
                group_indices[i * self.bs : (i + 1) * self.bs]
                for i in range(len(group_indices) // self.bs + self.drop_last)
            ]
        random.shuffle(indices)
        to_remove = len(indices) % self.n_gpu
        self.indices = indices if to_remove == 0 else indices[:-to_remove]
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class ValGroupedSampler(Sampler):
    def __init__(self, dataset, batch_size=32, drop_last=False, n_gpu=3):
        self.dataset = dataset
        self.segs = self.dataset.segs
        self.groups = self.dataset.groups
        self.bs = batch_size
        self.drop_last = 0 if drop_last else 1
        self.n_gpu = n_gpu

    def __iter__(self):
        indices = []
        for group in self.groups.values():
            group_indices = group
            indices += [
                group_indices[i * self.bs : (i + 1) * self.bs] for i in range(len(group) // self.bs + self.drop_last)
            ]
        to_remove = len(indices) % self.n_gpu
        self.indices = indices if to_remove == 0 else indices[:-to_remove]
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class EEGDataset(Dataset):
    def __init__(
        self,
        segs,
        groups,
        df_big,
        df_stats,
        path,
        clip,
        block_masking=True,
        masking_ratio=0.5,
        radius_spat_mask=0.03,
        radius_temp_mask=2,
        dropout_ratio=0.1,
        dropout_ratio_radius=0.05,
        no_masking=False,
        manual_seed=False,
    ):
        self.path = path
        self.path_pos = "/".join(self.path.split("/")[:-1] + ["positions"])
        self.path_stats = "/".join(self.path.split("/")[:-1] + ["stats"])
        self.segs = segs
        self.groups = groups
        self.clip = clip
        self.block_masking = block_masking
        self.masking_ratio = masking_ratio
        self.no_masking = no_masking
        if self.block_masking:
            self.radius_spat_mask = radius_spat_mask
            self.radius_temp_mask = radius_temp_mask
            self.dropout_ratio = dropout_ratio
            self.dropout_ratio_radius = dropout_ratio_radius
        else:
            self.manual_seed = manual_seed
        
        self.df_big = df_big
        self.df_stats = df_stats
        self.init_files_pos(self.df_big ,self.df_stats,self.path)
        self.counter = 0
    
    def init_files_pos(self,df_big,df_stats,path):
        self.files = {}
        self.positions_cache = {}
        for r_i, r_t, r_c in zip(
            df_big["big_recording_index"].values, df_big["duration"].values, df_big["n_chans"].values
        ):
            memmap_array = np.memmap(
                pjoin(path, "recording_-_eeg_-_" + str(r_i) + ".npy"), mode="r", shape=(r_t, r_c), dtype="float32"
            )
            memmap_array[:]
            self.files[r_i] = memmap_array
            self.positions_cache[r_i] = np.load(pjoin(self.path_pos, f"recording_-_positions_-_{r_i}.npy"))
        self.stats = {}
        for r_i, n_s, n_c in zip(
            df_stats["big_recording_index"].values, df_stats["n_sessions"].values, df_stats["n_chans"].values
        ):
            memmap_stats = np.memmap(
                pjoin(self.path_stats, "recording_-_stats_-_" + str(r_i) + ".npy"),
                mode="r",
                shape=(n_s, 2, n_c),
                dtype="float32",
            )
            memmap_stats[:]
            self.stats[r_i] = memmap_stats
            
    def del_files(self):
        for k,v in self.files.items():
            v._mmap.close()
            del v
        for k,v in self.stats.items():
            v._mmap.close()
            del v
        del self.files, self.positions_cache, self.stats
        gc.collect()
        
    def __getitem__(self, index):
        if self.counter>=500000:
            self.del_files()
            self.init_files_pos(self.df_big,self.df_stats,self.path)
            self.counter = 0
        self.counter +=1
        splitted_index = index.split("_-_")
        if len(splitted_index) == 3:
            b_rec, rec, offset = index.split("_-_")
            bad_chans = False
        else:
            b_rec, rec, offset, bad_chans_ = index.split("_-_")
            bad_chans = [int(x) for x in bad_chans_.split("/")]
        b_rec = int(b_rec)
        rec = int(rec)
        offset = int(offset)
        positions = self.positions_cache[b_rec].copy()
        stats = self.stats[b_rec][rec]
        n_chans = positions.shape[0]
        eeg = self.files[b_rec][offset : offset + 2000].copy()
        #del holder
        if bad_chans != False:
            mask = np.ones(positions.shape[0], dtype=bool)
            mask[bad_chans] = False
            eeg = eeg[:, mask]
            positions = positions[mask]
            stats = stats[:, mask]

        eps = 1e-10 if any(stats[1] == 0) else 0
        eeg -= stats[0]
        eeg /= stats[1] + eps
        eeg = torch.from_numpy(eeg)
        eeg = eeg.transpose(0, 1)
        if self.no_masking:
            return eeg.float().clip(-self.clip, self.clip), torch.from_numpy(positions).float()
        elif self.block_masking:
            patches = eeg.unfold(dimension=1, size=200, step=200 - 20)  # achanger ici
            c, h, p = patches.shape
            batch_mask, batch_unmask = optimize_masking_process(
                c,
                self.masking_ratio,
                self.radius_spat_mask,
                self.radius_temp_mask,
                h,
                positions,
                self.dropout_ratio,
                self.dropout_ratio_radius,
            )
            return (
                eeg.float().clip(-self.clip, self.clip),
                torch.from_numpy(positions).float(),
                batch_mask,
                batch_unmask,
            )
        else:
            patches = eeg.unfold(dimension=1, size=200, step=200 - 20)
            c, h, p = patches.shape
            num_patches = c * h
            num_masked = int(self.masking_ratio * num_patches)
            if self.manual_seed:
                torch.manual_seed(42)  # for validation set on same patch if val is done in MAE
            rand_indices = torch.rand(1, num_patches).argsort(dim=-1)
            batch_mask, batch_unmask = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
            batch_mask = batch_mask.squeeze(0)
            batch_unmask = batch_unmask.squeeze(0)
            return (
                eeg.float().clip(-self.clip, self.clip),
                torch.from_numpy(positions).float(),
                batch_mask,
                batch_unmask,
            )

    def __len__(
        self,
    ):
        return len(self.segs)


def block_masking(C, masking_ratio, radius, preselected_masked_indices=None):
    n_points = C.shape[0]
    n_masked_points = int(masking_ratio * n_points)
    mask = np.zeros(n_points, dtype=bool)
    if preselected_masked_indices is not None:
        if len(preselected_masked_indices) > 0:
            mask[preselected_masked_indices] = True
    # Create a KD-tree for efficient neighbor searching
    kdtree = cKDTree(C)
    while np.sum(mask) < n_masked_points:
        unmasked_indices = np.where(~mask)[0]
        if len(unmasked_indices) == 0:
            break
        random_index = np.random.choice(unmasked_indices)
        random_point = C[random_index]
        indices_within_radius = kdtree.query_ball_point(random_point, radius)
        mask[indices_within_radius] = True
    masked_indices = np.where(mask)[0]
    return masked_indices


def optimize_masking_process(
    n_chans,
    masking_ratio,
    radius_spat_mask,
    radius_temp_mask,
    num_patches,
    pos,
    dropout_ratio=0.1,
    dropout_ratio_radius=0.05,
):
    num_block_patches = num_patches // radius_temp_mask
    num_missing_patches = num_patches - num_block_patches * radius_temp_mask
    num_masked_chans = int(masking_ratio * n_chans)
    all_indices = np.arange(n_chans)
    if dropout_ratio > 0:
        dropout_channels = block_masking(pos, dropout_ratio, dropout_ratio_radius)[: int(dropout_ratio * n_chans)]
    else:
        dropout_channels = []
    block_masks = [
        block_masking(pos, masking_ratio, radius_spat_mask, dropout_channels)[:num_masked_chans]
        for _ in range(num_block_patches)
    ]
    idx_block = np.repeat(np.array(block_masks)[:, np.newaxis, :], radius_temp_mask, axis=1).reshape(
        -1, num_masked_chans
    )

    missing_masks = [
        block_masking(pos, masking_ratio, radius_spat_mask)[:num_masked_chans] for _ in range(num_missing_patches)
    ]
    idx_missing = np.array(missing_masks)
    mask = np.concatenate([idx_block, idx_missing], axis=0)
    # Efficient unmasking
    unmask = np.array([np.setdiff1d(all_indices, masked_indices, assume_unique=True) for masked_indices in mask])
    row_indices = np.arange(mask.shape[0])[:, np.newaxis]
    masked_indices = (n_chans * row_indices + mask).ravel()
    unmasked_indices = (n_chans * row_indices + unmask).ravel()
    np.random.shuffle(masked_indices)
    np.random.shuffle(unmasked_indices)
    batch_mask = torch.from_numpy(masked_indices)
    batch_unmask = torch.from_numpy(unmasked_indices)
    return batch_mask, batch_unmask


def compute_groups_segs(df_, df_big_, duration_min):
    # add filter on duration maybe to avoid it to break.
    df_big = df_big_.copy()
    df = df_.copy()
    dict_groups = {}
    groups_agg = df_big.groupby("n_chans").agg({"big_recording_index": list}).reset_index()
    for c in groups_agg["n_chans"].values.tolist():
        dict_groups[c] = []

    dict_groups_incorrect = {}
    for c in dict_groups.keys():
        windows = []
        for big_i in groups_agg[groups_agg["n_chans"] == c].big_recording_index.values[0]:
            df_tmp = df[df["big_recording_index"] == big_i].copy()
            last_two = df_tmp.iloc[-2:]["flag_reduce"].values
            if last_two[0] != last_two[1]:
                df_tmp.loc[df_tmp.index[-1], "flag_remove"] = True
            correct_start = np.concatenate([np.array([0]), np.cumsum(df_tmp["duration"].tolist())[:-1]])
            df_tmp["start"] = correct_start
            df_tmp["end"] = df_tmp["start"] + df_tmp["duration"]
            df_tmp = df_tmp[df_tmp["duration"] > duration_min]
            for i in range(len(df_tmp)):
                row = df_tmp.iloc[i]
                if not row.flag_remove and row.n_chans_to_remove == 0:
                    start, end, duration = row.start, row.end, row.duration
                    windows_index = np.arange(start, end - duration_min, duration_min)
                    windows += [str(big_i) + "_-_" + str(i) + "_-_" + str(w) for w in windows_index]
                elif not row.flag_remove and row.n_chans_to_remove > 0:
                    start, end, duration = row.start, row.end, row.duration
                    n_chans_to_remove = row.n_chans_to_remove
                    chans_to_remove = "/".join([str(x) for x in ast.literal_eval(str(row.flag_reduce))])
                    windows_index = np.arange(start, end - duration_min, duration_min)
                    windows_incorrect = [
                        str(big_i) + "_-_" + str(i) + "_-_" + str(w) + "_-_" + chans_to_remove for w in windows_index
                    ]
                    corrected_num_chans = c - n_chans_to_remove
                    if corrected_num_chans not in dict_groups_incorrect.keys():
                        dict_groups_incorrect[corrected_num_chans] = []
                    dict_groups_incorrect[corrected_num_chans] += windows_incorrect
        dict_groups[c] += windows

    for c in dict_groups_incorrect.keys():
        if c not in dict_groups.keys():
            dict_groups[c] = []
        dict_groups[c] += dict_groups_incorrect[c]
    segs = []
    for c in dict_groups.keys():
        segs += dict_groups[c]
    dict_groups = dict(sorted(dict_groups.items(), key=lambda item: int(item[0])))

    return dict_groups, segs, df_big


def return_train_val_loaders(args, return_val=True):
    path_recordings = pjoin(args.data_path, "recordings")
    path_csv = pjoin(args.data_path, "csv_recordings")
    df_corrected = pd.read_csv(pjoin(path_csv, "df_corrected.csv"))
    df_big = pd.read_csv(pjoin(path_csv, "df_big.csv"))
    df_stats = pd.read_csv(pjoin(path_csv, "df_stats_tmp.csv"))
    duration_min = 2000
    val_set = [78, 83, 84, 93, 97]  ##### Specfici to NAS ####
    if args.load_subset == "nas":
        train_set = [72, 73, 74, 75, 77, 79, 80, 81, 82, 85, 86, 87, 88, 90, 92, 98, 99, 100, 101, 102, 103, 104]
    elif args.load_subset == "subset":
        train_set = [23,36,59,74,75,79,80,81,82,85,86,87,88,90,92,98,99,100,101,102,103,104,180,198,233,337,338,339,364,365,366,371,372,373,374,375,376,377]
    elif args.load_subset == "all":
        train_set = [x for x in df_big.big_recording_index.unique()]
        #bad_guys = [419, 427, 437, 393, 76, 78, 93, 354]  # 78,93 just to validate i guess
        #train_set = [x for x in train_set if x not in bad_guys]
        #if args.remove_toxic:
        #    train_set = [x for x in train_set if x < 380]
        assert len(train_set) > 200
    df_train = df_corrected[df_corrected["big_recording_index"].isin(train_set)].copy()
    df_big_train = df_big[df_big["big_recording_index"].isin(train_set)].copy()
    df_stats_train = df_stats[df_stats["big_recording_index"].isin(train_set)].copy()
    train_loader, len_train, len_train_sampler = return_loader(
        args, df_train, df_big_train, df_stats_train, path_recordings, duration_min, True
    )
    if not return_val:
        return train_loader, len_train, len_train_sampler
    else:
        df_val = df_corrected[df_corrected["big_recording_index"].isin(val_set)].copy()
        df_big_val = df_big[df_big["big_recording_index"].isin(val_set)].copy()
        df_stats_val = df_stats[df_stats["big_recording_index"].isin(val_set)].copy()
        val_loader, len_val, len_val_sampler = return_loader(
            args, df_val, df_big_val, df_stats_val, path_recordings, duration_min, False
        )
        return train_loader, val_loader, len_train, len_val, len_train_sampler, len_val_sampler


def return_loader(args, df, df_big, df_stats, path_recordings, duration_min, train):
    if train and args.remove_toxic:
        with open(pjoin(args.data_path,'data_tuple'+str(args.remove_toxic_number)+'.pkl'), 'rb') as f:
            dict_groups, segs = pickle.load(f)
        dict_groups = {k:v for k,v in dict_groups.items() if len(v)>0}
        df_big = pd.read_csv(pjoin(args.data_path,'df_big_no_toxic'+str(args.remove_toxic_number)+'.csv'))
    else:
        dict_groups, segs, df_big = compute_groups_segs(df, df_big, duration_min)
    if train:
        dataset = EEGDataset(
            segs,
            dict_groups,
            df_big,
            df_stats,
            path_recordings,
            clip=args.clip,
            block_masking=args.block_masking,
            masking_ratio=args.masking_ratio,
            radius_spat_mask=0.03,
            radius_temp_mask=3,
            dropout_ratio=0.1,
            dropout_ratio_radius=0.04,
            no_masking=False,
        )
        sampler = GroupedSampler(dataset, batch_size=args.batch_size, drop_last=True, n_gpu=args.n_gpus*args.n_nodes)
    else:
        dataset = EEGDataset(
            segs,
            dict_groups,
            df_big,
            df_stats,
            path_recordings,
            args.clip,
            block_masking=False,
            masking_ratio=0.5,
            no_masking=False,
            manual_seed=True,
        )
        sampler = ValGroupedSampler(dataset, batch_size=args.batch_size, drop_last=True, n_gpu=args.n_gpus)
    sampler.__iter__()  # can be done only in main if we want (for multi process)
    len_sampler = len(sampler)
    loader = DataLoader(
        dataset,
        pin_memory=True,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        persistent_workers=False,
        prefetch_factor=args.prefetch_factor,
    )
    return loader, len(dataset), len_sampler
