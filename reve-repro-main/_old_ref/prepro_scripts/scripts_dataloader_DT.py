import random
from os.path import join as pjoin

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


# seed = 42  # Replace with your desired seed


def shuffle_by_blocks(lst, block_size):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)
    shuffled_list = [item for block in blocks for item in block]
    return shuffled_list


def shuffle_order(indices, modulo):
    indices_final = []
    for i in range(modulo):
        indices_final.extend([indices[j] for j in range(i, len(indices) - (len(indices) % modulo), modulo)])
    return indices_final


class GroupedSampler(Sampler):
    def __init__(self, dataset, batch_size=32, drop_last=False, n_gpu=3):
        self.dataset = dataset
        self.segs = self.dataset.segs
        self.groups = self.dataset.groups
        self.bs = batch_size
        self.drop_last = 0 if drop_last else 1
        self.n_gpu = n_gpu
        self._bs_ = batch_size
        self.indices = []

    def __iter__(self):
        indices = []
        grps_end = {}
        keys = list(self.groups.keys())
        # random.shuffle(keys)
        for key in keys:
            self.bs = self._bs_ // 4 if key >= 120 else self._bs_ // 2 if key >= 64 else self._bs_
            group = self.groups[key]
            group_indices = sorted(group)
            random.shuffle(group_indices)
            end_number = len(group) % (self.bs * self.n_gpu)
            # grps_end[key] = group_indices[-end_number:]
            # group_indices = group_indices[:-end_number]
            # if group_indices != []:
            #    indices += [group_indices[i*self.bs:(i+1)*self.bs] for i in range(len(group_indices)//self.bs+self.drop_last)]
            indices += [
                group_indices[i * self.bs : (i + 1) * self.bs]
                for i in range(len(group_indices) // self.bs + self.drop_last)
            ]
        # indices = shuffle_by_blocks(indices,self.n_gpu)
        # indices = shuffle_order(indices,self.n_gpu)
        # indices = shuffle_by_blocks(indices,self.n_gpu)
        random.shuffle(indices)

        # keys_ = sorted(list(grps_end.keys()))
        # for key in keys_[::-1]:
        #     self.bs = self._bs_//4 if key >= 120 else self._bs_//2 if key >= 64 else self._bs_
        #     indices += [grps_end[key][i*self.bs:(i+1)*self.bs] for i in range(len(grps_end[key])//self.bs+self.drop_last)]

        to_remove = len(indices) % self.n_gpu
        if to_remove != 0:
            self.indices = indices[:-to_remove]
        else:
            self.indices = indices
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
            # if not self.random:
            #    random.seed(42)
            # group_indices = sorted(group)
            group_indices = group
            # random.shuffle(group_indices)
            indices += [
                group_indices[i * self.bs : (i + 1) * self.bs] for i in range(len(group) // self.bs + self.drop_last)
            ]
        # random.shuffle(indices)
        to_remove = len(indices) % self.n_gpu
        if to_remove != 0:
            self.indices = indices[:-to_remove]
        else:
            self.indices = indices
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class EEGDataset(Dataset):
    def __init__(
        self,
        segs,
        groups,
        df_big,
        df_stats,
        path,
        clip,
        masking=0,
        random_masking=False,
        radius_spat_mask=0.03,
        radius_temp_mask=2,
        dropout_ratio=0.1,
        dropout_ratio_radius=0.05,
    ):  # ,dict_eeg):
        self.path = path
        self.path_pos = "/".join(self.path.split("/")[:-1] + ["positions"])
        self.path_stats = "/".join(self.path.split("/")[:-1] + ["stats"])
        self.segs = segs
        self.groups = groups
        self.clip = clip
        if masking > 0:
            self.masking = True
            self.masking_ratio = masking
            self.radius_spat_mask = radius_spat_mask
            self.radius_temp_mask = radius_temp_mask
            self.dropout_ratio = dropout_ratio
            self.dropout_ratio_radius = dropout_ratio_radius
        else:
            self.masking = False  # achanger ici et le true la haute
        if random_masking:
            self.random_masking = True
            self.masking_ratio = masking
        else:
            self.random_masking = False
        self.files = {}
        for r_i, r_t, r_c in zip(
            df_big["big_recording_index"].values, df_big["duration"].values, df_big["n_chans"].values
        ):
            memmap_array = np.memmap(
                pjoin(path, "recording_-_eeg_-_" + str(r_i) + ".npy"), mode="r", shape=(r_t, r_c), dtype="float32"
            )
            memmap_array[:]
            self.files[r_i] = memmap_array

        self.stats = {}
        for r_i, n_s, n_c in zip(
            df_stats["big_recording_index"].values, df_stats["n_sessions"].values, df_stats["n_chans"].values
        ):
            # memmap_stats = np.memmap(pjoin(path, 'recording_-_eeg_-_'+str(r_i)+'.npy'), mode='r', shape=(n_s,2, n_c), dtype='float32')
            memmap_stats = np.memmap(
                pjoin(self.path_stats, "recording_-_stats_-_" + str(r_i) + ".npy"),
                mode="r",
                shape=(n_s, 2, n_c),
                dtype="float32",
            )
            memmap_stats[:]
            self.stats[r_i] = memmap_stats

    def __getitem__(self, index):
        b_rec, rec, offset = index.split("_-_")
        b_rec = int(b_rec)
        rec = int(rec)
        offset = int(offset)
        eeg = self.files[b_rec][offset : offset + 2000].copy()
        positions = np.load(pjoin(self.path_pos, "recording_-_positions_-_" + str(b_rec) + ".npy"))
        eeg = torch.from_numpy(eeg)
        stats = self.stats[b_rec][rec]
        # bad_chan = np.where(stats.min(axis=0) == stats.max(axis=0))[0]
        # chans_to_keep = [int(x) for x in np.arange(eeg.shape[1]) if x not in bad_chan]
        eeg = (eeg - stats[0]) / (stats[1] + 1e-10)
        eeg = eeg.transpose(0, 1)
        # eeg = eeg[chans_to_keep]
        # positions = positions[chans_to_keep]
        if self.masking and not self.random_masking:
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
        elif self.random_masking:
            patches = eeg.unfold(dimension=1, size=200, step=200 - 20)
            c, h, p = patches.shape
            num_patches = c * h
            num_masked = int(self.masking_ratio * num_patches)
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
        else:
            return eeg.float().clip(-self.clip, self.clip), torch.from_numpy(positions).float()

    def __len__(
        self,
    ):
        return len(self.segs)


import torch
from scipy.spatial import cKDTree


# def block_masking(C, masking_ratio, radius):
#     n_points = C.shape[0]
#     n_masked_points = int(masking_ratio * n_points)

#     mask = np.zeros(n_points, dtype=bool)
#     kdtree = cKDTree(C)

#     while np.sum(mask) < n_masked_points:
#         unmasked_indices = np.where(~mask)[0]
#         if len(unmasked_indices) == 0:
#             break
#         random_index = np.random.choice(unmasked_indices)
#         random_point = C[random_index]

#         indices_within_radius = kdtree.query_ball_point(random_point, radius)
#         mask[indices_within_radius] = True

#     masked_indices = np.where(mask)[0]
#     return masked_indices


def block_masking(C, masking_ratio, radius, preselected_masked_indices=None):
    n_points = C.shape[0]
    n_masked_points = int(masking_ratio * n_points)

    # Initialize the mask (False for unmasked, True for masked)
    mask = np.zeros(n_points, dtype=bool)

    # Apply preselected masked indices
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

    # Precompute common operations
    num_masked_chans = int(masking_ratio * n_chans)
    all_indices = np.arange(n_chans)

    if dropout_ratio > 0:
        dropout_channels = block_masking(pos, dropout_ratio, dropout_ratio_radius)[: int(dropout_ratio * n_chans)]
    else:
        dropout_channels = []

    # Precompute the block and missing masks
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

    # Vectorized operations for masked and unmasked indices
    masked_indices = (n_chans * row_indices + mask).ravel()
    unmasked_indices = (n_chans * row_indices + unmask).ravel()

    # Shuffle only at the end
    np.random.shuffle(masked_indices)
    np.random.shuffle(unmasked_indices)

    # Convert to tensors in one step
    batch_mask = torch.from_numpy(masked_indices)
    batch_unmask = torch.from_numpy(unmasked_indices)

    return batch_mask, batch_unmask


def compute_groups_segs(df_, df_big_, duration_min, subset=[0, 72, 73]):
    # AJOUTER UN FILTRE SUR LA DURATION SANS QUE CA CASSE MAYBE
    df_big = df_big_.copy()
    df = df_.copy()
    if subset != None:
        df_big = df_big[df_big["big_recording_index"].isin(subset)]
    dict_groups = {}
    # df_big = df_big.sort_values('n_chans').reset_index(drop=True)
    groups_agg = df_big.groupby("n_chans").agg({"big_recording_index": list}).reset_index()
    for c in groups_agg["n_chans"].values.tolist():
        dict_groups[c] = []

    for c in dict_groups.keys():
        windows = []
        for big_i in groups_agg[groups_agg["n_chans"] == c].big_recording_index.values[0]:
            # print(big_i,end='')
            df_tmp = df[df["big_recording_index"] == big_i].copy()
            correct_start = np.concatenate([np.array([0]), np.cumsum(df_tmp["duration"].tolist())[:-1]])
            df_tmp["start"] = correct_start
            df_tmp["end"] = df_tmp["start"] + df_tmp["duration"]
            df_tmp = df_tmp[df_tmp["duration"] > duration_min]
            for i in range(len(df_tmp)):
                row = df_tmp.iloc[i]
                start, end, duration = row.start, row.end, row.duration
                windows_index = np.arange(start, end - duration_min, duration_min)
                windows += [str(big_i) + "_-_" + str(i) + "_-_" + str(w) for w in windows_index]
        dict_groups[c] += windows

    segs = []
    for c in dict_groups.keys():
        segs += dict_groups[c]

    # print(len(segs))
    # print([len(k) for k in dict_groups.keys()])

    # dict_groups = dict(sorted(dict_groups.items(), key=lambda item: len(str(item[0]))))
    # print([len(k) for k in dict_groups.keys()])
    dict_groups = dict(sorted(dict_groups.items(), key=lambda item: int(item[0])))

    return dict_groups, segs, df_big


def return_loaders_eegnas(args):
    path_recordings = pjoin(args.data_path, "recordings")
    path_csv = pjoin(args.data_path, "csv_recordings")
    df = pd.read_csv(pjoin(path_csv, "df.csv"))
    df_big = pd.read_csv(pjoin(path_csv, "df_big.csv"))
    df_stats = pd.read_csv(pjoin(path_csv, "df_stats_tmp.csv"))
    duration_min = 2000
    ##### Specfici to NAS ####
    val_set = [78, 83, 84, 93, 97]  # [83,87,97,100]
    # train_set = [x for x in np.arange(72,105) if x not in val_set+[76]]
    train_set = [x for x in df.big_recording_index.unique() if x not in val_set + [76]]
    train_set = [x for x in train_set if x < 300]
    df_train = df[df["big_recording_index"].isin(train_set)].copy()
    df_big_train = df_big[df_big["big_recording_index"].isin(train_set)].copy()
    df_stats_train = df_stats[df_stats["big_recording_index"].isin(train_set)].copy()
    df_val = df[df["big_recording_index"].isin(val_set)].copy()
    df_big_val = df_big[df_big["big_recording_index"].isin(val_set)].copy()
    df_stats_val = df_stats[df_stats["big_recording_index"].isin(val_set)].copy()
    ##### Specfic to NAS ####
    train_loader, len_train, len_train_sampler = return_loader(
        args, df_train, df_big_train, df_stats_train, path_recordings, duration_min, True
    )
    val_loader, len_val, len_val_sampler = return_loader(
        args, df_val, df_big_val, df_stats_val, path_recordings, duration_min, False
    )
    return train_loader, val_loader, len_train, len_val, len_train_sampler, len_val_sampler


def return_loader(args, df, df_big, df_stats, path_recordings, duration_min, train):
    dict_groups, segs, df_big = compute_groups_segs(df, df_big, duration_min, None)
    if train:
        dataset = EEGDataset(
            segs,
            dict_groups,
            df_big,
            df_stats,
            path_recordings,
            clip=args.clip,
            masking=0.55,
            radius_spat_mask=0.03,
            radius_temp_mask=3,
            dropout_ratio=0.1,
            dropout_ratio_radius=0.04,
        )
        sampler = GroupedSampler(dataset, batch_size=args.batch_size, drop_last=True, n_gpu=args.n_gpus)
    else:
        dataset = EEGDataset(segs, dict_groups, df_big, df_stats, path_recordings, args.clip)
        sampler = ValGroupedSampler(dataset, batch_size=args.batch_size, drop_last=True, n_gpu=args.n_gpus)
    sampler.__iter__()  # can be done only in main if we want
    len_sampler = len(sampler)
    loader = DataLoader(
        dataset,
        pin_memory=True,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        persistent_workers=True,
        prefetch_factor=args.prefect_factor,
    )
    return loader, len(dataset), len_sampler


# def return_single_loader(args,df,df_big,df_stats,path_recordings,duration_min,train):
#     dict_groups,segs,df_big = compute_groups_segs(df,df_big, duration_min, None)
#     random = True if train else False
#     if train:
#         dataset = EEGDataset(segs,dict_groups,df_big,df_stats,path_recordings, clip = args.clip,masking=0.55,radius_spat_mask=0.03,radius_temp_mask=3,dropout_ratio=0.1,dropout_ratio_radius=0.04)
#     else:
#         dataset = EEGDataset(segs, dict_groups, df_big, df_stats,path_recordings,args.clip)
#     sampler = GroupedSampler(dataset,batch_size = args.batch_size,drop_last=True)
#     sampler.__iter__() #can be done only in main if we want
#     len_sampler = len(sampler)
#     #if args.dataloader == 'standard':
#     #    loader = DataLoader(dataset, pin_memory=True,batch_sampler=sampler, num_workers=args.num_workers,persistent_workers=True)
#     if args.dataloader == 'multi' and train:
#         loader = MultiEpochsDataLoader(dataset, pin_memory=True,batch_sampler=sampler, num_workers=args.num_workers,persistent_workers=True,prefetch_factor=args.prefect_factor) #754
#     else:
#         loader = DataLoader(dataset, pin_memory=True,batch_sampler=sampler, num_workers=args.num_workers,persistent_workers=True)
#     return loader,len(dataset),len_sampler
