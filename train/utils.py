import logging
import os
import random
from collections import Counter

import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader


CLUSTER_COLUMN_CANDIDATES = (
    "homology_cluster",
    "cluster_id",
    "family_id",
    "mmseqs_cluster",
    "cdhit_cluster",
    "sequence_cluster",
)


def _is_main_process():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _valid_text(value):
    return pd.notna(value) and str(value).strip().lower() not in {"", "nan", "none", "null"}


def split_ec_numbers(value):
    if not _valid_text(value):
        return []
    labels = []
    for token in str(value).replace("|", ";").replace(",", ";").split(";"):
        token = token.strip()
        if token:
            labels.append(token)
    return sorted(set(labels))


def _merge_ec_numbers(values):
    labels = []
    for value in values:
        labels.extend(split_ec_numbers(value))
    return ";".join(sorted(set(labels)))


def _merge_pdb_ids(values):
    pdb_ids = []
    for value in values:
        if _valid_text(value):
            pdb_id = str(value).strip()
            if pdb_id not in pdb_ids:
                pdb_ids.append(pdb_id)
    return pdb_ids


def _first_valid(values):
    for value in values:
        if _valid_text(value):
            return value
    return None


def _read_table(path):
    suffix = os.path.splitext(path)[1].lower()
    sep = "\t" if suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def _attach_homology_clusters(df, config):
    train_config = config.get("train", {})
    cluster_file = train_config.get("homology_cluster_file") or config.get("homology_cluster_file")
    if not cluster_file:
        return df

    if not os.path.exists(cluster_file):
        raise FileNotFoundError(f"homology_cluster_file not found: {cluster_file}")

    clusters = _read_table(cluster_file)
    if "uniprot_id" not in clusters.columns:
        raise ValueError("homology_cluster_file must contain a 'uniprot_id' column")

    configured_col = train_config.get("homology_group_column")
    cluster_col = configured_col if configured_col in clusters.columns else None
    if cluster_col is None:
        cluster_col = next((col for col in CLUSTER_COLUMN_CANDIDATES if col in clusters.columns), None)
    if cluster_col is None:
        raise ValueError(
            "homology_cluster_file must contain one cluster column, for example "
            "'homology_cluster', 'cluster_id' or 'mmseqs_cluster'"
        )

    clusters = clusters[["uniprot_id", cluster_col]].drop_duplicates("uniprot_id")
    if cluster_col != "homology_cluster":
        clusters = clusters.rename(columns={cluster_col: "homology_cluster"})

    return df.merge(clusters, on="uniprot_id", how="left")


def _find_group_column(df, train_config):
    configured_col = train_config.get("homology_group_column")
    if configured_col and configured_col in df.columns:
        return configured_col
    return next((col for col in CLUSTER_COLUMN_CANDIDATES if col in df.columns), None)


def _make_multihot(labels, label_to_idx):
    vector = torch.zeros(len(label_to_idx), dtype=torch.float32)
    for label in labels:
        vector[label_to_idx[label]] = 1.0
    return vector


def load_and_process_database(config):
    db_path = config.get("database_path") or config.get("initialize", {}).get("database_path")
    if not db_path:
        raise ValueError("database_path not found in config")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at {db_path}")

    raw_df = pd.read_csv(db_path, low_memory=False)
    if "uniprot_id" not in raw_df.columns or "ec_numbers" not in raw_df.columns:
        raise ValueError("database must contain 'uniprot_id' and 'ec_numbers' columns")

    raw_df = _attach_homology_clusters(raw_df, config)
    raw_df = raw_df[raw_df["uniprot_id"].apply(_valid_text)].copy()

    agg = {
        "ec_numbers": _merge_ec_numbers,
        "pdb_id": _merge_pdb_ids,
    }
    for optional_col in ["sequence", "active_sites", *CLUSTER_COLUMN_CANDIDATES]:
        if optional_col in raw_df.columns and optional_col not in agg:
            agg[optional_col] = _first_valid

    filtered_df = raw_df.groupby("uniprot_id", as_index=False).agg(agg)
    filtered_df["ec_list"] = filtered_df["ec_numbers"].apply(split_ec_numbers)
    filtered_df = filtered_df[filtered_df["ec_list"].map(len) > 0].reset_index(drop=True)

    label_names = sorted({label for labels in filtered_df["ec_list"] for label in labels})
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}
    filtered_df["label_vector"] = filtered_df["ec_list"].apply(lambda labels: _make_multihot(labels, label_to_idx))
    filtered_df["label_key"] = filtered_df["ec_list"].apply(lambda labels: ";".join(labels))

    label_matrix = torch.stack(filtered_df["label_vector"].tolist())
    positive_counts = label_matrix.sum(dim=0)
    negative_counts = label_matrix.size(0) - positive_counts
    pos_weight = negative_counts / positive_counts.clamp_min(1.0)
    max_pos_weight = config.get("train", {}).get("max_pos_weight")
    if max_pos_weight is not None:
        pos_weight = pos_weight.clamp(max=float(max_pos_weight))

    if _is_main_process():
        multi_label_samples = int((label_matrix.sum(dim=1) > 1).sum().item())
        logging.info(f"Total proteins after grouping: {len(filtered_df)}")
        logging.info(f"Total individual EC labels: {len(label_names)}")
        logging.info(f"Multi-label proteins: {multi_label_samples}")
        logging.info(f"Positive class weights range: [{pos_weight.min():.4f}, {pos_weight.max():.4f}]")

    return filtered_df, len(label_names), label_to_idx, pos_weight, label_names


def split_dataframe(df, config):
    train_config = config["train"]
    test_size = train_config["test_size"]
    random_state = train_config["random_state"]
    strategy = str(train_config.get("split_strategy", "homology")).lower()

    group_col = _find_group_column(df, train_config)
    if strategy in {"homology", "group", "cluster"}:
        if group_col is None:
            raise ValueError(
                "Homology split was requested, but no homology cluster column was found. "
                "Add a column such as 'homology_cluster' to database.csv or provide "
                "train.homology_cluster_file in config.yaml. Set train.split_strategy=random "
                "only for debugging, not for the manuscript benchmark."
            )
        missing_groups = ~df[group_col].apply(_valid_text)
        if missing_groups.any():
            raise ValueError(
                f"Homology split requires a non-empty '{group_col}' value for every protein; "
                f"{int(missing_groups.sum())} grouped proteins are missing it."
            )
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        overlap = set(train_df[group_col].dropna()) & set(test_df[group_col].dropna())
        if overlap:
            raise RuntimeError(f"Homology split leakage detected in {len(overlap)} groups")
        if _is_main_process():
            logging.info(
                f"Using homology-group split by '{group_col}': "
                f"{len(train_df)} train / {len(test_df)} test proteins"
            )
        return train_df, test_df

    if strategy == "auto" and group_col is not None:
        auto_config = dict(config)
        auto_config["train"] = dict(train_config)
        auto_config["train"]["split_strategy"] = "homology"
        return split_dataframe(df, auto_config)

    if strategy not in {"random", "stratified", "auto"}:
        raise ValueError(f"Unknown split_strategy: {strategy}")

    stratify = None
    if strategy == "stratified":
        counts = Counter(df["label_key"])
        if all(counts[key] >= 2 for key in df["label_key"]):
            stratify = df["label_key"]
        elif _is_main_process():
            logging.warning("Composite-label stratification skipped because at least one label set has <2 samples.")

    if strategy == "auto" and _is_main_process():
        logging.warning("No homology cluster column found; falling back to random split.")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


class ProteinGraphDataset(Dataset):
    def __init__(self, dataframe, graph_dir, random_state=42, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.dataframe = dataframe
        self.graph_dir = graph_dir
        self.data_list = []
        rng = random.Random(random_state)

        if _is_main_process():
            logging.info(f"Initializing dataset with {len(dataframe)} entries from graph_dir={graph_dir} ...")

        missing_count = 0
        found_with_pdb = 0
        found_without_pdb = 0

        for _, row in dataframe.iterrows():
            uniprot_id = str(row["uniprot_id"]).strip()
            pdb_ids = row["pdb_id"] if isinstance(row["pdb_id"], list) else _merge_pdb_ids([row["pdb_id"]])
            label_vector = row["label_vector"].clone().detach().float()

            valid_file_paths = []
            for selected_pdb in pdb_ids:
                file_path = os.path.join(graph_dir, f"{uniprot_id}_{selected_pdb}.pt")
                if os.path.exists(file_path):
                    valid_file_paths.append(file_path)

            if valid_file_paths:
                self.data_list.append((rng.choice(valid_file_paths), label_vector))
                found_with_pdb += 1
                continue

            file_path = os.path.join(graph_dir, f"{uniprot_id}.pt")
            if os.path.exists(file_path):
                self.data_list.append((file_path, label_vector))
                found_without_pdb += 1
            else:
                missing_count += 1

        if missing_count > 0 and _is_main_process():
            logging.warning(f"Skipped {missing_count} entries with missing graph files.")
        if _is_main_process():
            logging.info(f"Found samples with pdb_id: {found_with_pdb}")
            logging.info(f"Found samples without pdb_id: {found_without_pdb}")
            logging.info(f"Dataset initialized with {len(self.data_list)} valid samples.")

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        file_path, label_vector = self.data_list[idx]
        try:
            data = torch.load(file_path, map_location="cpu")
            data.y = label_vector.unsqueeze(0)
            return data
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            raise


def create_dataloaders(config, distributed=False):
    df, num_classes, _, class_weights, label_names = load_and_process_database(config)
    train_df, test_df = split_dataframe(df, config)

    save_dir = config.get("save_dir") or config.get("initialize", {}).get("save_dir") or config.get("train", {}).get("save_dir")
    if not save_dir:
        raise ValueError("save_dir not found in config")

    graph_dir = config.get("graph_dir") or os.path.join(save_dir, "graph")
    if not os.path.exists(graph_dir):
        fallback_graph_dir = "/data/lihb/enzyme_functional_annotation/optimal_radius/graph"
        if os.path.exists(fallback_graph_dir):
            graph_dir = fallback_graph_dir
    if not os.path.exists(graph_dir):
        raise FileNotFoundError(f"graph directory not found: {graph_dir}")

    if _is_main_process():
        logging.info(f"Using graph_dir: {graph_dir}")

    random_state = config["train"]["random_state"]
    train_dataset = ProteinGraphDataset(train_df, graph_dir, random_state=random_state)
    test_dataset = ProteinGraphDataset(test_df, graph_dir, random_state=random_state)

    batch_size = config["train"]["batch_size"]
    num_workers = config.get("num_workers", 0)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = False
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["multiprocessing_context"] = "spawn"

    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        shuffle = False
    else:
        train_sampler = None
        test_sampler = None
        shuffle = True

    train_loader = DataLoader(train_dataset, shuffle=shuffle, sampler=train_sampler, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, sampler=test_sampler, **loader_kwargs)

    return train_loader, test_loader, num_classes, class_weights, label_names
