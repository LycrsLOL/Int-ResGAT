import os
import logging
import random
import pandas as pd
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler

def load_and_process_database(config):
    db_path = config.get('database_path')
    if not db_path:
        db_path = config.get('initialize', {}).get('database_path')
        
    if not db_path:
         raise ValueError("database_path not found in config")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at {db_path}")

    df = pd.read_csv(db_path)
    
    grouped = df.groupby('uniprot_id').agg({
        'ec_numbers': 'first',
        'pdb_id': lambda x: list(x.dropna())
    }).reset_index()
    
    filtered_df = grouped

    unique_ecs = sorted(filtered_df['ec_numbers'].unique())
    ec_to_label = {ec: i for i, ec in enumerate(unique_ecs)}
    filtered_df['label'] = filtered_df['ec_numbers'].map(ec_to_label)
    
    num_classes = len(unique_ecs)
    
    class_counts = filtered_df['label'].value_counts().sort_index()
    
    counts_np = class_counts.values
    total_samples = len(filtered_df)
    class_weights = total_samples / (num_classes * counts_np + 1e-6)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logging.info(f"Total samples after filtering: {len(filtered_df)}")
        logging.info(f"Total classes: {num_classes}")
        logging.info(f"Class Weights Range: [{class_weights.min():.4f}, {class_weights.max():.4f}]")
    
    return filtered_df, num_classes, ec_to_label, class_weights

class ProteinGraphDataset(Dataset):
    def __init__(self, dataframe, graph_dir, random_state=42, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.dataframe = dataframe
        self.graph_dir = graph_dir
        self.data_list = []
        rng = random.Random(random_state)

        is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        
        if is_main_process:
            logging.info(f"Initializing dataset with {len(dataframe)} entries from graph_dir={graph_dir} ...")
        missing_count = 0
        found_with_pdb = 0
        found_without_pdb = 0
        
        for _, row in dataframe.iterrows():
            uniprot_id = row['uniprot_id']
            pdb_ids = row['pdb_id']
            label = row['label']
            
            if pdb_ids and len(pdb_ids) > 0:
                valid_file_paths = []
                for selected_pdb in pdb_ids:
                    file_name = f"{uniprot_id}_{selected_pdb}.pt"
                    file_path = os.path.join(graph_dir, file_name)
                    if os.path.exists(file_path):
                        valid_file_paths.append(file_path)
                if valid_file_paths:
                    selected_file_path = rng.choice(valid_file_paths)
                    self.data_list.append((selected_file_path, label))
                    found_with_pdb += 1
                else:
                    missing_count += 1
            else:
                file_name = f"{uniprot_id}.pt"
                file_path = os.path.join(graph_dir, file_name)
                if os.path.exists(file_path):
                    self.data_list.append((file_path, label))
                    found_without_pdb += 1
                else:
                    missing_count += 1
        
        if missing_count > 0 and is_main_process:
            logging.warning(f"Warning: Skipped {missing_count} missing files.")
        if is_main_process:
            logging.info(f"Found samples with pdb_id: {found_with_pdb}")
            logging.info(f"Found samples without pdb_id: {found_without_pdb}")
            logging.info(f"Dataset initialized with {len(self.data_list)} valid samples.")

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        file_path, label = self.data_list[idx]
        try:
            data = torch.load(file_path, map_location='cpu')
            data.y = torch.tensor([label], dtype=torch.long)
            return data
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            raise e

def create_dataloaders(config, distributed=False):
    df, num_classes, _, class_weights = load_and_process_database(config)
    
    try:
        train_df, test_df = train_test_split(
            df, 
            test_size=config['train']['test_size'], 
            stratify=df['label'],
            random_state=config['train']['random_state']
        )
    except ValueError:
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logging.warning("Stratified split failed (likely due to classes with < 2 samples). Falling back to random split.")
        
        train_df, test_df = train_test_split(
            df, 
            test_size=config['train']['test_size'], 
            stratify=None,
            random_state=config['train']['random_state']
        )
    
    save_dir = config.get('save_dir')
    if not save_dir:
        save_dir = config.get('initialize', {}).get('save_dir') or config.get('train', {}).get('save_dir')
        if not save_dir:
             raise ValueError("save_dir not found in config")
             
    graph_dir = config.get('graph_dir')
    if not graph_dir:
        graph_dir = os.path.join(save_dir, 'graph')
    if not os.path.exists(graph_dir):
        fallback_graph_dir = '/data/lihb/enzyme_functional_annotation/optimal_radius/graph'
        if os.path.exists(fallback_graph_dir):
            graph_dir = fallback_graph_dir
    if not os.path.exists(graph_dir):
        raise FileNotFoundError(f"graph directory not found: {graph_dir}")

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logging.info(f"Using graph_dir: {graph_dir}")

    random_state = config['train']['random_state']
    train_dataset = ProteinGraphDataset(train_df, graph_dir, random_state=random_state)
    test_dataset = ProteinGraphDataset(test_df, graph_dir, random_state=random_state)
    
    batch_size = config['train']['batch_size']
    num_workers = config.get('num_workers', 0)
    
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
    }
    
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = False
        loader_kwargs['prefetch_factor'] = 2
        loader_kwargs['multiprocessing_context'] = 'spawn'

    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        shuffle = False
    else:
        train_sampler = None
        test_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset, 
        shuffle=shuffle, 
        sampler=train_sampler,
        **loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset, 
        shuffle=False, 
        sampler=test_sampler,
        **loader_kwargs
    )
    
    return train_loader, test_loader, num_classes, class_weights
