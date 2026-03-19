import argparse
import logging
import os
import sys
import yaml
import torch
import multiprocessing as mp
from datetime import datetime

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataloader.dataloader import load_database
from train.train import run_train

def parse_args() -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('-c', '--config', type=str, default='config.yaml')
    base_parser.add_argument('--mode', type=str)
    args, remaining_argv = base_parser.parse_known_args()

    yaml_data = {}
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f) or {}

    current_mode = args.mode if args.mode else yaml_data.get('mode', 'initialize')

    config_defaults = {}
    for k, v in yaml_data.items():
        if not isinstance(v, dict): config_defaults[k] = v
    config_defaults['mode'] = current_mode

    mode_defaults = yaml_data.get(current_mode, {})
    if isinstance(mode_defaults, dict): config_defaults.update(mode_defaults)

    parser = argparse.ArgumentParser()
    parser.set_defaults(**config_defaults)

    parser.add_argument('--mode', type=str, choices=['initialize', 'train'])
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--num_workers', type=int, help="多进程并发数")
    
    parser.add_argument('--database_path', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--graph_dir', type=str, help="Path to graph directory")
    parser.add_argument('--pretrained_model', type=str, help="Path to pretrained model for optimization")
    parser.add_argument('--resume', type=str, help="Path to checkpoint to resume from")
    parser.add_argument('--device', type=str)
    parser.add_argument('--max_edge_distance', type=float)
    
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')
    parser.add_argument('--print_freq', type=int)

    final_args = parser.parse_args(remaining_argv)
    
    # Preserve config path
    final_args.config = args.config

    if final_args.device is None: final_args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if final_args.save_dir is None: final_args.save_dir = "./output"
    if final_args.log_dir is None: final_args.log_dir = "./logs"
    if final_args.num_workers is None: final_args.num_workers = 4
    if final_args.max_edge_distance is None: final_args.max_edge_distance = 8.0

    return final_args

def setup_logging(log_dir: str, mode: str):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Get rank from environment variables (torchrun sets these)
    rank = int(os.environ.get('RANK', 0))
    
    handlers = [logging.StreamHandler(sys.stdout)]
    log_file = None
    
    # Only add FileHandler for rank 0 to prevent duplicate logs
    if rank == 0:
        prefix = "_" if mode == "optimize" else ""
        log_file = os.path.join(log_dir, f"{prefix}{mode}_{timestamp}.log")
        handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True
    )
    return log_file

if __name__ == "__main__":
    args = parse_args()
    log_file_path = setup_logging(args.log_dir, args.mode)
    
    # Get rank again for conditional logging
    rank = int(os.environ.get('RANK', 0))
    
    if rank == 0:
        logging.info("==============================================")
        logging.info(f"      Starting Engine in Mode: [{args.mode.upper()}]")
        logging.info("==============================================")

    if args.mode == 'initialize':
        load_database(args)
    elif args.mode == 'train':
        run_train(args)
    
    try:
        if args.mode == 'initialize':
            load_database(args)
            if rank == 0:
                logging.info(f"[{args.mode.upper()}] executed successfully.")
        elif args.mode == 'train':
            run_train(args)
            if rank == 0:
                logging.info(f"[{args.mode.upper()}] executed successfully.")
    except Exception as e:
        logging.error(f"Engine terminated with an error: {e}", exc_info=True)