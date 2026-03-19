import os
import sys
import argparse
import logging
import math
import yaml
import numpy as np
from datetime import datetime, timedelta
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, recall_score, precision_score, top_k_accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from tqdm import tqdm

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="The usage of `scatter(reduce='max')`")

from train.models import HE_ResGATConv
from train.loss import FocalLoss
from train.utils import create_dataloaders

def setup_distributed():
    """初始化分布式环境"""
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        # Set timeout to 60 minutes to prevent crash during long epoch transitions (e.g. model saving or data loading)
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=60))
        return local_rank, True
    return 0, False

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def gather_tensor(tensor, device):
    """从所有进程收集 Tensor"""
    world_size = dist.get_world_size()
    # 假设所有进程的 tensor 形状相同 (DistributedSampler 默认会 pad 数据)
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list)

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, is_distributed, print_freq=100):
    """
    训练一个 Epoch
    """
    if is_distributed:
        loader.sampler.set_epoch(epoch)
        
    model.train()
    total_loss = 0.0
    
    total_steps = len(loader)
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # 前向传播
        out = model(batch)
        
        # 计算损失
        loss = criterion(out, batch.y)
        loss.backward()
        
        # 梯度裁剪 (可选，但推荐)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        
        # 打印进度
        step_idx = batch_idx + 1
        if step_idx == 1 or step_idx % print_freq == 0 or step_idx == total_steps:
            # 仅在主进程打印，或者每个进程都打印但带上 rank
            if not is_distributed or dist.get_rank() == 0:
                logging.info(f"Epoch: [{epoch}] Step : [{step_idx}/{total_steps}] Loss: {loss.item():.4f}")
        
    # 计算平均 Loss
    avg_loss = total_loss / len(loader.dataset) 
    if is_distributed:
        # 简单 reduce 平均 loss (假设各进程 batch 数差不多)
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / dist.get_world_size()

    return avg_loss

def evaluate(model, loader, criterion, device, is_distributed):
    """
    验证/测试模型
    """
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    
    is_main_process = not is_distributed or (dist.is_initialized() and dist.get_rank() == 0)
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", disable=not is_main_process)
        for batch in pbar:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            
            total_loss += loss.item() * batch.num_graphs
            
            # 获取Logits (用于Top-k) 和 标签
            all_logits.append(out)
            all_labels.append(batch.y)
            
    # 局部指标
    local_logits = torch.cat(all_logits)
    local_labels = torch.cat(all_labels)
    
    # 分布式汇总
    if is_distributed:
        # 汇总 Loss
        loss_tensor = torch.tensor(total_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        # 注意：len(loader.dataset) 是总数据集大小
        global_avg_loss = loss_tensor.item() / len(loader.dataset)
        
        # 汇总 Logits 和 Labels
        # 注意: gather_tensor 假设所有进程的数据量相等或经过 padding，如果最后一批不一样可能会有问题
        # 但在 DistributedSampler 中通常会自动补齐。
        global_logits = gather_tensor(local_logits, device)
        global_labels = gather_tensor(local_labels, device)
        
        # 转移到 CPU
        np_logits = global_logits.cpu().numpy()
        np_labels = global_labels.cpu().numpy()
    else:
        global_avg_loss = total_loss / len(loader.dataset)
        np_logits = local_logits.cpu().numpy()
        np_labels = local_labels.cpu().numpy()
    
    if np_labels.ndim > 1:
        if np_labels.shape[-1] == 1:
            np_labels = np_labels.reshape(-1)
        else:
            np_labels = np.argmax(np_labels, axis=1)
    else:
        np_labels = np_labels.reshape(-1)
    np_labels = np_labels.astype(np.int64, copy=False)
    np_preds = np_logits.argmax(axis=1).reshape(-1).astype(np.int64, copy=False)
    all_class_labels = np.arange(np_logits.shape[1], dtype=np.int64)
    
    # 计算指标
    metrics = {}
    metrics['loss'] = global_avg_loss
    
    # Accuracy
    metrics['acc'] = accuracy_score(np_labels, np_preds)
    # Balanced Accuracy (Macro Recall)
    metrics['acc_macro'] = balanced_accuracy_score(np_labels, np_preds)
    
    # Top-k Accuracy (Default k=5, ensure k < n_classes)
    n_classes = np_logits.shape[1]
    k = min(5, n_classes)
    if hasattr(top_k_accuracy_score, '__call__'): # Check if available (sklearn > 0.24)
        try:
            metrics[f'top{k}_acc'] = top_k_accuracy_score(np_labels, np_logits, k=k, labels=np.arange(n_classes))
        except Exception:
            metrics[f'top{k}_acc'] = 0.0
    
    # Precision, Recall, F1 (Macro & Weighted)
    metrics['precision_macro'] = precision_score(np_labels, np_preds, average='macro', labels=all_class_labels, zero_division=0)
    metrics['recall_macro'] = recall_score(np_labels, np_preds, average='macro', labels=all_class_labels, zero_division=0)
    metrics['f1_macro'] = f1_score(np_labels, np_preds, average='macro', labels=all_class_labels, zero_division=0)
    
    # Calculate per-class metrics first to avoid sklearn weighted average crash
    try:
        p_per_class, r_per_class, f_per_class, s_per_class = precision_recall_fscore_support(
            np_labels, np_preds, average=None, labels=all_class_labels, zero_division=0
        )
        
        # Calculate weighted averages manually
        total_support = np.sum(s_per_class)
        if total_support > 0:
            metrics['precision_weighted'] = np.average(p_per_class, weights=s_per_class)
            metrics['recall_weighted'] = np.average(r_per_class, weights=s_per_class)
            metrics['f1_weighted'] = np.average(f_per_class, weights=s_per_class)
        else:
            metrics['precision_weighted'] = 0.0
            metrics['recall_weighted'] = 0.0
            metrics['f1_weighted'] = 0.0
    except Exception as e:
        # logging.error(f"Error calculating weighted metrics: {e}")
        metrics['precision_weighted'] = 0.0
        metrics['recall_weighted'] = 0.0
        metrics['f1_weighted'] = 0.0
    
    # AUROC (Macro & Weighted) - using One-vs-Rest strategy for multi-class
    # Softmax probabilities are needed for AUROC
    try:
        # Check if we have probabilities (logits -> softmax)
        np_probs = torch.softmax(torch.tensor(np_logits), dim=1).numpy()
        metrics['auroc_macro'] = roc_auc_score(np_labels, np_probs, multi_class='ovr', average='macro', labels=all_class_labels)
        metrics['auroc_weighted'] = roc_auc_score(np_labels, np_probs, multi_class='ovr', average='weighted', labels=all_class_labels)
    except Exception:
        # AUROC might fail if not all classes are present in y_true
        metrics['auroc_macro'] = 0.0
        metrics['auroc_weighted'] = 0.0
    
    return metrics

def run_train(args):
    """
    训练流程入口函数
    """
    local_rank, is_distributed = setup_distributed()
    is_main_process = local_rank == 0
    
    try:
        # 1. 加载配置
        config_path = getattr(args, 'config', 'config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if is_main_process:
            logging.info("========== Training Configuration ==========")
            train_config_log = config.get('train', {})
            for key, value in train_config_log.items():
                logging.info(f"  {key}: {value}")
            logging.info("============================================")
            
        # 设置设备
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if is_main_process:
            logging.info(f"Using device: {device} (Distributed: {is_distributed})")
        
        # 2. 数据处理与加载
        if is_main_process:
            logging.info("Loading data...")
        
        # 只有主进程打印数据加载信息，但所有进程都需要创建 loader
        # 我们已经在 utils.py 里处理了打印逻辑
        train_loader, test_loader, num_classes, class_weights = create_dataloaders(config, distributed=is_distributed)
        
        if is_main_process:
            world_size = dist.get_world_size() if is_distributed else 1
            train_samples = len(train_loader.dataset)
            test_samples = len(test_loader.dataset)
            batch_size_per_rank = train_config_log.get('batch_size', config.get('train', {}).get('batch_size', 1))
            global_batch_size = batch_size_per_rank * world_size
            expected_train_steps = math.ceil(train_samples / global_batch_size)
            expected_test_steps = math.ceil(test_samples / global_batch_size)
            logging.info(
                f"Data loaded. Train steps per epoch (optimizer updates): {len(train_loader)} | "
                f"Expected: {expected_train_steps} | Train samples: {train_samples} | "
                f"Batch/GPU: {batch_size_per_rank} | World Size: {world_size} | Global Batch: {global_batch_size}"
            )
            logging.info(
                f"Data loaded. Test steps per epoch: {len(test_loader)} | "
                f"Expected: {expected_test_steps} | Test samples: {test_samples}"
            )
        
        # 3. 初始化模型
        train_config = config['train']
        model = HE_ResGATConv(
            input_dim=train_config['input_dim'],
            hidden_dim=train_config['hidden_dim'],
            num_classes=num_classes,
            dropout=train_config['dropout'],
            num_layers=train_config['num_layers'],
            heads=train_config['heads']
        ).to(device)
        
        # DDP 包装
        if is_distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        
        if is_main_process:
            # print(f"Model initialized: {model}") # 模型结构太长，不仅打印
            pass
        
        # 4. 损失函数与优化器
        alpha = train_config.get('focal_loss_alpha', 0.25)
        
        # Check if alpha is 'auto' (str)
        if isinstance(alpha, str) and alpha.lower() == 'auto':
            alpha = class_weights.to(device)
            if is_main_process:
                logging.info(f"Using Automatic Class Weights for Focal Loss. Range: [{alpha.min():.4f}, {alpha.max():.4f}]")
        
        criterion = FocalLoss(
            alpha=alpha, 
            gamma=train_config['focal_loss_gamma']
        ).to(device)
        
        optimizer = optim.Adam(
            model.parameters(), 
            lr=train_config['learning_rate'],
            weight_decay=train_config.get('weight_decay', 0.0)
        )
        
        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, 
            patience=train_config.get('lr_patience', 10), 
            verbose=True
        )
        
        # 5. 训练循环
        epochs = train_config['epochs']
        save_dir = config.get('save_dir')
        if not save_dir:
            save_dir = train_config.get('save_dir')
        
        print_freq = config.get('print_freq', 100)
        
        if is_main_process:
            os.makedirs(save_dir, exist_ok=True)
        
        best_f1 = 0.0
        best_val_loss = float('inf')
        early_stopping_counter = 0
        start_epoch = 1

        # 断点续训逻辑
        if hasattr(args, 'resume') and args.resume:
            if os.path.isfile(args.resume):
                if is_main_process:
                    logging.info(f"=> loading checkpoint '{args.resume}'")
                # map_location ensures we load to the correct device
                checkpoint = torch.load(args.resume, map_location=device)
                start_epoch = checkpoint['epoch'] + 1
                best_f1 = checkpoint['best_f1']
                best_val_loss = checkpoint.get('best_val_loss', float('inf')) # Handle backward compatibility
                early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
                
                # Load model state
                if is_distributed:
                    model.module.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint['state_dict'])
                    
                optimizer.load_state_dict(checkpoint['optimizer'])
                if 'scheduler' in checkpoint and scheduler:
                     scheduler.load_state_dict(checkpoint['scheduler'])
                     
                if is_main_process:
                    logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            else:
                if is_main_process:
                    logging.info(f"=> no checkpoint found at '{args.resume}'")

        if is_main_process:
            logging.info("Start training...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            
        for epoch in range(start_epoch, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, is_distributed, print_freq=print_freq)
            test_metrics = evaluate(model, test_loader, criterion, device, is_distributed)
            
            # LR Scheduler Step
            scheduler.step(test_metrics['loss'])

            # Early Stopping Check
            if test_metrics['loss'] < best_val_loss:
                best_val_loss = test_metrics['loss']
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter >= train_config.get('early_stopping_patience', 20):
                if epoch >= train_config.get('min_epochs', 0):
                    if is_main_process:
                        logging.info(f"Early stopping triggered at epoch {epoch} (min_epochs={train_config.get('min_epochs', 0)}).")
                    break
                else:
                    if is_main_process and epoch % print_freq == 0:
                         logging.info(f"Early stopping condition met (counter={early_stopping_counter}), but waiting for min_epochs ({train_config.get('min_epochs', 0)}). Current epoch: {epoch}")

            
            if is_main_process:
                # 动态构建日志字符串
                log_str = f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_metrics['loss']:.4f} | "
                log_str += f"Acc: {test_metrics['acc']:.4f} | "
                
                # 添加 Top-k 如果存在
                for k in [1, 5]:
                    if f'top{k}_acc' in test_metrics:
                         log_str += f"Top-{k}: {test_metrics[f'top{k}_acc']:.4f} | "
                
                log_str += f"F1(M): {test_metrics['f1_macro']:.4f} | Prec(M): {test_metrics['precision_macro']:.4f} | Rec(M): {test_metrics['recall_macro']:.4f} | Prec(W): {test_metrics['precision_weighted']:.4f}"
                
                # 添加 AUROC
                if 'auroc_macro' in test_metrics and test_metrics['auroc_macro'] > 0:
                     log_str += f" | AUROC(M): {test_metrics['auroc_macro']:.4f}"
                
                logging.info(log_str)
                
                # 6. 模型保存逻辑 (监控 F1-Macro)
                current_f1 = test_metrics['f1_macro']
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    save_subdir = os.path.join(save_dir, 'best_model')
                    os.makedirs(save_subdir, exist_ok=True)
                    save_path = os.path.join(save_subdir, f'train_{timestamp}.pth')
                    
                    # 保存原始模型（去掉 DDP wrapper）
                    model_to_save = model.module if is_distributed else model
                    torch.save(model_to_save.state_dict(), save_path)
                    logging.info(f"--> Best model saved with F1: {best_f1:.4f} to {save_path}")

                # Save latest checkpoint for resuming
                checkpoint_state = {
                    'epoch': epoch,
                    'state_dict': model.module.state_dict() if is_distributed else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler else None,
                    'best_f1': best_f1,
                    'best_val_loss': best_val_loss,
                    'early_stopping_counter': early_stopping_counter
                }
                latest_checkpoint_path = os.path.join(save_dir, 'checkpoint_latest.pth')
                torch.save(checkpoint_state, latest_checkpoint_path)
        
        if is_main_process:
            logging.info("Training completed.")

    except KeyboardInterrupt:
        if is_main_process:
            logging.warning("Training interrupted by user. Cleaning up...")
    except Exception as e:
        if is_main_process:
            logging.error(f"An error occurred: {e}")
        raise e
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Graph Neural Network")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default='', help='Path to latest checkpoint (default: none)')
    args = parser.parse_args()
    
    # Setup logging for standalone execution
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode='a', encoding='utf-8')]
    )
    logging.info(f" >> Log File      : {log_file}")
    
    run_train(args)
