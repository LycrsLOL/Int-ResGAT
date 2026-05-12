import argparse
import logging
import os
import sys
import pandas as pd
import torch

from esm.models.esm3 import ESM3
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.utils.constants.models import ESM3_OPEN_SMALL

from .utils import (
    download_structures,
    complete_structures,
    extract_embedding,
    verify_embeddings,
    build_graphs
)

# 集中清理 Tokenizer 属性防止冲突
for _attr in ['cls_token', 'cls_token_id', 'eos_token', 'eos_token_id', 
              'mask_token', 'mask_token_id', 'pad_token', 'pad_token_id']:
    if hasattr(EsmSequenceTokenizer, _attr):
        delattr(EsmSequenceTokenizer, _attr)

def load_database(args: argparse.Namespace) -> None:
    # 确保 ESM 源码路径在 sys.path 中
    sys.path.append("/data/lihb/enzyme_functional_annotation/optimal_radius/esm-main")
    os.makedirs(args.save_dir, exist_ok=True)

    df = pd.read_csv(args.database_path)
    
    pdb_ids_ls = df['pdb_id'].tolist()
    uniprot_ids_ls = df['uniprot_id'].tolist()

    # [Step 1] 下载结构
    download_structures(df, args)
    
    # [Step 2] 结构补全与优化
    complete_structures(df, args)

    # [Step 3] 提取特征
    logging.info("========== [Step 3] Extract Embeddings ==========")
    
    m_path = args.model_path if hasattr(args, 'model_path') else None
    
    with torch.device(args.device):
        try:
            logging.info(f"   >> Initializing ESM3 architecture ({ESM3_OPEN_SMALL})...")
            # 第一步：必须传入合法的 registry name 实例化模型骨架
            model = ESM3.from_pretrained(ESM3_OPEN_SMALL)
            
            # 第二步：手动覆盖加载指定的本地 .pth 权重文件
            if m_path and os.path.exists(m_path):
                logging.info(f"   >> Overriding with local weights from: {m_path}")
                state_dict = torch.load(m_path, map_location='cpu')
                
                # 兼容不同的权重保存格式 (提取真实权重层)
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
                    
                model.load_state_dict(state_dict, strict=False)
            elif m_path:
                logging.warning(f"   [Warning] Local weights not found at {m_path}. Proceeding with default.")
                
            model = model.to(args.device).eval()
            logging.info("   Model loaded successfully.")
            
        except Exception as e:
            logging.error(f"   [Error] Failed to load ESM3 model: {e}")
            return

        extract_embedding(model=model,
                          pdb_ids_ls=pdb_ids_ls,
                          uniprot_ids_ls=uniprot_ids_ls,
                          args=args)

    # [Step 3.5] 验证特征
    verify_embeddings(pdb_ids_ls=pdb_ids_ls, 
                      uniprot_ids_ls=uniprot_ids_ls, 
                      args=args)

    # [Step 4] 构图
    nan_mask = df['ec_numbers'].isna()
    if nan_mask.any():
        logging.warning(f"   [NaN Warning] Dropping {nan_mask.sum()} rows with NaN ec_numbers.")
        df = df[~nan_mask].reset_index(drop=True)

    build_graphs(df, args)
    
    logging.info("==============================================")
    logging.info("       Database Initialization Completed      ")
    logging.info("==============================================")