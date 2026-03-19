import os
import shutil
import logging
import subprocess
import requests
import numpy as np
import pandas as pd
import torch
import multiprocessing as mp
import signal
import sys
from sklearn.preprocessing import MultiLabelBinarizer
from Bio import Align
from Bio.Align import substitution_matrices
from Bio.PDB import MMCIFParser, PDBIO, Superimposer, PPBuilder, PDBParser, Model, Structure
from esm.sdk.api import ESMProtein, SamplingConfig
from torch_geometric.data import Data


def _valid_pid(pid: str) -> bool:
    return str(pid).strip().lower() not in ['nan', 'none', '']

def has_valid_pdb_id(pdb_id) -> bool:
    return str(pdb_id).strip().lower() not in ['nan', 'none', '']

# =============================================================================
# [Safe Mode] 进程防僵尸管控系统
# =============================================================================
def _worker_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def run_safe_parallel(worker_func, tasks, num_workers, desc="", print_freq=100, verbose=True):
    total_tasks = len(tasks)
    if total_tasks == 0: return []
    if num_workers <= 1:
        results = []
        for i, t in enumerate(tasks):
            results.append(worker_func(t))
            if verbose and (i + 1) % print_freq == 0:
                logging.info(f"   >> [{desc}] Progress: {i + 1}/{total_tasks}")
        return results

    ctx = mp.get_context('spawn')
    pool = ctx.Pool(processes=num_workers, initializer=_worker_init)
    results = []
    try:
        for i, res in enumerate(pool.imap_unordered(worker_func, tasks)):
            results.append(res)
            if verbose and (i + 1) % print_freq == 0:
                logging.info(f"   >> [{desc}] Progress: {i + 1}/{total_tasks}")
        pool.close()
        pool.join()
        return results
    except KeyboardInterrupt:
        logging.error(f"\n[安全机制触发] 捕获到 Ctrl+C！正在安全杀死 [{desc}] 的所有子进程，防止资源泄露...")
        pool.terminate()
        pool.join()
        sys.exit(1)
    except Exception as e:
        pool.terminate()
        pool.join()
        raise e

# =============================================================================
# [Step 1] Download Structures (多进程)
# =============================================================================
def _single_download(task):
    p_id, u_id, cryst_dir, pred_dir = task
    def _dl(urls, out_path):
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0: return True
        for url in urls:
            try:
                if requests.head(url, timeout=5).status_code == 200:
                    with open(out_path, "wb") as f: f.write(requests.get(url, timeout=30).content)
                    return True
            except: continue
        return False
        
    c_ok = _dl([f"https://www.ebi.ac.uk/pdbe/entry-files/download/{p_id}.cif", f"https://files.rcsb.org/download/{p_id}.cif"], 
               os.path.join(cryst_dir, f'{p_id}_{u_id}.cif')) if _valid_pid(p_id) else False
    p_ok = _dl([f"https://alphafold.ebi.ac.uk/files/AF-{u_id}-F1-model_v4.cif", f"https://swissmodel.expasy.org/repository/uniprot/{u_id}.cif"], 
               os.path.join(pred_dir, f'{u_id}.cif'))
    return 1 if (c_ok or p_ok) else 0

def download_structures(df: pd.DataFrame, args):
    logging.info("========== [Step 1] Download Structures (Parallel) ==========")
    pred_dir, cryst_dir = os.path.join(args.save_dir, 'predicted'), os.path.join(args.save_dir, 'crystal')
    os.makedirs(pred_dir, exist_ok=True); os.makedirs(cryst_dir, exist_ok=True)
    
    tasks = [(str(row['pdb_id']).strip(), str(row['uniprot_id']).strip(), cryst_dir, pred_dir) for _, row in df.iterrows()]
    results = run_safe_parallel(_single_download, tasks, args.num_workers, desc="下载", print_freq=args.print_freq, verbose=args.verbose)
    logging.info(f"   >>> Download Finished. Total Usable: {sum(results)}/{len(df)}")

# =============================================================================
# [Step 2] Complete Structures (多进程)
# =============================================================================
def _run_scwrl(in_pdb, out_pdb):
    try: return subprocess.run(["Scwrl4", "-i", in_pdb, "-o", out_pdb], capture_output=True).returncode == 0
    except: return False

def _run_openmm(in_pdb, out_pdb):
    try:
        import openmm.app as app, openmm as mm, openmm.unit as unit
        pdb = app.PDBFile(in_pdb)
        modeller = app.Modeller(pdb.topology, pdb.positions)
        ff = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
        modeller.addHydrogens(ff, pH=7.0)
        sys = ff.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
        force = mm.CustomExternalForce("10.0*periodicdistance(x, y, z, x0, y0, z0)^2")
        for p in ["x0", "y0", "z0"]: force.addPerParticleParameter(p)
        for atom in modeller.topology.atoms():
            if atom.name in {'N', 'CA', 'C', 'O'}: force.addParticle(atom.index, modeller.positions[atom.index])
        sys.addForce(force)
        sim = app.Simulation(modeller.topology, sys, mm.LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds))
        sim.context.setPositions(modeller.positions)
        sim.minimizeEnergy(maxIterations=100)
        with open(out_pdb, 'w') as f: app.PDBFile.writeFile(sim.topology, sim.context.getState(getPositions=True).getPositions(), f)
    except: shutil.copy(in_pdb, out_pdb)

def _single_completion(task):
    p_id, u_id, save_dir, pred_dir, cryst_dir = task
    final_pdb = os.path.join(save_dir, f"{u_id}_{p_id}.pdb")
    if os.path.exists(final_pdb) and os.path.getsize(final_pdb) > 0: return 0 

    af_path, sifts_path = os.path.join(pred_dir, f"{u_id}.cif"), os.path.join(cryst_dir, f"{p_id}_{u_id}.cif")
    if not (os.path.exists(af_path) and os.path.exists(sifts_path)): return 0

    try:
        cif_parser, ppb, io = MMCIFParser(QUIET=True), PPBuilder(), PDBIO()
        aligner = Align.PairwiseAligner(mode='local', open_gap_score=-10, extend_gap_score=-0.5)
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        
        af_peps = ppb.build_peptides(cif_parser.get_structure(u_id, af_path))
        if not af_peps: return 0
        af_seq, af_chain = str(af_peps[0].get_sequence()), af_peps[0][0].get_parent()

        best_score, sifts_seq, sifts_chain = -float('inf'), "", None
        for chain in cif_parser.get_structure(p_id, sifts_path).get_chains():
            for pep in ppb.build_peptides(chain):
                if len(seq := str(pep.get_sequence())) >= 5 and (score := aligner.score(af_seq, seq)) > best_score:
                    best_score, sifts_seq, sifts_chain = score, seq, pep[0].get_parent()
        
        if not sifts_chain or not (alns := aligner.align(af_seq, sifts_seq)): return 0
        
        af_res, sifts_res = list(af_chain.get_residues()), list(sifts_chain.get_residues())
        fixed, moving = [], []
        for ar, sr in zip(*alns[0].aligned):
            for idx in range(ar[1] - ar[0]):
                if af_res[ar[0]+idx].has_id('CA') and sifts_res[sr[0]+idx].has_id('CA'):
                    moving.append(af_res[ar[0]+idx]['CA']); fixed.append(sifts_res[sr[0]+idx]['CA'])

        if len(moving) < 3: return 0
        Superimposer().set_atoms(fixed, moving)
        Superimposer().apply(af_chain.get_atoms())
        for ar, sr in zip(*alns[0].aligned):
            for idx in range(ar[1] - ar[0]):
                for atom in ['N', 'CA', 'C', 'O']:
                    if af_res[ar[0]+idx].has_id(atom) and sifts_res[sr[0]+idx].has_id(atom):
                        af_res[ar[0]+idx][atom].set_coord(sifts_res[sr[0]+idx][atom].get_coord())
        
        out_model, out_struct = Model.Model(0), Structure.Structure("OUT")
        af_chain.detach_parent(); out_model.add(af_chain); out_struct.add(out_model)
        temp_path = os.path.join(save_dir, f"{u_id}_{p_id}_temp.pdb")
        packed_path = os.path.join(save_dir, f"{u_id}_{p_id}_packed.pdb")
        io.set_structure(out_struct); io.save(temp_path)
        next_in = packed_path if _run_scwrl(temp_path, packed_path) else temp_path
        _run_openmm(next_in, final_pdb)
        for p in [temp_path, packed_path]: 
            if os.path.exists(p): os.remove(p)
        return 1
    except: return 0

def complete_structures(df: pd.DataFrame, args):
    logging.info("========== [Step 2] Complete Structures (Parallel) ==========")
    save_dir, pred_dir, cryst_dir = [os.path.join(args.save_dir, d) for d in ("complete", "predicted", "crystal")]
    os.makedirs(save_dir, exist_ok=True)
    
    tasks = [(str(r['pdb_id']).strip(), str(r['uniprot_id']).strip(), save_dir, pred_dir, cryst_dir) for _, r in df.iterrows() if _valid_pid(r['pdb_id'])]
    results = run_safe_parallel(_single_completion, tasks, args.num_workers, desc="补全", print_freq=args.print_freq, verbose=args.verbose)
    logging.info(f"   >>> Completion Finished. New Processed: {sum(results)}")

# =============================================================================
# [Step 3] Extract Embeddings
# =============================================================================
def extract_embedding(model, pdb_ids_ls, uniprot_ids_ls, args):
    complete_dir = os.path.join(args.save_dir, "complete")
    predicted_dir = os.path.join(args.save_dir, "predicted")
    embedding_dir = os.path.join(args.save_dir, "embedding")
    os.makedirs(embedding_dir, exist_ok=True)
    
    pdb_parser = PDBParser(QUIET=True)
    cif_parser = MMCIFParser(QUIET=True)
    ppb = PPBuilder()
    
    count_processed = 0
    total_tasks = len(uniprot_ids_ls)

    with torch.no_grad():
        for i, (pdb_id_raw, uniprot_id_raw) in enumerate(zip(pdb_ids_ls, uniprot_ids_ls)):
            pdb_id = str(pdb_id_raw).strip()
            uniprot_id = str(uniprot_id_raw).strip()
            has_pdb = has_valid_pdb_id(pdb_id)
            base_name = f"{uniprot_id}_{pdb_id}" if has_pdb else uniprot_id
            save_path = os.path.join(embedding_dir, f"{base_name}.pt")
            
            if os.path.exists(save_path):
                try:
                    emb = torch.load(save_path, map_location='cpu')
                    is_valid = True
                    # Check dimensions
                    if emb.dim() != 2: 
                        is_valid = False
                    # Check for all zeros
                    elif (emb == 0).all(): 
                        is_valid = False
                    # Check for NaN/Inf
                    elif torch.isnan(emb).any() or torch.isinf(emb).any(): 
                        is_valid = False
                    # Check for constant sequence (variance check)
                    elif emb.size(0) > 1 and torch.var(emb, dim=0, unbiased=False).sum().item() < 1e-6: 
                        is_valid = False
                    
                    if is_valid:
                        if args.verbose and (i + 1) % args.print_freq == 0:
                            logging.info(f"   >> Progress: {i + 1}/{total_tasks} (Skipping existing)")
                        continue
                    else:
                        logging.warning(f"   [Overwrite] {base_name} exists but is invalid. Re-generating...")
                except Exception:
                    logging.warning(f"   [Overwrite] {base_name} exists but cannot be loaded. Re-generating...")

            complete_path = os.path.join(complete_dir, f"{uniprot_id}_{pdb_id}.pdb") if has_pdb else ""
            predicted_path = os.path.join(predicted_dir, f"{uniprot_id}.cif")

            if complete_path and os.path.exists(complete_path):
                struct_path, parser = complete_path, pdb_parser
            elif os.path.exists(predicted_path):
                struct_path, parser = predicted_path, cif_parser
            else:
                continue
                
            try:
                structure = parser.get_structure(uniprot_id, struct_path)
                peptides = ppb.build_peptides(structure)
                sequence = "".join([str(pp.get_sequence()) for pp in peptides])
                
                if len(sequence) == 0: continue

                protein = ESMProtein(sequence=sequence)
                protein_tensor = model.encode(protein)
                config = SamplingConfig(return_per_residue_embeddings=True)
                output = model.forward_and_sample(protein_tensor, config)
                embeddings = output.per_residue_embedding
                
                if (embeddings == 0).all():
                    logging.warning(f"   [Warning] {uniprot_id}: Raw embeddings (before norm) are ALL ZEROS!")

                if hasattr(model, 'transformer') and hasattr(model.transformer, 'norm'):
                    embeddings_raw = embeddings 
                    
                    embeddings_double = embeddings.double()
                    norm_layer = model.transformer.norm
                    
                    mean = embeddings_double.mean(dim=-1, keepdim=True)
                    var = embeddings_double.var(dim=-1, keepdim=True, unbiased=False)
                    eps = 1e-5
                    
                    x_norm = (embeddings_double - mean) / torch.sqrt(var + eps)
                    
                    if norm_layer.weight is not None:
                        x_norm = x_norm * norm_layer.weight.double()
                    if norm_layer.bias is not None:
                        x_norm = x_norm + norm_layer.bias.double()
                        
                    embeddings = x_norm.float()
                    
                    if (embeddings == 0).all():
                        logging.warning(f"   [Warning] {uniprot_id}: Embeddings became ALL ZEROS even after float64 LayerNorm!")
                
                embeddings = embeddings.cpu()

                is_nan = torch.isnan(embeddings).any()
                is_inf = torch.isinf(embeddings).any()
                max_abs = embeddings.abs().max().item()
                is_huge = max_abs > 1e20
                is_zeros = (embeddings == 0).all()

                if is_nan or is_inf or is_huge or is_zeros:
                    status_msg = []
                    if is_nan: status_msg.append("NaN")
                    if is_inf: status_msg.append("Inf")
                    if is_huge: status_msg.append("Huge")
                    if is_zeros: status_msg.append("Zeros")
                    
                    logging.debug(f"   [Debug] Embedding Issue for {uniprot_id}: {', '.join(status_msg)}")
                    logging.debug(f"      Max: {embeddings.max().item()}, Min: {embeddings.min().item()}")
                    logging.debug(f"      Mean: {embeddings.mean().item()}, Std: {embeddings.std().item()}")
                    if is_zeros:
                        logging.debug(f"      First 10 values: {embeddings.flatten()[:10].tolist()}")
                
                torch.save(embeddings, save_path)
                
                count_processed += 1
                if args.verbose and (i + 1) % args.print_freq == 0:
                    logging.info(f"   >> Progress: {i + 1}/{total_tasks} | Processed: {count_processed}")
                    
            except Exception as e:
                logging.debug(f"   [Debug] Failed to extract embedding for {uniprot_id}: {e}")
                continue

    logging.info(f"   >>> Extraction Finished. Processed: {count_processed} | Total: {total_tasks}")

# =============================================================================
# [Step 3.5] Verify Embeddings
# =============================================================================
def verify_embeddings(pdb_ids_ls, uniprot_ids_ls, args):
    logging.info("========== [Step 3.5] Verify Embeddings ==========")
    
    embedding_dir = os.path.join(args.save_dir, "embedding")
    total_files = len(pdb_ids_ls)
    invalid_files = []
    
    logging.info(f"   Checking {total_files} embedding files in {embedding_dir}...")
    
    count_pass = 0
    count_fail = 0
    first_valid_mean = None
    
    for i, (pdb_id, uniprot_id) in enumerate(zip(pdb_ids_ls, uniprot_ids_ls)):
        pdb_id = str(pdb_id).strip()
        uniprot_id = str(uniprot_id).strip()
        base_name = f"{uniprot_id}_{pdb_id}" if has_valid_pdb_id(pdb_id) else uniprot_id
        file_path = os.path.join(embedding_dir, f"{base_name}.pt")
        
        if not os.path.exists(file_path):
            logging.error(f"   [Missing] {base_name}: File not found.")
            invalid_files.append((base_name, "Missing"))
            count_fail += 1
            continue
            
        try:
            emb = torch.load(file_path, map_location='cpu')
            
            # Check 1: Dimensions
            if emb.dim() != 2:
                 logging.warning(f"   [Invalid Dim] {base_name}: Shape {emb.shape} is not 2D.")
                 invalid_files.append((base_name, f"Invalid Dim {emb.shape}"))
                 count_fail += 1
                 continue
                 
            # Check 2: NaN / Inf
            if torch.isnan(emb).any():
                logging.warning(f"   [NaN Detected] {base_name}: Contains NaN values.")
                invalid_files.append((base_name, "NaN"))
                count_fail += 1
                continue
                
            if torch.isinf(emb).any():
                logging.warning(f"   [Inf Detected] {base_name}: Contains Inf values.")
                invalid_files.append((base_name, "Inf"))
                count_fail += 1
                continue
                
            # Check 3: All Zeros
            if (emb == 0).all():
                logging.warning(f"   [All Zeros] {base_name}: Embedding is all zeros.")
                invalid_files.append((base_name, "All Zeros"))
                count_fail += 1
                continue
            
            # Check 4: Constant Embeddings (Intra-file Model Collapse)
            if emb.size(0) > 1:
                row_var = torch.var(emb, dim=0, unbiased=False).sum().item()
                if row_var < 1e-6:
                     logging.warning(f"   [Constant Sequence] {base_name}: All residues have identical embeddings (Var={row_var:.2e}).")
                     invalid_files.append((base_name, "Constant Sequence"))
                     count_fail += 1
                     continue
            
            # Check 5: Value Range (Extreme Values)
            max_val = emb.abs().max().item()
            if max_val > 100: 
                 if max_val > 1e4:
                     logging.warning(f"   [Extreme Value] {base_name}: Max abs value is {max_val:.2e} (Suspiciously high).")
                     invalid_files.append((base_name, f"Extreme Value {max_val:.2e}"))
                     count_fail += 1
                     continue
            
            # Check 6: Inter-file Diversity (Global Model Collapse)
            current_mean = emb.mean(dim=0)
            if first_valid_mean is None:
                first_valid_mean = current_mean
            else:
                diff = (current_mean - first_valid_mean).abs().sum().item()
                if diff < 1e-6:
                     logging.warning(f"   [Duplicate Content] {base_name}: Embedding is identical to the first processed file (Diff={diff:.2e}).")
                     invalid_files.append((base_name, "Duplicate Content"))
                     count_fail += 1
                     continue

            count_pass += 1
            
            if (i + 1) % 100 == 0:
                logging.info(f"   Checked {i + 1}/{total_files} files...")

        except Exception as e:
            logging.error(f"   [Read Error] {base_name}: {e}")
            invalid_files.append((base_name, f"Read Error: {e}"))
            count_fail += 1
            
    logging.info(f"   >>> Verification Finished. Pass: {count_pass} | Fail: {count_fail}")
    
    if len(invalid_files) > 0:
        logging.error(f"   Found {len(invalid_files)} invalid files:")
        for name, reason in invalid_files[:20]:
            logging.error(f"      - {name}: {reason}")
        if len(invalid_files) > 20:
            logging.error(f"      ... and {len(invalid_files) - 20} more.")
    else:
        logging.info("   All embeddings appear valid and distinct.")

# =============================================================================
# [Step 4] Build Graphs (多进程)
# =============================================================================
def _single_graph(task):
    u_id, p_id, labels_tensor, active_sites_raw, max_edge_distance, dirs = task
    save_dir, comp_dir, pred_dir, emb_dir = dirs
    
    base_name = f"{u_id}_{p_id}" if _valid_pid(p_id) else u_id
    save_path = os.path.join(save_dir, f"{base_name}.pt")
    if os.path.exists(save_path): return 0

    emb_path = os.path.join(emb_dir, f"{base_name}.pt")
    if not os.path.exists(emb_path): return 0

    comp_path = os.path.join(comp_dir, f"{u_id}_{p_id}.pdb") if _valid_pid(p_id) else ""
    pred_path = os.path.join(pred_dir, f"{u_id}.cif")
    struct_path = comp_path if comp_path and os.path.exists(comp_path) else pred_path if os.path.exists(pred_path) else None
    if not struct_path: return 0

    try:
        pdb_parser, cif_parser, ppb = PDBParser(QUIET=True), MMCIFParser(QUIET=True), PPBuilder()
        parser = pdb_parser if struct_path.endswith('.pdb') else cif_parser
        residues = sum([list(pp) for pp in ppb.build_peptides(parser.get_structure(u_id, struct_path))], [])
        
        emb = torch.load(emb_path, map_location='cpu', weights_only=True)
        if isinstance(emb, np.ndarray): emb = torch.from_numpy(emb)
        
        if len(residues) != emb.shape[0]:
            if emb.shape[0] == len(residues) + 2: emb = emb[1:-1]
            elif emb.shape[0] == len(residues) + 1: emb = emb[1:]
            else: return 0

        coords, node_ids = [], []
        for res in residues:
            node_ids.append(f"{res.get_parent().id}_{res.id[1]}")
            coords.append(res['CA'].get_coord() if 'CA' in res else np.array([a.get_coord() for a in res]).mean(axis=0) if len(res)>0 else np.array([0.,0.,0.]))
        
        pos = torch.tensor(np.array(coords), dtype=torch.float, device='cpu')
        dist_matrix = torch.cdist(pos, pos).fill_diagonal_(float('inf'))
        mask = dist_matrix <= max_edge_distance
        
        data = Data(x=emb.cpu().float(), pos=pos, edge_index=mask.nonzero(as_tuple=False).t(), edge_attr=dist_matrix[mask].view(-1, 1))
        data.y = labels_tensor.cpu().unsqueeze(0)
        data.pdb_ids = node_ids

        res_num_to_indices = {res.id[1]: [] for res in residues}
        for i, res in enumerate(residues): res_num_to_indices[res.id[1]].append(i)

        active_idx = []
        if pd.notna(active_sites_raw):
            s_list = [int(s) for s in str(active_sites_raw).replace(';', ',').split(',') if s.strip().lstrip('-').isdigit()] if isinstance(active_sites_raw, str) else [int(active_sites_raw)]
            for s in s_list:
                if s in res_num_to_indices: active_idx.extend(res_num_to_indices[s])
        
        data.active_sites_idx = torch.tensor(list(set(active_idx)), dtype=torch.long, device='cpu')
        torch.save(data.cpu(), save_path)
        return 1
    except Exception: return 0

def build_graphs(df: pd.DataFrame, args):
    logging.info("========== [Step 4] Build Graphs (Parallel) ==========")
    dirs = tuple(os.path.join(args.save_dir, d) for d in ("graph", "complete", "predicted", "embedding"))
    os.makedirs(dirs[0], exist_ok=True)
    
    mlb = MultiLabelBinarizer()
    labels_ls = torch.tensor(mlb.fit_transform([str(i).split(';') for i in df['ec_numbers']]), dtype=torch.float32)

    tasks = [(str(row['uniprot_id']).strip(), str(row['pdb_id']).strip(), labels, row.get('active_sites', []), args.max_edge_distance, dirs) 
             for (idx, row), labels in zip(df.iterrows(), labels_ls)]
    
    results = run_safe_parallel(_single_graph, tasks, args.num_workers, desc="buidling", print_freq=args.print_freq, verbose=args.verbose)
    logging.info(f"   >>> Graph Generation Finished. New Processed: {sum(results)}")
