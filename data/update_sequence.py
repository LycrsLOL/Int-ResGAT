import csv
import os
import re
import multiprocessing as mp

input_file = r"c:\Users\h2406\Desktop\Int_ResGAT\data\database.csv"
output_file = r"c:\Users\h2406\Desktop\Int_ResGAT\data\database.csv"

def parse_xml_file(file_path):
    """
    Worker function to parse an XML file and extract Entry -> Sequence mapping.
    Runs on CPU via multiprocessing.
    """
    seq_dict = {}
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return seq_dict
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Find header row
        row1 = re.search(r'<row r="1">(.*?)</row>', content, re.DOTALL)
        if not row1:
            return seq_dict
            
        entry_col = None
        seq_col = None
        for match in re.finditer(r'<c r="([A-Z]+)1"[^>]*><is><t>(.*?)</t></is></c>', row1.group(1), re.DOTALL):
            col, name = match.groups()
            if name == 'Entry':
                entry_col = col
            elif name == 'Sequence':
                seq_col = col
                
        if not entry_col or not seq_col:
            print(f"[{os.path.basename(file_path)}] Missing 'Entry' or 'Sequence' column. Skipping.")
            return seq_dict
            
        # Extract all rows
        rows = re.findall(r'<row r="\d+">(.*?)</row>', content, re.DOTALL)
        for row_content in rows:
            entry_match = re.search(rf'<c r="{entry_col}\d+"[^>]*><is><t>(.*?)</t></is></c>', row_content, re.DOTALL)
            seq_match = re.search(rf'<c r="{seq_col}\d+"[^>]*><is><t>(.*?)</t></is></c>', row_content, re.DOTALL)
            if entry_match and seq_match:
                entry = entry_match.group(1).strip()
                seq = seq_match.group(1).strip()
                if entry and seq:
                    seq_dict[entry] = seq
                    
        print(f"[{os.path.basename(file_path)}] Extracted {len(seq_dict)} sequences.")
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        
    return seq_dict

def update_row_chunk(args):
    """
    Update a chunk of rows using the sequence cache.
    """
    rows_chunk, seq_cache = args
    updated = []
    for row in rows_chunk:
        uid = row.get('uniprot_id', '').strip()
        # Only update if sequence is currently missing or we found a new one
        if uid and uid in seq_cache:
            if not row.get('sequence'):
                row['sequence'] = seq_cache[uid]
        updated.append(row)
    return updated

def main():
    print("Reading CSV...")
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames)
        if 'sequence' not in fieldnames:
            fieldnames.append('sequence')
        rows = list(reader)

    print(f"Total rows to process: {len(rows)}")
    
    unique_ids = list(set(row.get('uniprot_id', '').strip() for row in rows if row.get('uniprot_id', '').strip()))
    print(f"Total unique UniProt IDs in CSV: {len(unique_ids)}")
    
    xml_files = [
        r"c:\Users\h2406\Desktop\Int_ResGAT\data\sheet1.xml",
        r"c:\Users\h2406\Desktop\Int_ResGAT\data\sheet2.xml"
    ]
    
    # 采用num-workers，数据加载到cpu上
    # Using multiprocessing Pool to parse XML files and process rows in parallel on CPU
    num_workers = min(mp.cpu_count(), 4) # Limit to a reasonable number of workers
    print(f"Starting {num_workers} CPU workers for data loading and processing...")
    
    seq_cache = {}
    with mp.Pool(processes=num_workers) as pool:
        # 1. Parse XML files in parallel
        dicts = pool.map(parse_xml_file, xml_files)
        for d in dicts:
            seq_cache.update(d)
            
    print(f"Total sequences loaded into CPU memory: {len(seq_cache)}")
    
    # Check how many missing sequences can be resolved
    missing_before = sum(1 for r in rows if not r.get('sequence', '').strip())
    print(f"Missing sequences before update: {missing_before}")

    # 2. Update rows in parallel using CPU workers
    chunk_size = max(1, len(rows) // num_workers)
    chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
    
    print("Updating rows...")
    with mp.Pool(processes=num_workers) as pool:
        updated_chunks = pool.map(update_row_chunk, [(chunk, seq_cache) for chunk in chunks])
        
    # Flatten updated chunks
    updated_rows = [row for chunk in updated_chunks for row in chunk]

    missing_after = sum(1 for r in updated_rows if not r.get('sequence', '').strip())
    print(f"Missing sequences after update: {missing_after}")

    print("Writing updated CSV...")
    temp_file = input_file + ".tmp"
    with open(temp_file, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)
        
    os.replace(temp_file, output_file)
    print("Done!")

if __name__ == "__main__":
    main()
