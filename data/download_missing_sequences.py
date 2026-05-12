import csv
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

input_file = r"c:\Users\h2406\Desktop\Int_ResGAT\data\database.csv"
output_file = r"c:\Users\h2406\Desktop\Int_ResGAT\data\database.csv"

def fetch_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            lines = response.text.strip().split('\n')
            if len(lines) > 1:
                return uniprot_id, "".join(lines[1:])
    except Exception as e:
        pass
    return uniprot_id, None

def main():
    print("Reading CSV...")
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames)
        if 'sequence' not in fieldnames:
            fieldnames.append('sequence')
        rows = list(reader)

    missing_ids = list(set(row.get('uniprot_id', '').strip() for row in rows 
                           if not row.get('sequence', '').strip() and row.get('uniprot_id', '').strip()))
    
    print(f"Total unique missing UniProt IDs to fetch: {len(missing_ids)}")

    seq_cache = {}
    
    print("Fetching sequences from UniProt...")
    # Using 20 threads to fetch concurrently
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_sequence, uid): uid for uid in missing_ids}
        
        count = 0
        for future in as_completed(futures):
            uid, seq = future.result()
            if seq:
                seq_cache[uid] = seq
            count += 1
            if count % 100 == 0:
                print(f"Fetched {count}/{len(missing_ids)}...")
                
    print(f"Successfully fetched {len(seq_cache)} sequences.")

    print("Updating rows...")
    updated_rows = []
    for row in rows:
        uid = row.get('uniprot_id', '').strip()
        if not row.get('sequence', '').strip() and uid in seq_cache:
            row['sequence'] = seq_cache[uid]
        updated_rows.append(row)

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
