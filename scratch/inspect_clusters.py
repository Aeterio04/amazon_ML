import pandas as pd

gt = pd.read_csv('data/student_resource/dataset/train/train_ground_truth.tsv', sep='\t', nrows=100)
s1 = pd.read_csv('data/student_resource/dataset/train/train_source1.tsv', sep='\t')
s2 = pd.read_csv('data/student_resource/dataset/train/train_source2.tsv', sep='\t')
s3 = pd.read_csv('data/student_resource/dataset/train/train_source3.tsv', sep='\t')

s1_map = s1.set_index('entity_id').to_dict('index')
s2_map = s2.set_index('entity_id').to_dict('index')
s3_map = s3.set_index('entity_id').to_dict('index')

print('=' * 80)
print('SAMPLE TRUE MATCHING CLUSTERS IN TRAINING DATA')
print('=' * 80)

shown = 0
for _, row in gt.iterrows():
    sid = row['source1_entity_id']
    m = row['matched_entity_ids']
    if pd.isna(m) or not m:
        continue
    m_ids = [x.strip() for x in m.split(',')]
    if len(m_ids) < 2:
        continue
    
    print(f"\n--- S1: {sid} ---")
    r1 = s1_map.get(sid, {})
    print(f"  [S1] Name: {r1.get('business_name')} | Addr: {r1.get('business_address')} | Country: {r1.get('country')}")
    
    for mid in m_ids:
        if mid.startswith('S2'):
            r = s2_map.get(mid, {})
            print(f"  [S2: {mid}] Name: {r.get('business_name')} | Addr: {r.get('business_address')}")
        elif mid.startswith('S3'):
            r = s3_map.get(mid, {})
            print(f"  [S3: {mid}] Name: {r.get('business_name')} | Addr: {r.get('business_address')}")
            
    shown += 1
    if shown >= 5:
        break
