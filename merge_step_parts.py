#!/usr/bin/env python3
"""
Merge nearby STEP parts based on spatial proximity.
Groups parts by clustering their centroids and merges each cluster into one STEP file.
"""

import sys
import re
from pathlib import Path
from collections import defaultdict
import math


def get_part_centroid(part_file):
    """Extract centroid from a STEP part file."""
    with open(part_file, 'r') as f:
        content = f.read()
    
    # Extract all CARTESIAN_POINT coordinates
    points = re.findall(r'CARTESIAN_POINT\([^)]*\(([^)]+)\)', content)
    coords = []
    for p in points:
        try:
            x, y, z = map(float, p.split(','))
            coords.append((x, y, z))
        except:
            pass
    
    if not coords:
        return None
    
    xs, ys, zs = zip(*coords)
    cx, cy, cz = sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)
    return (cx, cy, cz)


def distance(p1, p2):
    """Euclidean distance between two 3D points."""
    return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))


def cluster_parts(part_files, max_distance=30.0):
    """
    Cluster parts based on centroid proximity using simple agglomerative clustering.
    Returns list of clusters (each cluster is a list of part file paths).
    """
    # Get centroids
    centroids = {}
    for pf in part_files:
        centroid = get_part_centroid(pf)
        if centroid:
            centroids[pf] = centroid
        else:
            print(f"Warning: No centroid for {pf.name}")
    
    if not centroids:
        return []
    
    # Simple clustering: start with each part as its own cluster
    clusters = [[pf] for pf in centroids.keys()]
    
    # Merge clusters that are close
    merged = True
    while merged:
        merged = False
        new_clusters = []
        used = set()
        
        for i, cluster_a in enumerate(clusters):
            if i in used:
                continue
            
            # Find all clusters close to this one
            merged_cluster = list(cluster_a)
            for j, cluster_b in enumerate(clusters):
                if i == j or j in used:
                    continue
                
                # Check if any part in cluster_a is close to any part in cluster_b
                close = False
                for pa in cluster_a:
                    for pb in cluster_b:
                        if distance(centroids[pa], centroids[pb]) < max_distance:
                            close = True
                            break
                    if close:
                        break
                
                if close:
                    merged_cluster.extend(cluster_b)
                    used.add(j)
                    merged = True
            
            new_clusters.append(merged_cluster)
            used.add(i)
        
        clusters = new_clusters
    
    return clusters


def parse_step_entities(filepath):
    """Parse STEP file and return header and entities dict."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split into header and data sections
    header_end = content.find('ENDSEC;\nDATA;')
    if header_end == -1:
        header_end = content.find('ENDSEC;\r\nDATA;')
    if header_end == -1:
        raise ValueError("Invalid STEP file format")
    
    header = content[:header_end + len('ENDSEC;')]
    data_section = content[header_end + len('ENDSEC;\nDATA;'):]
    if data_section.startswith('\r\n'):
        data_section = data_section[2:]
    elif data_section.startswith('\n'):
        data_section = data_section[1:]
    
    # Parse entities - handle both formats: #1=TYPE and #1 = TYPE
    entities = {}
    entity_order = []
    
    pos = 0
    while pos < len(data_section):
        match = re.match(r'(#\d+)\s*=\s*(\w+)\s*\(', data_section[pos:])
        if not match:
            pos += 1
            continue
        
        entity_id = match.group(1)
        entity_type = match.group(2)
        start_pos = pos + match.end() - 1
        
        paren_count = 0
        end_pos = start_pos
        while end_pos < len(data_section):
            if data_section[end_pos] == '(':
                paren_count += 1
            elif data_section[end_pos] == ')':
                paren_count -= 1
                if paren_count == 0:
                    if end_pos + 1 < len(data_section) and data_section[end_pos + 1] == ';':
                        end_pos += 2
                        break
            end_pos += 1
        
        entity_text = data_section[pos:end_pos]
        entities[entity_id] = {
            'type': entity_type,
            'text': entity_text,
            'refs': re.findall(r'(#\d+)', entity_text)
        }
        entity_order.append(entity_id)
        pos = end_pos
    
    return header, entities, entity_order


def merge_step_files(part_files, output_file):
    """Merge multiple STEP part files into one."""
    print(f"Merging {len(part_files)} parts into {output_file}")
    
    # Parse all files
    all_entities = {}
    all_order = []
    header = None
    
    # We need to renumber entities to avoid conflicts
    entity_map = {}  # old_id -> new_id
    next_id = 1
    
    for pf in part_files:
        h, entities, order = parse_step_entities(pf)
        if header is None:
            header = h
        
        # Create mapping for this file's entities
        file_map = {}
        for old_id in order:
            new_id = f"#{next_id}"
            file_map[old_id] = new_id
            next_id += 1
        
        entity_map.update(file_map)
        
        # Remap entity references in this file's entities
        for old_id in order:
            e = entities[old_id]
            new_text = e['text']
            # Replace all references
            for old_ref, new_ref in file_map.items():
                new_text = new_text.replace(old_ref, new_ref)
            
            new_id = file_map[old_id]
            all_entities[new_id] = {
                'type': e['type'],
                'text': new_text,
                'refs': [file_map.get(r, r) for r in e['refs']]
            }
            all_order.append(new_id)
    
    # Write merged file
    with open(output_file, 'w') as f:
        f.write(header + '\nDATA;\n')
        for eid in all_order:
            f.write(all_entities[eid]['text'] + '\n')
        f.write('ENDSEC;\nEND-ISO-10303-21;\n')
    
    print(f"  Merged: {len(all_entities)} entities")


def main():
    if len(sys.argv) < 2:
        print("Usage: python merge_step_parts.py <parts_dir> [output_dir] [max_distance]")
        print("Example: python merge_step_parts.py c:/Users/Juan/Downloads/elektronik_boxl_parts")
        print("  max_distance: maximum distance between centroids to merge (default: 30.0)")
        sys.exit(1)
    
    parts_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else parts_dir.parent / f"{parts_dir.name}_merged"
    max_distance = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    
    part_files = sorted(parts_dir.glob('*.step'))
    if not part_files:
        print(f"No .step files found in {parts_dir}")
        sys.exit(1)
    
    print(f"Found {len(part_files)} parts")
    print(f"Clustering with max_distance={max_distance}")
    
    # Get centroids for display
    for pf in part_files:
        centroid = get_part_centroid(pf)
        if centroid:
            print(f"  {pf.name}: centroid=({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")
    
    # Cluster
    clusters = cluster_parts(part_files, max_distance)
    print(f"\nFormed {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters):
        names = [p.name for p in cluster]
        print(f"  Cluster {i+1}: {names}")
    
    # Merge each cluster
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, cluster in enumerate(clusters):
        output_file = output_dir / f"merged_cluster_{i+1:02d}.step"
        merge_step_files(cluster, output_file)
    
    print(f"\nDone! Merged files in: {output_dir}")


if __name__ == "__main__":
    main()