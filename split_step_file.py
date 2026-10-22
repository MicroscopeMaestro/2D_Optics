#!/usr/bin/env python3
"""
Split a STEP file containing multiple parts into separate STEP files.
Each FACETED_BREP / CLOSED_SHELL becomes its own STEP file.
"""

import sys
import re
from pathlib import Path
from collections import defaultdict


def parse_step_file(filepath):
    """Parse STEP file and return header, data entities, and entity references."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split into header and data sections
    header_end = content.find('ENDSEC;\nDATA;')
    if header_end == -1:
        raise ValueError("Invalid STEP file format")
    
    header = content[:header_end + len('ENDSEC;')]
    data_section = content[header_end + len('ENDSEC;\nDATA;'):]
    
    # Parse entities: #NUM=TYPE(...)
    entity_pattern = r'(#\d+)=(\w+)\((.*?)\);'
    entities = {}
    entity_order = []
    
    # Use a more careful parsing approach for nested parentheses
    pos = 0
    while pos < len(data_section):
        match = re.match(r'(#\d+)=(\w+)\(', data_section[pos:])
        if not match:
            pos += 1
            continue
        
        entity_id = match.group(1)
        entity_type = match.group(2)
        start_pos = pos + match.end() - 1  # Position of opening paren
        
        # Find matching closing paren + semicolon
        paren_count = 0
        end_pos = start_pos
        while end_pos < len(data_section):
            if data_section[end_pos] == '(':
                paren_count += 1
            elif data_section[end_pos] == ')':
                paren_count -= 1
                if paren_count == 0:
                    # Check for semicolon
                    if end_pos + 1 < len(data_section) and data_section[end_pos + 1] == ';':
                        end_pos += 2  # Include semicolon
                        break
            end_pos += 1
        
        entity_text = data_section[pos:end_pos]
        entities[entity_id] = {
            'type': entity_type,
            'text': entity_text,
            'refs': extract_refs(entity_text)
        }
        entity_order.append(entity_id)
        pos = end_pos
    
    return header, entities, entity_order


def extract_refs(text):
    """Extract all entity references (#123) from entity text."""
    return re.findall(r'(#\d+)', text)


def find_connected_entities(entities, start_ids):
    """Find all entities connected to the given start IDs (transitive closure)."""
    connected = set(start_ids)
    queue = list(start_ids)
    
    while queue:
        current = queue.pop(0)
        if current in entities:
            for ref in entities[current]['refs']:
                if ref not in connected:
                    connected.add(ref)
                    queue.append(ref)
    
    return connected


def write_step_file(output_path, header, entities, entity_ids, entity_order):
    """Write a STEP file with the given entities."""
    # Filter and sort entities by original order
    filtered = [eid for eid in entity_order if eid in entity_ids]
    
    with open(output_path, 'w') as f:
        f.write(header + '\nDATA;\n')
        for eid in filtered:
            f.write(entities[eid]['text'] + '\n')
        f.write('ENDSEC;\nEND-ISO-10303-21;\n')


def split_step_file(input_path, output_dir=None):
    """Split a STEP file into separate files for each top-level shape."""
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: File not found: {input_path}")
        return False
    
    if output_dir is None:
        output_dir = input_file.parent / f"{input_file.stem}_parts"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Parsing STEP file: {input_file}")
    header, entities, entity_order = parse_step_file(input_path)
    print(f"Total entities: {len(entities)}")
    
    # Find all FACETED_BREP entities (top-level shapes)
    faceted_breps = [eid for eid, e in entities.items() if e['type'] == 'FACETED_BREP']
    print(f"Found {len(faceted_breps)} FACETED_BREP entities: {faceted_breps}")
    
    # Also find CLOSED_SHELL entities
    closed_shells = [eid for eid, e in entities.items() if e['type'] == 'CLOSED_SHELL']
    print(f"Found {len(closed_shells)} CLOSED_SHELL entities: {closed_shells}")
    
    # For each FACETED_BREP, find all connected entities and write separate file
    for i, brep_id in enumerate(faceted_breps):
        print(f"\nProcessing part {i+1}/{len(faceted_breps)}: {brep_id}")
        
        # Get all entities connected to this BREP
        connected = find_connected_entities(entities, [brep_id])
        print(f"  Connected entities: {len(connected)}")
        
        # Write separate STEP file
        output_file = output_dir / f"{input_file.stem}_part_{i+1:03d}.step"
        write_step_file(output_file, header, entities, connected, entity_order)
        print(f"  Written: {output_file}")
    
    print(f"\nDone! Split into {len(faceted_breps)} files in: {output_dir}")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python split_step_file.py <input.step> [output_dir]")
        print("Example: python split_step_file.py elektronik_boxl.stp")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = split_step_file(input_file, output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()