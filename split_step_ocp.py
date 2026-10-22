#!/usr/bin/env python3
"""
Split a STEP file into individual parts using OCP (OpenCascade).
Reads the STEP file, extracts each top-level shape, and writes separate STEP files.
"""

import sys
from pathlib import Path

from OCP.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCP.IFSelect import IFSelect_RetDone
from OCP.TopoDS import TopoDS_Iterator, TopoDS_Compound
from OCP.BRep import BRep_Builder
from OCP.TopAbs import TopAbs_ShapeEnum


def split_step_file(input_path, output_dir=None):
    """Split a STEP file into individual parts."""
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: File not found: {input_path}")
        return False
    
    if output_dir is None:
        output_dir = input_file.parent / f"{input_file.stem}_parts_ocp"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading STEP file: {input_file}")
    
    # Read the STEP file
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(input_file))
    
    if status != IFSelect_RetDone:
        print(f"Error: Failed to read STEP file (status: {status})")
        return False
    
    # Transfer roots
    reader.TransferRoots()
    print(f"Number of roots: {reader.NbRootsForTransfer()}")
    print(f"Number of shapes: {reader.NbShapes()}")
    
    # Get the main compound
    main_shape = reader.Shape(1)
    print(f"Main shape type: {main_shape.ShapeType()}")
    
    # Iterate through top-level parts
    it = TopoDS_Iterator(main_shape)
    part_count = 0
    
    while it.More():
        part_shape = it.Value()
        part_count += 1
        
        print(f"\nProcessing part {part_count}: {part_shape.ShapeType()}")
        
        # Create a compound for this part (in case it has multiple shells)
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        
        # Add all sub-shapes (shells) to the compound
        it2 = TopoDS_Iterator(part_shape)
        shell_count = 0
        while it2.More():
            shell = it2.Value()
            builder.Add(compound, shell)
            shell_count += 1
            it2.Next()
        
        print(f"  Part {part_count} has {shell_count} shells")
        
        # Write this part as a separate STEP file
        output_file = output_dir / f"{input_file.stem}_part_{part_count:03d}.step"
        
        writer = STEPControl_Writer()
        writer.Transfer(compound, STEPControl_AsIs)
        write_status = writer.Write(str(output_file))
        
        if write_status == IFSelect_RetDone:
            print(f"  Written: {output_file}")
        else:
            print(f"  Error writing: {write_status}")
        
        it.Next()
    
    print(f"\nDone! Split into {part_count} parts in: {output_dir}")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python split_step_ocp.py <input.step> [output_dir]")
        print("Example: python split_step_ocp.py c:/Users/Juan/Downloads/elektronik_boxl.stp")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = split_step_file(input_file, output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()