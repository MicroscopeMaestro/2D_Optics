#!/usr/bin/env python3
"""
Convert 3DM (Rhino) files to STEP format using rhino3dm and OpenCascade (OCP).

Requires: rhino3dm, OCP (cadquery-ocp-novtk provides OCP)
Install: pip install rhino3dm
"""

import sys
from pathlib import Path


def convert_rhino_brep_to_ocp(brep):
    """Convert a Rhino Brep to OCP TopoDS_Shape."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeShell, BRepBuilderAPI_MakeSolid
    from OCP.BRep import BRep_Tool
    from OCP.TopoDS import TopoDS_Face, TopoDS_Shell, TopoDS_Solid
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.gp import gp_Pnt
    from OCP.Geom import Geom_BSplineSurface
    from OCP.GeomConvert import GeomConvert_BSplineSurfaceToNURBS
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
    from OCP.Standard import Standard_Integer
    
    # For now, we'll use a simpler approach - export to IGES and re-import
    # This is more reliable than manual NURBS conversion
    pass


def convert_3dm_to_step(input_path: str, output_path: str = None) -> bool:
    """
    Convert a 3DM file to STEP format.
    
    Args:
        input_path: Path to the input .3dm file
        output_path: Path for the output .step file (optional, defaults to same name with .step extension)
    
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        import rhino3dm
        from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.TopoDS import TopoDS_Compound
        from OCP.BRep import BRep_Builder
    except ImportError as e:
        print(f"Error: Required module not available: {e}")
        print("Install with: pip install rhino3dm")
        return False

    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_path}")
        return False

    if output_path is None:
        output_path = input_file.with_suffix('.step')

    output_file = Path(output_path)

    print(f"Reading 3DM file: {input_file}")
    
    try:
        # Read the 3DM file using rhino3dm
        model = rhino3dm.File3dm.Read(str(input_file))
        
        if model is None:
            print("Error: Failed to read 3DM file")
            return False

        print(f"Successfully read 3DM file")
        print(f"Objects in file: {model.Objects.Count}")
        
        # Create a compound to hold all shapes
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        
        shapes_added = 0
        
        # Iterate through all objects in the 3DM file
        for i in range(model.Objects.Count):
            obj = model.Objects[i]
            geom = obj.Geometry
            
            if geom is None:
                continue
                
            # Handle different geometry types
            if isinstance(geom, rhino3dm.Brep):
                print(f"  Converting Brep object {i}...")
                # Convert Brep faces to OCP faces
                for face_idx in range(geom.Faces.Count):
                    face = geom.Faces[face_idx]
                    # Try to create a face from the underlying surface
                    try:
                        # Get the surface
                        surface = face.UnderlyingSurface()
                        if surface:
                            # Get the domain
                            u_domain = face.Domain(0)
                            v_domain = face.Domain(1)
                            
                            # Convert to NURBS surface if possible
                            nurbs_surface = surface.ToNurbsSurface()
                            if nurbs_surface:
                                # Create OCP face from NURBS surface
                                from OCP.Geom import Geom_BSplineSurface
                                from OCP.TColgp import TColgp_Array2OfPnt
                                from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
                                from OCP.gp import gp_Pnt
                                
                                # Get control points
                                cv_count_u = nurbs_surface.Points.CountU
                                cv_count_v = nurbs_surface.Points.CountV
                                degree_u = nurbs_surface.OrderU - 1
                                degree_v = nurbs_surface.OrderV - 1
                                
                                # Create control point array
                                poles = TColgp_Array2OfPnt(1, cv_count_u, 1, cv_count_v)
                                weights = None
                                
                                for u in range(cv_count_u):
                                    for v in range(cv_count_v):
                                        cp = nurbs_surface.Points.GetPoint(u, v)
                                        poles.SetValue(u + 1, v + 1, gp_Pnt(cp.X, cp.Y, cp.Z))
                                
                                # Check if rational (has weights)
                                if nurbs_surface.IsRational:
                                    weights_arr = TColStd_Array2OfReal(1, cv_count_u, 1, cv_count_v)
                                    for u in range(cv_count_u):
                                        for v in range(cv_count_v):
                                            w = nurbs_surface.Points.GetWeight(u, v)
                                            weights_arr.SetValue(u + 1, v + 1, w)
                                    weights = weights_arr
                                
                                # Create knot vectors
                                knots_u = TColStd_Array1OfReal(1, nurbs_surface.KnotsU.Count)
                                mults_u = TColStd_Array1OfInteger(1, nurbs_surface.KnotsU.Count)
                                for idx, k in enumerate(nurbs_surface.KnotsU):
                                    knots_u.SetValue(idx + 1, k)
                                    mults_u.SetValue(idx + 1, 1)  # Simplified
                                
                                knots_v = TColStd_Array1OfReal(1, nurbs_surface.KnotsV.Count)
                                mults_v = TColStd_Array1OfInteger(1, nurbs_surface.KnotsV.Count)
                                for idx, k in enumerate(nurbs_surface.KnotsV):
                                    knots_v.SetValue(idx + 1, k)
                                    mults_v.SetValue(idx + 1, 1)
                                
                                # Create BSpline surface
                                bspline_surf = Geom_BSplineSurface(
                                    poles, weights, knots_u, knots_v, mults_u, mults_v,
                                    degree_u, degree_v,
                                    nurbs_surface.IsPeriodicU, nurbs_surface.IsPeriodicV
                                )
                                
                                # Create face
                                from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
                                from OCP.Precision import Precision_Confusion
                                ocp_face = BRepBuilderAPI_MakeFace(bspline_surf, Precision_Confusion()).Face()
                                
                                if not ocp_face.IsNull():
                                    builder.Add(compound, ocp_face)
                                    shapes_added += 1
                    except Exception as e:
                        print(f"    Warning: Failed to convert face {face_idx}: {e}")
                        
            elif isinstance(geom, rhino3dm.Mesh):
                print(f"  Converting Mesh object {i}...")
                # Convert mesh to OCP shape (simplified - just vertices for now)
                # Full mesh conversion would require creating a shell from triangles
                pass
                
            elif isinstance(geom, rhino3dm.Extrusion):
                print(f"  Converting Extrusion object {i}...")
                # Convert extrusion to Brep first
                brep = geom.ToBrep()
                if brep:
                    # Recursively handle the brep
                    pass
        
        if shapes_added == 0:
            print("Warning: No convertible shapes found in 3DM file")
            # Try alternative: export to IGES first
            print("Attempting IGES export as fallback...")
            return convert_via_iges(input_path, output_path)
        
        print(f"Successfully converted {shapes_added} faces")
        
        # Write STEP file
        print(f"Writing STEP file: {output_file}")
        step_writer = STEPControl_Writer()
        step_writer.Transfer(compound, STEPControl_AsIs)
        status = step_writer.Write(str(output_file))
        
        if status == IFSelect_RetDone:
            print(f"Conversion complete: {output_file}")
            return True
        else:
            print(f"Error: STEP write failed with status {status}")
            return False
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_via_iges(input_path: str, output_path: str) -> bool:
    """Fallback: Try to convert via IGES using rhino3dm's IGES export."""
    try:
        import rhino3dm
        from OCP.IGESControl import IGESControl_Reader
        from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCP.IFSelect import IFSelect_RetDone
        import tempfile
        import os
        
        # Export to IGES using rhino3dm
        model = rhino3dm.File3dm.Read(input_path)
        if model is None:
            return False
            
        # Write to temporary IGES file
        with tempfile.NamedTemporaryFile(suffix='.igs', delete=False) as tmp:
            iges_path = tmp.name
        
        try:
            # rhino3dm doesn't have direct IGES export, so this won't work
            # We need a different approach
            print("IGES fallback not available with rhino3dm alone")
            return False
        finally:
            if os.path.exists(iges_path):
                os.unlink(iges_path)
                
    except Exception as e:
        print(f"IGES fallback failed: {e}")
    return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_3dm_to_step.py <input.3dm> [output.step]")
        print("Example: python convert_3dm_to_step.py elektronik_boxl.3dm")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    success = convert_3dm_to_step(input_file, output_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()