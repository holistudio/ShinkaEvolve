import argparse
import numpy as np
from pathlib import Path
import rhino3dm as rh


def load_packing_data(results_dir):
    """Load packing data from the results directory."""
    extra_file = Path(results_dir) / "extra.npz"
    
    if not extra_file.exists():
        raise FileNotFoundError(f"Could not find {extra_file}")
    
    data = np.load(extra_file)
    return {
        'positions': data['positions'],
        'dimensions': data['dimensions'],
        'rotations': data['rotations'],
        'num_rectangles': int(data['num_rectangles']),
        'combined_score': float(data['combined_score']),
        'total_area': float(data['total_area']),
        'packing_efficiency': float(data['packing_efficiency']),
    }


def create_rectangle_polyline(x, y, width, height):
    """
    Create a closed polyline representing a rectangle in the XY plane.
    
    Args:
        x, y: Bottom-left corner coordinates
        width, height: Rectangle dimensions
        
    Returns:
        rhino3dm.PolylineCurve
    """
    # Create rectangle corners (counter-clockwise from bottom-left)
    points = [
        rh.Point3d(x, y, 0),              # Bottom-left
        rh.Point3d(x + width, y, 0),      # Bottom-right
        rh.Point3d(x + width, y + height, 0),  # Top-right
        rh.Point3d(x, y + height, 0),     # Top-left
        rh.Point3d(x, y, 0)               # Close the polyline
    ]
    
    # Create polyline curve
    polyline = rh.Polyline(points)
    polyline_curve = rh.PolylineCurve(polyline)
    
    return polyline_curve


def create_rectangle_surface(x, y, width, height):
    """
    Create a planar surface for the rectangle.
    
    Args:
        x, y: Bottom-left corner coordinates
        width, height: Rectangle dimensions
        
    Returns:
        rhino3dm.PlaneSurface
    """
    # Create corner points
    corner = rh.Point3d(x, y, 0)
    
    # Create plane at the corner
    plane = rh.Plane(corner, rh.Vector3d.ZAxis)
    
    # Create interval for width and height
    x_interval = rh.Interval(0, width)
    y_interval = rh.Interval(0, height)
    
    # Create plane surface
    surface = rh.PlaneSurface(plane, x_interval, y_interval)
    
    return surface


def export_to_3dm(data, output_path, include_surfaces=True, include_curves=True):
    """
    Export rectangle packing to Rhino 3DM file.
    
    Args:
        data: Dictionary with packing data
        output_path: Path to save 3DM file
        include_surfaces: Include rectangle surfaces
        include_curves: Include rectangle boundary curves
    """
    positions = data['positions']
    dimensions = data['dimensions']
    rotations = data['rotations']
    num_rectangles = data['num_rectangles']
    
    # Create a new 3DM file
    model = rh.File3dm()
    
    # Set units to meters (you can change this)
    model.Settings.ModelUnitSystem = rh.UnitSystem.Meters
    
    # Create layers for organization
    layer_curves = rh.Layer()
    layer_curves.Name = "Rectangle_Curves"
    layer_curves.Color = (0, 0, 0, 255)  # Black
    curves_layer_index = model.Layers.Add(layer_curves)
    
    layer_surfaces = rh.Layer()
    layer_surfaces.Name = "Rectangle_Surfaces"
    layer_surfaces.Color = (200, 200, 200, 255)  # Light gray
    surfaces_layer_index = model.Layers.Add(layer_surfaces)
    
    layer_boundary = rh.Layer()
    layer_boundary.Name = "Unit_Square_Boundary"
    layer_boundary.Color = (255, 0, 0, 255)  # Red
    boundary_layer_index = model.Layers.Add(layer_boundary)
    
    # Create color palette for rectangles
    colors = [
        (255, 100, 100, 255),  # Red
        (100, 255, 100, 255),  # Green
        (100, 100, 255, 255),  # Blue
        (255, 255, 100, 255),  # Yellow
        (255, 100, 255, 255),  # Magenta
        (100, 255, 255, 255),  # Cyan
        (255, 150, 100, 255),  # Orange
        (150, 100, 255, 255),  # Purple
        (100, 255, 150, 255),  # Lime
    ]
    
    # Add unit square boundary
    boundary_curve = create_rectangle_polyline(0, 0, 1, 1)
    boundary_attrs = rh.ObjectAttributes()
    boundary_attrs.LayerIndex = boundary_layer_index
    boundary_attrs.ColorSource = rh.ObjectColorSource.ColorFromLayer
    model.Objects.AddCurve(boundary_curve, boundary_attrs)
    
    # Add each rectangle
    for i in range(num_rectangles):
        x, y = positions[i]
        w, h = dimensions[i]
        
        # Apply rotation if needed
        if rotations[i]:
            w, h = h, w  # Swap for 90° rotation
        
        # Create geometry
        if include_curves:
            curve = create_rectangle_polyline(x, y, w, h)
            curve_attrs = rh.ObjectAttributes()
            curve_attrs.LayerIndex = curves_layer_index
            curve_attrs.ObjectColor = colors[i % len(colors)]
            curve_attrs.ColorSource = rh.ObjectColorSource.ColorFromObject
            curve_attrs.Name = f"Rectangle_{i+1}_Curve"
            model.Objects.AddCurve(curve, curve_attrs)
        
        if include_surfaces:
            surface = create_rectangle_surface(x, y, w, h)
            surface_attrs = rh.ObjectAttributes()
            surface_attrs.LayerIndex = surfaces_layer_index
            surface_attrs.ObjectColor = colors[i % len(colors)]
            surface_attrs.ColorSource = rh.ObjectColorSource.ColorFromObject
            surface_attrs.Name = f"Rectangle_{i+1}_Surface"
            model.Objects.AddSurface(surface, surface_attrs)
    
    # Add text labels for each rectangle
    layer_labels = rh.Layer()
    layer_labels.Name = "Rectangle_Labels"
    layer_labels.Color = (0, 0, 0, 255)
    labels_layer_index = model.Layers.Add(layer_labels)
    
    for i in range(num_rectangles):
        x, y = positions[i]
        w, h = dimensions[i]
        
        # Apply rotation if needed
        if rotations[i]:
            w, h = h, w
        
        # Calculate center for label
        cx = x + w / 2
        cy = y + h / 2
        
        # Create text dot
        area = dimensions[i, 0] * dimensions[i, 1]
        rot_str = "90°" if rotations[i] else "0°"
        label_text = f"R{i+1}\nArea: {area:.3f}\nRot: {rot_str}"
        
        text_point = rh.Point3d(cx, cy, 0)
        text_dot = rh.TextDot(label_text, text_point)
        
        text_attrs = rh.ObjectAttributes()
        text_attrs.LayerIndex = labels_layer_index
        text_attrs.Name = f"Rectangle_{i+1}_Label"
        model.Objects.AddTextDot(text_dot, text_attrs)
    
    # Add notes with metrics
    layer_notes = rh.Layer()
    layer_notes.Name = "Packing_Info"
    layer_notes.Color = (0, 0, 0, 255)
    notes_layer_index = model.Layers.Add(layer_notes)
    
    info_text = (
        f"Rectangle Packing Results\n"
        f"Rectangles: {num_rectangles}\n"
        f"Total Area: {data['total_area']:.4f}\n"
        f"Efficiency: {data['packing_efficiency']:.4f}\n"
        f"Score: {data['combined_score']:.4f}"
    )
    
    info_location = rh.Point3d(1.1, 0.9, 0)
    info_dot = rh.TextDot(info_text, info_location)
    
    info_attrs = rh.ObjectAttributes()
    info_attrs.LayerIndex = notes_layer_index
    info_attrs.Name = "Packing_Metrics"
    model.Objects.AddTextDot(info_dot, info_attrs)
    
    # Write the 3DM file
    model.Write(str(output_path), version=7)  # Rhino 7 format
    print(f"✓ 3DM file saved to: {output_path}")
    print(f"  - {num_rectangles} rectangles")
    print(f"  - Layers: Curves, Surfaces, Labels, Boundary, Info")
    print(f"  - File size: {Path(output_path).stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Export rectangle packing to Rhino 3DM file"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing packing results (extra.npz)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rectangle_packing.3dm",
        help="Output 3DM file path"
    )
    parser.add_argument(
        "--no-surfaces",
        action="store_true",
        help="Don't include rectangle surfaces (curves only)"
    )
    parser.add_argument(
        "--no-curves",
        action="store_true",
        help="Don't include rectangle curves (surfaces only)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load packing data
        print(f"Loading packing data from {args.results_dir}...")
        data = load_packing_data(args.results_dir)
        
        print(f"\nPacking Summary:")
        print(f"  Rectangles: {data['num_rectangles']}")
        print(f"  Total Area: {data['total_area']:.4f}")
        print(f"  Efficiency: {data['packing_efficiency']:.4f}")
        print(f"  Score: {data['combined_score']:.4f}")
        
        # Export to 3DM
        print(f"\nExporting to 3DM...")
        export_to_3dm(
            data,
            args.output,
            include_surfaces=not args.no_surfaces,
            include_curves=not args.no_curves
        )
        
        print(f"\n✓ Export complete!")
        print(f"\nTo view: Open '{args.output}' in Rhino")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run evaluate.py first to generate results.")
    except Exception as e:
        print(f"Error exporting to 3DM: {e}")
        raise


if __name__ == "__main__":
    main()