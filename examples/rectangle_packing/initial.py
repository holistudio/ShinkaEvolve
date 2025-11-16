import numpy as np


def construct_packing():
    """
    Construct an arrangement of 3-9 rectangles in a unit square [0,1]x[0,1]
    to maximize packing efficiency with bonus for more rectangles.

    Returns:
        Tuple of (positions, dimensions, rotations, num_rectangles)
        positions: np.array of shape (N, 2) with (x, y) bottom-left corners
        dimensions: np.array of shape (N, 2) with (width, height) for each rectangle
        rotations: np.array of shape (N,) with boolean rotation flags (True = 90° rotated)
        num_rectangles: int, number of rectangles (3-9)
    """
    # Start with a moderate number of rectangles
    num_rectangles = 6
    
    # Generate rectangle dimensions with areas in [0.15, 0.65]
    dimensions = generate_rectangle_dimensions(num_rectangles)
    
    # Initialize positions (simple grid-based approach)
    positions = initialize_positions(num_rectangles)
    
    # Initialize rotations (all False initially)
    rotations = np.zeros(num_rectangles, dtype=bool)
    
    # Apply simple placement strategy
    positions, rotations = place_rectangles(dimensions, positions, rotations)
    
    return positions, dimensions, rotations, num_rectangles


def generate_rectangle_dimensions(n):
    """
    Generate n rectangles with areas between 0.15 and 0.65.
    Uses varied aspect ratios for better packing opportunities.
    
    Args:
        n: number of rectangles to generate
        
    Returns:
        np.array of shape (n, 2) with (width, height) for each rectangle
    """
    dimensions = np.zeros((n, 2))
    
    # Generate target areas spread across the allowed range
    min_area, max_area = 0.15, 0.65
    areas = np.linspace(min_area, max_area, n)
    
    # Mix of aspect ratios (square to rectangular)
    aspect_ratios = [1.0, 1.5, 2.0, 0.667, 0.5, 1.2, 1.8, 0.8, 1.4]
    
    for i in range(n):
        area = areas[i]
        aspect = aspect_ratios[i % len(aspect_ratios)]
        
        # width * height = area
        # width / height = aspect
        # => width = sqrt(area * aspect), height = sqrt(area / aspect)
        width = np.sqrt(area * aspect)
        height = np.sqrt(area / aspect)
        
        dimensions[i] = [width, height]
    
    return dimensions


def initialize_positions(n):
    """
    Initialize positions in a simple grid layout.
    
    Args:
        n: number of rectangles
        
    Returns:
        np.array of shape (n, 2) with initial (x, y) positions
    """
    positions = np.zeros((n, 2))
    
    # Simple grid: try to arrange in roughly square grid
    cols = int(np.ceil(np.sqrt(n)))
    spacing = 1.0 / cols
    
    for i in range(n):
        row = i // cols
        col = i % cols
        positions[i] = [col * spacing + 0.05, row * spacing + 0.05]
    
    return positions


def place_rectangles(dimensions, positions, rotations):
    """
    Place rectangles using a simple shelf-packing strategy.
    
    Args:
        dimensions: (n, 2) array of rectangle dimensions
        positions: (n, 2) array of initial positions
        rotations: (n,) boolean array
        
    Returns:
        Updated (positions, rotations)
    """
    n = len(dimensions)
    
    # Sort rectangles by area (largest first) for better packing
    areas = dimensions[:, 0] * dimensions[:, 1]
    sorted_indices = np.argsort(-areas)
    
    # Shelf packing: place rectangles left to right, then new shelf
    current_x = 0.0
    current_y = 0.0
    shelf_height = 0.0
    
    new_positions = np.zeros_like(positions)
    new_rotations = np.zeros_like(rotations)
    
    for idx in sorted_indices:
        width, height = dimensions[idx]
        
        # Check if it fits in current position
        if current_x + width <= 1.0 and current_y + height <= 1.0:
            # Fits as-is
            new_positions[idx] = [current_x, current_y]
            new_rotations[idx] = False
            current_x += width
            shelf_height = max(shelf_height, height)
        elif current_x + height <= 1.0 and current_y + width <= 1.0:
            # Try 90° rotation
            new_positions[idx] = [current_x, current_y]
            new_rotations[idx] = True
            current_x += height
            shelf_height = max(shelf_height, width)
        else:
            # Move to next shelf
            current_x = 0.0
            current_y += shelf_height
            shelf_height = 0.0
            
            # Place on new shelf
            if current_y + height <= 1.0:
                new_positions[idx] = [current_x, current_y]
                new_rotations[idx] = False
                current_x += width
                shelf_height = height
            elif current_y + width <= 1.0:
                # Try rotated on new shelf
                new_positions[idx] = [current_x, current_y]
                new_rotations[idx] = True
                current_x += height
                shelf_height = width
            else:
                # Fallback: place at origin with minimal size
                new_positions[idx] = [0.05, 0.05]
                new_rotations[idx] = False
    
    return new_positions, new_rotations


def check_valid_placement(positions, dimensions, rotations):
    """
    Check if the placement is valid (no overlaps, all in bounds).
    
    Args:
        positions: (n, 2) array of positions
        dimensions: (n, 2) array of dimensions
        rotations: (n,) boolean array
        
    Returns:
        Boolean indicating if placement is valid
    """
    n = len(positions)
    
    # Get actual dimensions after rotation
    actual_dims = np.copy(dimensions)
    for i in range(n):
        if rotations[i]:
            actual_dims[i] = [dimensions[i, 1], dimensions[i, 0]]
    
    # Check bounds
    for i in range(n):
        x, y = positions[i]
        w, h = actual_dims[i]
        if x < 0 or y < 0 or x + w > 1.0 or y + h > 1.0:
            return False
    
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            if rectangles_overlap(positions[i], actual_dims[i], 
                                 positions[j], actual_dims[j]):
                return False
    
    return True


def rectangles_overlap(pos1, dim1, pos2, dim2):
    """
    Check if two axis-aligned rectangles overlap.
    
    Args:
        pos1, pos2: (x, y) positions
        dim1, dim2: (width, height) dimensions
        
    Returns:
        Boolean indicating overlap
    """
    x1, y1 = pos1
    w1, h1 = dim1
    x2, y2 = pos2
    w2, h2 = dim2
    
    # No overlap if separated horizontally or vertically
    if x1 + w1 <= x2 or x2 + w2 <= x1:
        return False
    if y1 + h1 <= y2 or y2 + h2 <= y1:
        return False
    
    return True


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the rectangle packing constructor"""
    positions, dimensions, rotations, num_rectangles = construct_packing()
    
    # Calculate total area and score
    total_area = 0.0
    for i in range(num_rectangles):
        total_area += dimensions[i, 0] * dimensions[i, 1]
    
    packing_efficiency = total_area / 1.0
    BONUS_WEIGHT = 0.1
    combined_score = packing_efficiency + BONUS_WEIGHT * num_rectangles
    
    return positions, dimensions, rotations, num_rectangles, combined_score