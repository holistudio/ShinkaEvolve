"""Constructor-based bathroom fixture layout optimization"""

import numpy as np


def construct_packing():
    """
    Construct an arrangement of 3 bathroom fixtures (toilet, urinal, wash basin)
    in a 3.0 x 2.5 unit rectangle to optimize layout based on entrance position.

    Returns:
        Tuple of (positions, dimensions, rotations, fixture_names)
        positions: np.array of shape (3, 2) with (x, y) bottom-left corners
        dimensions: np.array of shape (3, 2) with (width, height) for each fixture
        rotations: np.array of shape (3,) with boolean rotation flags (True = 90° rotated)
        fixture_names: list of fixture names ["toilet", "urinal", "wash_basin"]
    """
    # Define outer bounds
    outer_width = 3.0
    outer_height = 2.5
    
    # Define fixtures with their constraints
    fixtures = define_fixtures()
    
    # Select entrance point (this is the key generator variable)
    entrance = select_entrance_point()
    
    # Generate layout based on entrance
    positions, dimensions, rotations, fixture_names = generate_layout_from_entrance(
        entrance, fixtures, outer_width, outer_height
    )
    
    return positions, dimensions, rotations, fixture_names


def define_fixtures():
    """
    Define the three bathroom fixtures with their constraints.
    
    Returns:
        Dictionary with fixture specifications
    """
    fixtures = {
        'toilet': {
            'area': 1.5,
            'aspect_ratio': 0.514,  # width / height
            'placement_type': 'corner',  # Must touch 2 perpendicular edges
        },
        'urinal': {
            'area': 0.63,
            'aspect_ratio': 0.7,
            'placement_type': 'edge',  # Longer side along edge
        },
        'wash_basin': {
            'area': 0.63,
            'aspect_ratio': 0.7,
            'placement_type': 'edge',  # Longer side along edge
        }
    }
    
    # Calculate dimensions from area and aspect ratio
    # area = w * h, aspect = w/h
    # => w = sqrt(area * aspect), h = sqrt(area / aspect)
    for name, fixture in fixtures.items():
        w = np.sqrt(fixture['area'] * fixture['aspect_ratio'])
        h = np.sqrt(fixture['area'] / fixture['aspect_ratio'])
        fixture['width'] = w
        fixture['height'] = h
        fixture['name'] = name
    
    return fixtures


def select_entrance_point():
    """
    Select entrance point - this is the key variable that generates layouts.
    
    Entrance can be on any of 4 walls at various positions.
    Evolution will optimize this choice.
    
    Returns:
        dict with 'position' (x, y), 'wall' (bottom/top/left/right), 
        and 'offset' (position along wall 0.0-1.0)
    """
    # Define entrance options
    # For initial implementation, use bottom wall centered
    
    # Wall options: 'bottom', 'top', 'left', 'right'
    wall = 'bottom'
    offset = 0.5  # Center of wall (0.0 = start, 1.0 = end)
    
    # Calculate actual position based on wall and offset
    if wall == 'bottom':
        position = (3.0 * offset, 0)
    elif wall == 'top':
        position = (3.0 * offset, 2.5)
    elif wall == 'left':
        position = (0, 2.5 * offset)
    else:  # right
        position = (3.0, 2.5 * offset)
    
    return {
        'position': position,
        'wall': wall,
        'offset': offset
    }


def generate_layout_from_entrance(entrance, fixtures, outer_width, outer_height):
    """
    Generate fixture positions based on entrance location.
    
    Strategy:
    1. Toilet goes in corner (preferably away from entrance)
    2. Wash basin near entrance (for accessibility)
    3. Urinal fills remaining wall space
    
    Args:
        entrance: dict with entrance info
        fixtures: dict with fixture specs
        outer_width, outer_height: container dimensions
        
    Returns:
        positions, dimensions, rotations, fixture_names
    """
    entry_wall = entrance['wall']
    
    # Initialize arrays for 3 fixtures
    positions = np.zeros((3, 2))
    dimensions = np.zeros((3, 2))
    rotations = np.zeros(3, dtype=bool)
    fixture_names = ['toilet', 'urinal', 'wash_basin']
    
    # Get fixture data
    toilet = fixtures['toilet']
    urinal = fixtures['urinal']
    basin = fixtures['wash_basin']
    
    # Place toilet in corner based on entrance wall
    toilet_corner = select_toilet_corner(entry_wall)
    toilet_pos, toilet_rot = place_toilet_in_corner(
        toilet_corner, toilet, outer_width, outer_height
    )
    
    # Place wash basin near entrance
    basin_pos, basin_rot = place_wash_basin_near_entrance(
        entrance, basin, outer_width, outer_height, toilet_pos, toilet, toilet_rot
    )
    
    # Place urinal on remaining edge
    urinal_pos, urinal_rot = place_urinal_on_edge(
        entry_wall, urinal, outer_width, outer_height, 
        [toilet_pos, basin_pos], 
        [toilet, basin],
        [toilet_rot, basin_rot]
    )
    
    # Assemble results in order: toilet, urinal, wash_basin
    positions[0] = toilet_pos
    positions[1] = urinal_pos
    positions[2] = basin_pos
    
    # Dimensions (accounting for rotation)
    dimensions[0] = [toilet['width'], toilet['height']]
    dimensions[1] = [urinal['width'], urinal['height']]
    dimensions[2] = [basin['width'], basin['height']]
    
    rotations[0] = toilet_rot
    rotations[1] = urinal_rot
    rotations[2] = basin_rot
    
    return positions, dimensions, rotations, fixture_names


def select_toilet_corner(entry_wall):
    """
    Select corner for toilet placement based on entry wall.
    Prefer corner away from entrance.
    """
    # Strategy: place toilet in corner opposite or diagonal to entrance
    corner_preference = {
        'bottom': 'top-left',     # If entrance on bottom, toilet at top
        'top': 'bottom-right',    # If entrance on top, toilet at bottom
        'left': 'top-right',      # If entrance on left, toilet at right
        'right': 'bottom-left',   # If entrance on right, toilet at left
    }
    
    return corner_preference.get(entry_wall, 'bottom-left')


def place_toilet_in_corner(corner, toilet, outer_width, outer_height):
    """
    Place toilet in specified corner with 2 edges aligned.
    
    Args:
        corner: 'bottom-left', 'bottom-right', 'top-left', 'top-right'
        toilet: fixture dict
        outer_width, outer_height: bounds
        
    Returns:
        position (x, y), rotation (bool)
    """
    w, h = toilet['width'], toilet['height']
    rotation = False
    
    corner_positions = {
        'bottom-left': (0, 0),
        'bottom-right': (outer_width - w, 0),
        'top-left': (0, outer_height - h),
        'top-right': (outer_width - w, outer_height - h)
    }
    
    position = corner_positions.get(corner, (0, 0))
    
    return np.array(position), rotation


def place_wash_basin_near_entrance(entrance, basin, outer_width, outer_height, 
                                   toilet_pos, toilet, toilet_rot):
    """
    Place wash basin near entrance, typically on adjacent wall.
    Longer side must be along edge.
    
    Args:
        entrance: entrance dict
        basin: fixture dict
        outer_width, outer_height: bounds
        toilet_pos: toilet position to avoid
        toilet: toilet fixture
        toilet_rot: toilet rotation
        
    Returns:
        position (x, y), rotation (bool)
    """
    entry_wall = entrance['wall']
    w, h = basin['width'], basin['height']
    
    # Determine which wall to place basin on
    # Strategy: adjacent to entrance wall, avoiding toilet
    if entry_wall == 'bottom':
        # Try left wall
        wall = 'left'
        # Rotate to make longer side vertical (along wall)
        if w > h:
            rotation = True
            actual_w, actual_h = h, w
        else:
            rotation = False
            actual_w, actual_h = w, h
        
        # Position on left wall, offset from bottom
        position = np.array([0, 0.3])
        
    elif entry_wall == 'top':
        # Try right wall
        wall = 'right'
        if w > h:
            rotation = True
            actual_w, actual_h = h, w
        else:
            rotation = False
            actual_w, actual_h = w, h
        
        position = np.array([outer_width - actual_w, outer_height - actual_h - 0.3])
        
    elif entry_wall == 'left':
        # Try bottom wall
        wall = 'bottom'
        if h > w:
            rotation = True
            actual_w, actual_h = h, w
        else:
            rotation = False
            actual_w, actual_h = w, h
        
        position = np.array([0.3, 0])
        
    else:  # right
        # Try top wall
        wall = 'top'
        if h > w:
            rotation = True
            actual_w, actual_h = h, w
        else:
            rotation = False
            actual_w, actual_h = w, h
        
        position = np.array([outer_width - actual_w - 0.3, outer_height - actual_h])
    
    return position, rotation


def place_urinal_on_edge(entry_wall, urinal, outer_width, outer_height,
                        occupied_positions, occupied_fixtures, occupied_rotations):
    """
    Place urinal with longer side along edge, avoiding occupied spaces.
    
    Args:
        entry_wall: entrance wall
        urinal: fixture dict
        outer_width, outer_height: bounds
        occupied_positions: list of occupied positions
        occupied_fixtures: list of fixture dicts
        occupied_rotations: list of rotations
        
    Returns:
        position (x, y), rotation (bool)
    """
    w, h = urinal['width'], urinal['height']
    
    # Try to place on wall opposite to entrance or remaining space
    if entry_wall in ['bottom', 'top']:
        # Place on left or right wall
        wall = 'right'  # Default to right
        
        # Rotate to make longer side vertical
        if w > h:
            rotation = True
            actual_w, actual_h = h, w
        else:
            rotation = False
            actual_w, actual_h = w, h
        
        # Find available space on right wall
        position = np.array([outer_width - actual_w, 1.0])
        
    else:  # entry on left or right
        # Place on bottom or top wall
        wall = 'bottom'  # Default to bottom
        
        # Rotate to make longer side horizontal
        if h > w:
            rotation = True
            actual_w, actual_h = h, w
        else:
            rotation = False
            actual_w, actual_h = w, h
        
        # Find available space on bottom wall
        position = np.array([1.5, 0])
    
    return position, rotation


def check_overlap(pos1, dim1, rot1, pos2, dim2, rot2, margin=0.0):
    """
    Check if two rectangles overlap with optional margin.
    
    Args:
        pos1, pos2: (x, y) positions
        dim1, dim2: (width, height) dimensions
        rot1, rot2: rotation flags
        margin: minimum separation distance
        
    Returns:
        bool: True if overlap (including margin)
    """
    # Apply rotation to dimensions
    w1, h1 = (dim1[1], dim1[0]) if rot1 else (dim1[0], dim1[1])
    w2, h2 = (dim2[1], dim2[0]) if rot2 else (dim2[0], dim2[1])
    
    x1, y1 = pos1
    x2, y2 = pos2
    
    # Check separation with margin
    if x1 + w1 + margin <= x2 or x2 + w2 + margin <= x1:
        return False  # Separated horizontally
    if y1 + h1 + margin <= y2 or y2 + h2 + margin <= y1:
        return False  # Separated vertically
    
    return True  # Overlap


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the bathroom fixture layout constructor"""
    positions, dimensions, rotations, fixture_names = construct_packing()
    
    # Calculate metrics
    total_area = 0.0
    for i in range(len(fixture_names)):
        total_area += dimensions[i, 0] * dimensions[i, 1]
    
    # For bathroom, scoring could be based on:
    # - Valid placement (all constraints met)
    # - Accessibility (clearance from entrance)
    # - Efficiency (space utilization)
    
    outer_area = 3.0 * 2.5  # 7.5 sq units
    space_efficiency = total_area / outer_area
    
    # Simple scoring: high efficiency is good
    # In practice, you might add accessibility, flow, and ergonomic scores
    combined_score = space_efficiency
    
    return positions, dimensions, rotations, fixture_names, combined_score

    import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize():
    """Simple visualization of the bathroom layout"""
    # Get the layout
    positions, dimensions, rotations, fixture_names, score = run_packing()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw outer boundary (3.0 x 2.5)
    boundary = patches.Rectangle((0, 0), 3.0, 2.5, 
                                linewidth=3, edgecolor='black', 
                                facecolor='white')
    ax.add_patch(boundary)
    
    # Colors for each fixture
    colors = {'toilet': 'lightblue', 'urinal': 'lightyellow', 'wash_basin': 'lightgreen'}
    
    # Draw each fixture
    for i in range(3):
        x, y = positions[i]
        w, h = dimensions[i]
        name = fixture_names[i]
        
        # Apply rotation
        if rotations[i]:
            w, h = h, w
        
        # Draw rectangle
        rect = patches.Rectangle((x, y), w, h,
                                linewidth=2, 
                                edgecolor='black',
                                facecolor=colors[name])
        ax.add_patch(rect)
        # s
        # Add label
        cx, cy = x + w/2, y + h/2
        area = dimensions[i, 0] * dimensions[i, 1]
        label = f"{name}\n{area:.2f}m²"
        ax.text(cx, cy, label, ha='center', va='center', 
               fontsize=10, fontweight='bold')
    
    # Set limits and aspect
    ax.set_xlim(-0.3, 3.3)
    ax.set_ylim(-0.3, 2.8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Title
    ax.set_title(f'Bathroom Layout (Score: {score:.3f})', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Width (m)')
    ax.set_ylabel('Height (m)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize()