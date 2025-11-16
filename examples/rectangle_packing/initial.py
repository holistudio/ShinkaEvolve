import numpy as np



def construct_packing(
    entrance_wall,
    entrance_offset,
    toilet_wall_x,
    toilet_wall_y,
    toilet_offset_x,
    toilet_offset_y,
    urinal_wall,
    urinal_offset,
    basin_wall,
    basin_offset
):
    """
    Construct an arrangement of 3 bathroom fixtures (toilet, urinal, wash basin)
    in a 3.0 x 2.5 unit rectangle to optimize layout based on parameters.
    
    HYPERPARAMETERS:
    - entrance_wall: 'bottom', 'top', 'left', 'right'
    - entrance_offset: 0.0 to 1.0 (position along entrance wall)
    
    - toilet_wall_x: 'left', 'right' (which vertical edge toilet touches)
    - toilet_wall_y: 'bottom', 'top' (which horizontal edge toilet touches)
    - toilet_offset_x: 0.0 to 1.0 (offset along vertical wall)
    - toilet_offset_y: 0.0 to 1.0 (offset along horizontal wall)
    
    - urinal_wall: 'bottom', 'top', 'left', 'right' (which wall for urinal)
    - urinal_offset: 0.0 to 1.0 (position along urinal wall)
    
    - basin_wall: 'bottom', 'top', 'left', 'right' (which wall for basin)
    - basin_offset: 0.0 to 1.0 (position along basin wall)

    Returns:
        Tuple of (positions, dimensions, rotations, fixture_names)
    """
    # Define outer bounds
    outer_width = 3.0
    outer_height = 2.5
    
    # Define fixtures with their constraints
    fixtures = define_fixtures()
    
    # Generate layout from parameters
    positions, dimensions, rotations, fixture_names = generate_layout_from_params(
        entrance_wall, entrance_offset,
        toilet_wall_x, toilet_wall_y, toilet_offset_x, toilet_offset_y,
        urinal_wall, urinal_offset,
        basin_wall, basin_offset,
        fixtures, outer_width, outer_height
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


def generate_layout_from_params(entrance_wall, entrance_offset,
                               toilet_wall_x, toilet_wall_y, toilet_offset_x, toilet_offset_y,
                               urinal_wall, urinal_offset,
                               basin_wall, basin_offset,
                               fixtures, outer_width, outer_height):
    """
    Generate fixture positions based on hyperparameters.
    
    Args:
        entrance_wall, entrance_offset: entrance position
        toilet_wall_x, toilet_wall_y: which perpendicular walls toilet touches
        toilet_offset_x, toilet_offset_y: position along those walls
        urinal_wall, urinal_offset: urinal wall and position
        basin_wall, basin_offset: basin wall and position
        fixtures: fixture specifications
        outer_width, outer_height: container dimensions
        
    Returns:
        positions, dimensions, rotations, fixture_names
    """
    # Initialize arrays for 3 fixtures
    positions = np.zeros((3, 2))
    dimensions = np.zeros((3, 2))
    rotations = np.zeros(3, dtype=bool)
    fixture_names = ['toilet', 'urinal', 'wash_basin']
    
    # Get fixture data
    toilet = fixtures['toilet']
    urinal = fixtures['urinal']
    basin = fixtures['wash_basin']
    
    # Place toilet using two perpendicular walls
    toilet_pos, toilet_rot = place_toilet_on_walls(
        toilet_wall_x, toilet_wall_y, toilet_offset_x, toilet_offset_y,
        toilet, outer_width, outer_height
    )
    
    # Place urinal on specified wall
    urinal_pos, urinal_rot = place_fixture_on_wall(
        urinal_wall, urinal_offset, urinal, outer_width, outer_height
    )
    
    # Place wash basin on specified wall
    basin_pos, basin_rot = place_fixture_on_wall(
        basin_wall, basin_offset, basin, outer_width, outer_height
    )
    
    # Assemble results in order: toilet, urinal, wash_basin
    positions[0] = toilet_pos
    positions[1] = urinal_pos
    positions[2] = basin_pos
    
    # Dimensions (base dimensions, rotation applied during visualization)
    dimensions[0] = [toilet['width'], toilet['height']]
    dimensions[1] = [urinal['width'], urinal['height']]
    dimensions[2] = [basin['width'], basin['height']]
    
    rotations[0] = toilet_rot
    rotations[1] = urinal_rot
    rotations[2] = basin_rot
    
    return positions, dimensions, rotations, fixture_names


def place_toilet_on_walls(wall_x, wall_y, offset_x, offset_y, toilet, outer_width, outer_height):
    """
    Place toilet touching two perpendicular walls (corner constraint).
    
    Args:
        wall_x: 'left' or 'right' (which vertical edge)
        wall_y: 'bottom' or 'top' (which horizontal edge)
        offset_x: 0.0-1.0 position along vertical wall
        offset_y: 0.0-1.0 position along horizontal wall
        toilet: fixture dict
        outer_width, outer_height: bounds
        
    Returns:
        position (x, y), rotation (bool)
    """
    w, h = toilet['width'], toilet['height']
    rotation = False
    
    # X position based on vertical wall
    if wall_x == 'left':
        x = 0
    else:  # right
        x = outer_width - w
    
    # Y position based on horizontal wall
    if wall_y == 'bottom':
        y = 0
    else:  # top
        y = outer_height - h
    
    # Apply offsets (allows moving away from exact corner)
    # offset_x/y are ratios that can move toilet along the wall it touches
    if wall_x == 'left':
        # Can move right, but still touch left wall (x stays 0)
        x = 0
    else:  # right
        # Can move left, but still touch right wall
        x = outer_width - w
    
    if wall_y == 'bottom':
        # Can move up, but still touch bottom wall (y stays 0)
        y = 0
    else:  # top
        # Can move down, but still touch top wall
        y = outer_height - h
    
    # Note: offsets could be used for fine-tuning if corners aren't exactly at edges
    # For strict corner placement, we keep exact edge alignment
    
    position = np.array([x, y])
    return position, rotation


def place_fixture_on_wall(wall, position_ratio, fixture, outer_width, outer_height):
    """
    Place fixture on specified wall at specified position.
    Automatically orients longer side along the wall.
    
    Args:
        wall: 'bottom', 'top', 'left', 'right'
        position_ratio: position along wall (0.0 = start, 1.0 = end)
        fixture: fixture dict
        outer_width, outer_height: bounds
        
    Returns:
        position (x, y), rotation (bool)
    """
    w, h = fixture['width'], fixture['height']
    longer_side = max(w, h)
    shorter_side = min(w, h)
    
    # Determine rotation based on wall and fixture dimensions
    if wall in ['bottom', 'top']:
        # Horizontal walls - need horizontal orientation (longer side horizontal)
        if h > w:
            # Currently vertical, need to rotate
            rotation = True
            actual_w, actual_h = h, w
        else:
            rotation = False
            actual_w, actual_h = w, h
        
        # Calculate position
        available_length = outer_width - actual_w
        x = position_ratio * available_length
        
        if wall == 'bottom':
            y = 0
        else:  # top
            y = outer_height - actual_h
            
    else:  # left or right walls
        # Vertical walls - need vertical orientation (longer side vertical)
        if w > h:
            # Currently horizontal, need to rotate
            rotation = True
            actual_w, actual_h = h, w
        else:
            rotation = False
            actual_w, actual_h = w, h
        
        # Calculate position
        available_length = outer_height - actual_h
        y = position_ratio * available_length
        
        if wall == 'left':
            x = 0
        else:  # right
            x = outer_width - actual_w
    
    position = np.array([x, y])
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


    """
Add this function to initial.py to visualize with entrance point
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_with_entrance():
    """Visualize bathroom layout with entrance point marked"""
    # Get the layout
    positions, dimensions, rotations, fixture_names, score = run_packing()
    
    # Get entrance point (need to expose this from construct_packing)
    entrance = select_entrance_point()
    
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
        
        # Add label
        cx, cy = x + w/2, y + h/2
        area = dimensions[i, 0] * dimensions[i, 1]
        label = f"{name}\n{area:.2f}m²"
        ax.text(cx, cy, label, ha='center', va='center', 
               fontsize=10, fontweight='bold')
    
    # Draw entrance point
    ex, ey = entrance['position']
    wall = entrance['wall']
    
    # Draw entrance marker (arrow pointing inward)
    arrow_size = 0.2
    
    if wall == 'bottom':
        # Arrow pointing up from bottom
        arrow = patches.FancyArrow(
            ex, ey - arrow_size*1.5, 0, arrow_size,
            width=0.15, head_width=0.25, head_length=0.1,
            fc='red', ec='darkred', linewidth=2, zorder=10
        )
        text_pos = (ex, ey - arrow_size*2)
        
    elif wall == 'top':
        # Arrow pointing down from top
        arrow = patches.FancyArrow(
            ex, ey + arrow_size*1.5, 0, -arrow_size,
            width=0.15, head_width=0.25, head_length=0.1,
            fc='red', ec='darkred', linewidth=2, zorder=10
        )
        text_pos = (ex, ey + arrow_size*2)
        
    elif wall == 'left':
        # Arrow pointing right from left
        arrow = patches.FancyArrow(
            ex - arrow_size*1.5, ey, arrow_size, 0,
            width=0.15, head_width=0.25, head_length=0.1,
            fc='red', ec='darkred', linewidth=2, zorder=10
        )
        text_pos = (ex - arrow_size*2.5, ey)
        
    else:  # right
        # Arrow pointing left from right
        arrow = patches.FancyArrow(
            ex + arrow_size*1.5, ey, -arrow_size, 0,
            width=0.15, head_width=0.25, head_length=0.1,
            fc='red', ec='darkred', linewidth=2, zorder=10
        )
        text_pos = (ex + arrow_size*2.5, ey)
    
    ax.add_patch(arrow)
    
    # Add entrance label
    ax.text(text_pos[0], text_pos[1], 'ENTRANCE', 
           ha='center', va='center',
           fontsize=10, fontweight='bold', color='darkred',
           bbox=dict(boxstyle='round', facecolor='white', 
                    edgecolor='red', linewidth=2))
    
    # Add entrance dot
    ax.plot(ex, ey, 'ro', markersize=10, zorder=11)
    
    # Set limits and aspect
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.0)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Title with entrance info
    offset_pct = entrance['offset'] * 100
    title = (f"Bathroom Layout (Score: {score:.3f})\n"
            f"Entrance: {wall.upper()} wall @ {offset_pct:.0f}%")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Width (m)')
    ax.set_ylabel('Height (m)')
    
    plt.tight_layout()
    plt.show()


# # Run this to visualize with entrance
# if __name__ == "__main__":
#     visualize_with_entrance()

# EVOLVE-BLOCK-START

construct_packing(entrance_wall='bottom',
    entrance_offset=0.5,
    toilet_wall_x='left',
    toilet_wall_y='top',
    toilet_offset_x=0.0,
    toilet_offset_y=0.0,
    urinal_wall='left',
    urinal_offset=0.1,
    basin_wall='left',
    basin_offset=0.15)

# EVOLVE-BLOCK-END