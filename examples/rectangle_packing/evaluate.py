"""
Evaluator for rectangle packing example (3-9 rectangles)
"""

import os
import argparse
import numpy as np
from typing import Tuple, Optional, List, Dict, Any

from shinka.core import run_shinka_eval


BONUS_WEIGHT = 0.1  # Weight for number of rectangles in scoring


def format_rectangles_string(positions: np.ndarray, dimensions: np.ndarray, 
                             rotations: np.ndarray) -> str:
    """Formats rectangle configuration into a multi-line string for display."""
    lines = []
    for i in range(len(positions)):
        x, y = positions[i]
        w, h = dimensions[i]
        rot = rotations[i]
        area = w * h
        rot_str = "90°" if rot else "0°"
        
        # Actual dimensions after rotation
        actual_w = h if rot else w
        actual_h = w if rot else h
        
        lines.append(
            f"  Rect {i+1}: pos=({x:.4f}, {y:.4f}), "
            f"dim=({actual_w:.4f}×{actual_h:.4f}), "
            f"area={area:.4f}, rot={rot_str}"
        )
    return "\n".join(lines)


def adapted_validate_packing(
    run_output: Tuple[np.ndarray, np.ndarray, np.ndarray, int, float],
    atol=1e-6,
) -> Tuple[bool, Optional[str]]:
    """
    Validates rectangle packing results based on the output of 'run_packing'.

    Args:
        run_output: Tuple (positions, dimensions, rotations, num_rectangles, combined_score)

    Returns:
        (is_valid: bool, error_message: Optional[str])
    """
    positions, dimensions, rotations, num_rectangles, combined_score = run_output
    
    # Convert to numpy arrays if needed
    if not isinstance(positions, np.ndarray):
        positions = np.array(positions)
    if not isinstance(dimensions, np.ndarray):
        dimensions = np.array(dimensions)
    if not isinstance(rotations, np.ndarray):
        rotations = np.array(rotations)
    
    # Validate number of rectangles
    if not (3 <= num_rectangles <= 9):
        msg = f"Number of rectangles ({num_rectangles}) must be between 3 and 9"
        return False, msg
    
    # Validate shapes
    if positions.shape != (num_rectangles, 2):
        msg = f"Positions shape incorrect. Expected ({num_rectangles}, 2), got {positions.shape}"
        return False, msg
    
    if dimensions.shape != (num_rectangles, 2):
        msg = f"Dimensions shape incorrect. Expected ({num_rectangles}, 2), got {dimensions.shape}"
        return False, msg
    
    if rotations.shape != (num_rectangles,):
        msg = f"Rotations shape incorrect. Expected ({num_rectangles},), got {rotations.shape}"
        return False, msg
    
    # Validate all dimensions are positive
    if np.any(dimensions <= 0):
        msg = "All dimensions must be positive"
        return False, msg
    
    # Validate area constraints (0.15 to 0.65 per rectangle)
    areas = dimensions[:, 0] * dimensions[:, 1]
    min_area, max_area = 0.15, 0.65
    
    for i, area in enumerate(areas):
        if area < min_area - atol or area > max_area + atol:
            msg = (
                f"Rectangle {i} has area {area:.4f}, "
                f"which is outside allowed range [{min_area}, {max_area}]"
            )
            return False, msg
    
    # Get actual dimensions after rotation
    actual_dims = np.copy(dimensions)
    for i in range(num_rectangles):
        if rotations[i]:
            actual_dims[i] = [dimensions[i, 1], dimensions[i, 0]]
    
    # Validate rectangles are within unit square bounds
    for i in range(num_rectangles):
        x, y = positions[i]
        w, h = actual_dims[i]
        
        if x < -atol or y < -atol or x + w > 1 + atol or y + h > 1 + atol:
            msg = (
                f"Rectangle {i} at ({x:.4f}, {y:.4f}) with size ({w:.4f}×{h:.4f}) "
                f"is outside unit square [0,1]×[0,1]"
            )
            return False, msg
    
    # Validate no overlaps
    for i in range(num_rectangles):
        for j in range(i + 1, num_rectangles):
            if rectangles_overlap(
                positions[i], actual_dims[i],
                positions[j], actual_dims[j],
                atol
            ):
                msg = (
                    f"Rectangles {i} and {j} overlap. "
                    f"Rect {i}: ({positions[i][0]:.4f}, {positions[i][1]:.4f}) "
                    f"size ({actual_dims[i][0]:.4f}×{actual_dims[i][1]:.4f}), "
                    f"Rect {j}: ({positions[j][0]:.4f}, {positions[j][1]:.4f}) "
                    f"size ({actual_dims[j][0]:.4f}×{actual_dims[j][1]:.4f})"
                )
                return False, msg
    
    # Validate combined score calculation
    total_area = np.sum(areas)
    packing_efficiency = total_area / 1.0
    expected_score = packing_efficiency + BONUS_WEIGHT * num_rectangles
    
    if not np.isclose(combined_score, expected_score, atol=atol):
        msg = (
            f"Combined score ({combined_score:.6f}) does not match "
            f"expected ({expected_score:.6f})"
        )
        return False, msg
    
    msg = (
        f"Valid packing: {num_rectangles} rectangles, "
        f"total_area={total_area:.4f}, "
        f"efficiency={packing_efficiency:.4f}, "
        f"score={combined_score:.4f}"
    )
    return True, msg


def rectangles_overlap(pos1, dim1, pos2, dim2, atol=1e-6):
    """
    Check if two axis-aligned rectangles overlap.
    
    Args:
        pos1, pos2: (x, y) bottom-left positions
        dim1, dim2: (width, height) dimensions
        atol: tolerance for floating point comparison
        
    Returns:
        Boolean indicating overlap
    """
    x1, y1 = pos1
    w1, h1 = dim1
    x2, y2 = pos2
    w2, h2 = dim2
    
    # Rectangles don't overlap if separated horizontally or vertically
    # Using tolerance for numerical stability
    if x1 + w1 <= x2 + atol or x2 + w2 <= x1 + atol:
        return False
    if y1 + h1 <= y2 + atol or y2 + h2 <= y1 + atol:
        return False
    
    return True


def get_rectangle_packing_kwargs(run_index: int) -> Dict[str, Any]:
    """Provides keyword arguments for rectangle packing runs (none needed)."""
    return {}


def aggregate_rectangle_packing_metrics(
    results: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]], 
    results_dir: str
) -> Dict[str, Any]:
    """
    Aggregates metrics for rectangle packing. Assumes num_runs=1.
    Saves extra.npz with detailed packing information.
    """
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    positions, dimensions, rotations, num_rectangles, combined_score = results[0]
    
    # Calculate metrics
    areas = dimensions[:, 0] * dimensions[:, 1]
    total_area = np.sum(areas)
    packing_efficiency = total_area / 1.0
    
    public_metrics = {
        "rectangles_str": format_rectangles_string(positions, dimensions, rotations),
        "num_rectangles": int(num_rectangles),
        "packing_efficiency": float(packing_efficiency),
        "total_area": float(total_area),
    }
    
    private_metrics = {
        "individual_areas": [float(a) for a in areas],
        "bonus_weight": BONUS_WEIGHT,
        "score_breakdown": {
            "efficiency_component": float(packing_efficiency),
            "rectangle_bonus": float(BONUS_WEIGHT * num_rectangles),
        }
    }
    
    metrics = {
        "combined_score": float(combined_score),
        "public": public_metrics,
        "private": private_metrics,
    }

    # Save detailed packing data
    extra_file = os.path.join(results_dir, "extra.npz")
    try:
        np.savez(
            extra_file,
            positions=positions,
            dimensions=dimensions,
            rotations=rotations,
            num_rectangles=num_rectangles,
            combined_score=combined_score,
            total_area=total_area,
            packing_efficiency=packing_efficiency,
        )
        print(f"Detailed packing data saved to {extra_file}")
    except Exception as e:
        print(f"Error saving extra.npz: {e}")
        metrics["extra_npz_save_error"] = str(e)

    return metrics


def main(program_path: str, results_dir: str):
    """Runs the rectangle packing evaluation using shinka.eval."""
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    num_experiment_runs = 1

    # Define a nested function to pass results_dir to the aggregator
    def _aggregator_with_context(
        r: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]],
    ) -> Dict[str, Any]:
        return aggregate_rectangle_packing_metrics(r, results_dir)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_packing",
        num_runs=num_experiment_runs,
        get_experiment_kwargs=get_rectangle_packing_kwargs,
        validate_fn=adapted_validate_packing,
        aggregate_metrics_fn=_aggregator_with_context,
    )

    if correct:
        print("✓ Evaluation and Validation completed successfully.")
    else:
        print(f"✗ Evaluation or Validation failed: {error_msg}")

    print("\nMetrics:")
    for key, value in metrics.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: <string_too_long_to_display>")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rectangle packing evaluator using shinka.eval"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to program to evaluate (must contain 'run_packing')",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Dir to save results (metrics.json, correct.json, extra.npz)",
    )
    parsed_args = parser.parse_args()
    main(parsed_args.program_path, parsed_args.results_dir)