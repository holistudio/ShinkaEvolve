"""
Evaluator for the bathroom fixture layout problem.
"""

import os
import argparse
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from importlib.machinery import SourceFileLoader

from shinka.core import run_shinka_eval



def format_layout_string(
    positions: np.ndarray,
    dimensions: np.ndarray,
    rotations: np.ndarray,
    fixture_names: List[str],
) -> str:
    """Formats the layout data into a multi-line string for display."""
    return "\n".join(
        [
            f"  {fixture_names[i]}: pos=({positions[i][0]:.2f}, {positions[i][1]:.2f}), "
            f"dim=({dimensions[i][0]:.2f}, {dimensions[i][1]:.2f}), "
            f"rot={rotations[i]}"
            for i in range(len(fixture_names))
        ]
    )

def validate_layout(
    run_output: Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], float],
    check_overlap_fn: callable,
) -> Tuple[bool, Optional[str]]:
    """
    Validates the bathroom layout to ensure no overlaps and all fixtures are
    within the bounds.

    Args:
        run_output: The output from the `construct_packing` function.
        check_overlap_fn: The function to check for overlaps between fixtures.

    Returns:
        (is_valid: bool, error_message: Optional[str])
    """
    positions, dimensions, rotations, fixture_names, _ = run_output
    msg = "The layout is valid. Fixtures are within bounds and do not overlap."
    
    outer_width, outer_height = 3.0, 2.5
    num_fixtures = len(fixture_names)

    # 1. Check if all fixtures are within the outer boundary
    for i in range(num_fixtures):
        x, y = positions[i]
        w, h = dimensions[i]
        if rotations[i]:
            w, h = h, w  # Swap dimensions if rotated

        if x < 0 or y < 0 or x + w > outer_width or y + h > outer_height:
            msg = f"Fixture '{fixture_names[i]}' is outside the boundary."
            return False, msg

    # 2. Check for overlaps between any two fixtures
    for i in range(num_fixtures):
        for j in range(i + 1, num_fixtures):
            overlap = check_overlap_fn(
                positions[i],
                dimensions[i],
                rotations[i],
                positions[j],
                dimensions[j],
                rotations[j],
                margin=0.0,  # Using a margin of 0.0 for direct contact check
            )
            if overlap:
                msg = f"Fixtures '{fixture_names[i]}' and '{fixture_names[j]}' overlap."
                return False, msg

    return True, msg


def get_layout_kwargs(run_index: int) -> Dict[str, Any]:
    """Provides keyword arguments for layout runs (none needed)."""
    return {}


def aggregate_layout_metrics(
    results: List[Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], float]],
    results_dir: str,
) -> Dict[str, Any]:
    """
    Aggregates metrics for the bathroom layout. Assumes num_runs=1.
    Saves extra.npz with detailed layout information.
    """
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    # The score to maximize is the total area of the fixtures.
    # This is equivalent to minimizing the remaining area.
    positions, dimensions, rotations, fixture_names, _ = results[0]
    total_area = sum(dim[0] * dim[1] for dim in dimensions)

    public_metrics = {
        "layout_str": format_layout_string(
            positions, dimensions, rotations, fixture_names
        ),
        "num_fixtures": len(fixture_names),
    }
    private_metrics = {
        "total_fixture_area": float(total_area),
    }
    metrics = {
        "combined_score": float(total_area),
        "public": public_metrics,
        "private": private_metrics,
    }

    extra_file = os.path.join(results_dir, "extra.npz")
    try:
        np.savez(
            extra_file,
            positions=positions,
            dimensions=dimensions,
            rotations=rotations,
            fixture_names=np.array(fixture_names, dtype=object),
            total_area=total_area,
        )
        print(f"Detailed layout data saved to {extra_file}")
    except Exception as e:
        print(f"Error saving extra.npz: {e}")
        metrics["extra_npz_save_error"] = str(e)

    return metrics


def main(program_path: str, results_dir: str):
    """Runs the bathroom layout evaluation using shinka.eval."""
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    # Dynamically import the check_overlap function from the submission
    try:
        user_module = SourceFileLoader("user_module", program_path).load_module()
        check_overlap_fn = getattr(user_module, "check_overlap")
    except (FileNotFoundError, AttributeError) as e:
        print(f"Error: Could not load 'check_overlap' from {program_path}. {e}")
        # Create dummy files for a failed run
        with open(os.path.join(results_dir, "correct.json"), "w") as f:
            f.write('{"correct": false}')
        with open(os.path.join(results_dir, "metrics.json"), "w") as f:
            f.write('{"combined_score": 0.0, "error": "Failed to load check_overlap"}')
        return

    num_experiment_runs = 1

    # Define nested functions to pass context to the shinka runner
    def _validate_with_context(
        r: Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], float]
    ) -> Tuple[bool, Optional[str]]:
        return validate_layout(r, check_overlap_fn)

    def _aggregator_with_context(
        r: List[Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], float]]
    ) -> Dict[str, Any]:
        return aggregate_layout_metrics(r, results_dir)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="construct_packing",
        num_runs=num_experiment_runs,
        get_experiment_kwargs=get_layout_kwargs,
        validate_fn=_validate_with_context,
        aggregate_metrics_fn=_aggregator_with_context,
    )

    if correct:
        print("Evaluation and Validation completed successfully.")
    else:
        print(f"Evaluation or Validation failed: {error_msg}")

    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: <string_too_long_to_display>")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bathroom layout evaluator using shinka.eval"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to program to evaluate (must contain 'construct_packing' and 'check_overlap')",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Dir to save results (metrics.json, correct.json, extra.npz)",
    )
    parsed_args = parser.parse_args()
    main(parsed_args.program_path, parsed_args.results_dir)