import numpy as np


def check_constraints(solution, scenario, verbose=True):
    """
    Validates whether a solution satisfies all problem constraints.

    Constraints checked:
    1. Resource capacity constraints
    2. Non-negativity constraint
    3. Demand-related constraint (soft check)

    Parameters:
    - solution: 1D numpy array (flattened solution)
    - scenario: DisasterScenario object
    - verbose: whether to print logs

    Returns:
    - True if all HARD constraints are satisfied
    - False otherwise
    """

    reshaped = solution.reshape(
        (scenario.num_regions, scenario.num_resources)
    )

    if verbose:
        print("\n CONSTRAINT CHECK ")

    # 1. Resource Constraints (HARD)
    for j in range(scenario.num_resources):
        total_used = np.sum(reshaped[:, j])
        capacity = scenario.available_resources[j]

        if verbose:
            print(f"Resource {j}: {total_used:.2f} / {capacity}")

        if total_used > capacity + 1e-6:
            if verbose:
                print(f" VIOLATION: Resource {j} exceeded capacity")
            return False

    # 2. Non-negativity (HARD)
    if np.any(reshaped < 0):
        if verbose:
            print(" VIOLATION: Negative values detected")
        return False

    # 3. Demand Check (SOFT)
    excess = reshaped - scenario.demands

    if np.any(excess > 0):
        if verbose:
            print(" Warning: Some allocations exceed demand (allowed)")

    if verbose:
        print(" All HARD constraints satisfied")

    return True