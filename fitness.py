import numpy as np


def repair_constraints(position, available, num_regions, num_resources):
    """
    Ensures that allocation does not exceed available resources per type.
    """
    allocation = position.reshape(num_regions, num_resources)

    for j in range(num_resources):
        total = np.sum(allocation[:, j])

        if total > available[j]:
            allocation[:, j] = allocation[:, j] * (available[j] / (total + 1e-6))

    # prevent negative values
    allocation = np.maximum(allocation, 0)

    return allocation.flatten()

def fitness_function(position, scenario):
    allocation = position.reshape(scenario.num_regions, scenario.num_resources)

    # 1. Shortage (HARD GOAL)
    shortage = np.sum(np.maximum(0, scenario.demands - allocation))

    max_shortage = np.sum(scenario.demands)
    shortage_norm = shortage / (max_shortage + 1e-6)

    # 2. Transport Cost
    distances = np.linalg.norm(
        scenario.regions_coords - scenario.warehouse_coord,
        axis=1
    )

    delivered = np.sum(allocation, axis=1)
    cost = np.sum(delivered * distances)

    max_cost = np.max(distances) * np.sum(scenario.available_resources)
    cost_norm = cost / (max_cost + 1e-6)

    # -------------------------
    # 3. Over-supply penalty (soft)
    # -------------------------
    excess = np.sum(np.maximum(0, allocation - scenario.demands))
    excess_norm = excess / (max_shortage + 1e-6)

    # -------------------------
    # FINAL BALANCED FITNESS
    # -------------------------
    alpha = 0.7   # saving lives (HIGH AS THIS THE MOST IMPORTANT)
    beta = 0.3    # logistics cost
    gamma = 0.1   # oversupply penalty 

    fitness = (
        alpha * shortage_norm +
        beta * cost_norm +
        gamma * excess_norm
    )

    return fitness

