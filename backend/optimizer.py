import pulp

def optimize_hospitals(selected_areas_info: list[dict], capacity: float):
    """
    Solves the MILP to minimize the number of hospitals needed to cover all selected areas.
    
    selected_areas_info: list of dicts with:
        'name': str
        'demand': int
        'distances': list of floats (the distance_hospital_1 to distance_hospital_10)
    capacity: float (max demand a single hospital can serve)
    """
    n = len(selected_areas_info)
    
    # Define problem
    prob = pulp.LpProblem("MedMap_Facility_Placement", pulp.LpMinimize)
    
    # Decision Variables
    # x[i][j] = 1 if area i is assigned to hospital at area j
    # y[j] = 1 if hospital built at area j
    x = pulp.LpVariable.dicts("assign", ((i, j) for i in range(n) for j in range(n)), cat="Binary")
    y = pulp.LpVariable.dicts("hospital", (j for j in range(n)), cat="Binary")
    
    # Objective Function
    # Minimize total number of hospitals
    prob += pulp.lpSum(y[j] for j in range(n)), "Minimize_Hospitals"
    
    # --- Constraints ---
    
    # 1. Each area must be assigned to exactly one hospital
    for i in range(n):
        prob += pulp.lpSum(x[i, j] for j in range(n)) == 1, f"One_Assignment_{i}"
        
    # 2. Capacity constraints for each hospital location j
    for j in range(n):
        # sum of demands assigned to this hospital must be <= capacity * y[j]
        # if y[j] is 0, then sum must be 0 (no one can be assigned)
        prob += pulp.lpSum(selected_areas_info[i]['demand'] * x[i, j] for i in range(n)) <= capacity * y[j], f"Capacity_{j}"
        
    # 3. Assignment only if hospital exists
    for i in range(n):
        for j in range(n):
            prob += x[i, j] <= y[j], f"Exists_{i}_{j}"
            
    # 4. Distance Validity (CRITICAL logic from spec)
    # Using dataset distances: area i to hospital j uses the j-th distance column of area i.
    # If distance = 0 -> x[i][j] = 0
    for i in range(n):
        for j in range(n):
            # Access the distance from area i to "hospital j"
            # If j is larger than the available distances (e.g. j >= 10), default to 0.
            if j < len(selected_areas_info[i]['distances']):
                dist_val = float(selected_areas_info[i]['distances'][j])
            else:
                dist_val = 0.0
                
            if dist_val == 0:
                prob += x[i, j] == 0, f"Distance_Zero_{i}_{j}"
                
    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    
    if pulp.LpStatus[prob.status] != "Optimal":
        return {
            "status": "infeasible",
            "message": "Could not find a valid assignment. Capacity might be too low or distances might be 0 preventing assignment."
        }
        
    # Extract results
    hospitals_built = []
    assignments = []
    
    for j in range(n):
        if pulp.value(y[j]) is not None and pulp.value(y[j]) > 0.5:
            hospitals_built.append(selected_areas_info[j]['name'])
            
        for i in range(n):
            if pulp.value(x[i, j]) is not None and pulp.value(x[i, j]) > 0.5:
                assignments.append({
                    "area": selected_areas_info[i]['name'],
                    "assigned_hospital": selected_areas_info[j]['name']
                })
                
    return {
        "status": "success",
        "total_hospitals": len(hospitals_built),
        "hospitals": hospitals_built,
        "assignments": assignments
    }
