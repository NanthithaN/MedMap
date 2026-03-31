# MedMap – Intelligent Healthcare Facility Placement System

**MedMap** is a full-stack, data-driven application designed to forecast future healthcare demand for specific geographic areas and optimally assign hospital/camp locations to meet that demand. It achieves this by bridging advanced Machine Learning (XGBoost) with Mixed-Integer Linear Programming (MILP).

---

## 🎯 Objective
Given a list of user-selected regions and a maximum hospital capacity, the system accurately predicts the future patient demand of those areas and computes the mathematically proven minimum number of hospital facilities required to support the population.

### Key Workflows:
1. **Input**: User selects multiple candidate areas (via UI) and specifies the maximum bed capacity per hospital.
2. **Prediction**: An XGBoost ML engine forecasts the future demand exclusively for the selected areas based on historical trends and demographic attributes.
3. **Optimization**: A MILP solver (PuLP) takes those predictions and dictates precisely which locations should host a newly built hospital, and which neighboring areas should be routed to them.
4. **Output**: The minimum total hospitals needed, localized assignments, and an area coverage map.

---

## 💻 Tech Stack
* **Frontend**: React (Vite), Vanilla CSS (Modern Glassmorphism aesthetics)
* **Backend API**: Python, FastAPI, Uvicorn
* **Data Handling**: Pandas, Scikit-learn
* **Machine Learning**: XGBoost (`XGBRegressor`)
* **Operations Research**: PuLP (CBC Solver)

---

## 🧠 Machine Learning (XGBoost Regressor)
The machine learning pipeline is meant to analyze long-term patterns and predict short-term localized demand escalations.

**Training Strategy:**
The model trains globally over the 50,000+ local rows of datasets. It is configured to only compute predictions during inference for the specific areas the user selects, ensuring high computational speed.

**Variables:**
* **Features (`X`)**: `population`, `nearby_hospital_count`, `avg_distance_to_hospital_km`, `day_1`, `day_2`, `day_3`, `week_1`, `week_2`, `week_3`, `month_1`.
* **Target (`y`)**: `month_2` (Represents future unobserved demand).

---

## 📐 Optimization Model (MILP Formulation)
Once the predictions are established, the system creates a Mixed-Integer Linear Programming mathematical model to map areas. We define the selected user regions as both the demands that need covering AND the candidate locations to build hospitals.

### Sets and Parameters
* **$i \in S$**: Set of selected areas requiring healthcare coverage.
* **$j \in S$**: Set of candidate areas where a hospital can physically be constructed.
* **$D_i$**: The predicted patient demand parameter for area $i$ (from XGBoost).
* **$C$**: The maximum allowed capacity parameter per hospital (User Input).

### Decision Variables
* **$y_j \in \{0, 1\}$**: Binary variable; equals $1$ if a hospital is built at candidate location $j$, $0$ otherwise.
* **$x_{i,j} \in \{0, 1\}$**: Binary variable; equals $1$ if area $i$ is assigned to be covered by the hospital at $j$, $0$ otherwise.

### Objective Function
Minimize the absolute total number of constructed hospitals:
$$ \min \sum_{j} y_j $$

### Constraints

**1. Complete Assignment Constraint**
Every selected area $i$ must be assigned to exactly one serving hospital $j$. Demand cannot be ignored or split continuously.
$$ \sum_{j} x_{i,j} = 1 \quad \forall i $$

**2. Hard Capacity Constraint**
The total demand of all areas ($D_i$) assigned to a hospital $j$ cannot exceed the maximum input capacity ($C$). Furthermore, an area cannot be assigned to $j$ if there is no hospital built there ($y_j = 0$).
$$ \sum_{i} (D_i \cdot x_{i,j}) \le C \cdot y_j \quad \forall j $$

**3. Existence Bounding Constraint**
Logical reinforcement that an assignment $x_{i,j}$ can only exist if $y_j$ is open.
$$ x_{i,j} \le y_j \quad \forall i, \forall j $$

**4. Distance Feasibility Constraint**
Abstract distance rules state that if the recorded dataset distance column lookup evaluates to exactly `0`, the routing path is deemed physically invalid.
$$ \text{If } distance(i,j) = 0 \implies x_{i,j} = 0 $$

---

## 🚀 How to Run the Project

### 1. Start the API Backend
Open a terminal inside the project root:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
*The API will start locally on port `8000`.*

### 2. Start the React Frontend
Open a secondary terminal:
```bash
cd frontend
npm install
npm run dev
```
*The UI will mount at `http://localhost:5173`.*

### 3. Usage
1. Open the localized browser link.
2. Click **Train Global Model** (This will read the `medmap_ds.csv` file into memory and initialize the XGBoost weights — you only need to do this occasionally as the backend saves it as `.pkl`).
3. Select your target test scenario mapping via the Multi-Select pane.
4. Input your Hospital Capacity parameter (make sure it exceeds individual area demands, e.g., `10000`).
5. Click **Predict & Optimize Placement** to evaluate!
