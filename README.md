# PDE–CQE for the Traveling Salesman Problem

Research implementation of a hybrid quantum–classical TSP solver based on:

- **PDE** — Parameterized Distribution Evolution  
- **CQE** — Controlled Quantum Enhancement  
- **RBD** — Resource-Bounded Decomposition and classical reconstruction  

## Requirements

- Python 3.8+
- `numpy`, `matplotlib`, `pandas`, `scikit-learn`, `tsplib95`
- Qiskit (with Aer simulator)

Example:

```bash
pip install numpy matplotlib pandas scikit-learn tsplib95 qiskit qiskit-aer
```

## Repository layout

| Path | Role |
|------|------|
| `run_pde_cqe_tsp.py` | Main entry |
| `pde.py` | Parameterized Distribution Evolution |
| `cqe.py` | Controlled Quantum Enhancement + diagnostics |
| `tsp_common.py` | Shared utilities and default hyperparameters |
| `draw1.py` | Plot TSPLIB overall performance |
| `ablation_plot.py` | Plot PDE/CQE mechanism isolation |
| `results/` | Primary summary result tables and selected CQE diagnostic CSV |
| `aco.py` | ACO |
| `ga.py` | GA |
| `greedy.py` | Nearest-neighbor greedy |
| `qaco.py` | Variant QACO |
| `statistical_tests.py` | Mann–Whitney, Friedman, and Wilcoxon tests |

### Contents of `results/`

| File | Description |
|------|-------------|
| `tsplib_results.xlsx` | TSPLIB benchmark summary results |
| `synthetic_primary.xlsx` | Primary synthetic Random-*n* size series |
| `synthetic_second_geometry.xlsx` | Independent second-geometry check |
| `cqe_marked_probability_bays29.csv` | CQE marked-set probability events (bays29) |

Additional raw run-level outputs and intermediate files are available from the corresponding author upon reasonable request.

## Quick start

Place TSPLIB `.tsp` files under a local `DataSet/` folder (or pass any path).

```bash
# Full model (PDE + CQE)
python run_pde_cqe_tsp.py DataSet/bays29.tsp --mode full

# PDE only (no CQE)
python run_pde_cqe_tsp.py DataSet/bays29.tsp --mode qde

# Enable CQE marked-set diagnostics (slower; for mechanism plots)
python run_pde_cqe_tsp.py DataSet/bays29.tsp --mode full --cqe-mech-diag

# Performance / ablation figures
python draw1.py
python ablation_plot.py
```

## Default settings

Aligned with the manuscript Methods defaults:

- Generations T = 40, population N_p = 20, 128 shots
- α_0 = 0.20, soft-blend β = 0.12, bit confidence c = 0.80
- CQE interval ΔG = 8, neighborhood size K_nb = 8, one amplitude-amplification iteration per CQE event
- Local bound M = 8 (no local 2-opt inside PDE–CQE)
- ACO: 30 ants, 100 iterations, α = 1, β = 2, ρ = 0.1, Q = 100
- GA: population 100, 500 generations, crossover 0.8, mutation 0.2
- Greedy: nearest neighbor
- QACO: encoding m⌈log₂ m⌉, Ry pheromone, 128 shots, max_loops = 40 per subcluster
- `statistical_tests.py`: one-sided Mann–Whitney with Holm within each instance; Friedman plus one-sided Wilcoxon with Holm on the per-instance means

## Notes

- Quantum circuits are executed with the **Qiskit Aer** classical simulator.  
- Proprietary industrial platform components are **not** included and are not required to run these experiments.  
