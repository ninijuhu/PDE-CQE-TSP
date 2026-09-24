# PDE–CQE for the Traveling Salesman Problem

Research implementation of a hybrid quantum–classical TSP solver based on:

- **PDE** — Parameterized Distribution Evolution  
- **CQE** — Controlled Quantum Enhancement  
- **CDE** — Classical Distribution Evolution (matched control)  
- **RBD** — Resource-Bounded Decomposition and classical reconstruction  

## Requirements

- Python 3.8+
- `numpy`, `matplotlib`, `scikit-learn`, `tsplib95`
- Qiskit (with Aer simulator)

Example:

```bash
pip install numpy matplotlib scikit-learn tsplib95 qiskit qiskit-aer
```

## Repository layout

| Path | Role |
|------|------|
| `run_pde_cqe_tsp.py` | Main entry (clustering, reconstruction, CLI) |
| `pde.py` | Parameterized Distribution Evolution |
| `cqe.py` | Controlled Quantum Enhancement + diagnostics |
| `cde.py` | Classical Distribution Evolution control |
| `tsp_common.py` | Shared utilities and default hyperparameters |
| `results/` | Primary summary result tables and selected CQE diagnostic CSV (not a full dump of all raw runs) |

### Contents of `results/`

| File | Description |
|------|-------------|
| `tsplib_results.xlsx` | TSPLIB benchmark summary results |
| `synthetic_primary.xlsx` | Primary synthetic Random-*n* size series |
| `synthetic_second_geometry.xlsx` | Independent second-geometry check |
| `CDE.xlsx` | Classical Distribution Evolution control means |
| `pde_only_2opt_replacement.xlsx` | PDE-only and 2-opt replacement controls |
| `cqe_marked_probability_bays29.csv` | CQE marked-set probability events (bays29) |

## Quick start

Place TSPLIB `.tsp` files under a local `DataSet/` folder (or pass any path).

```bash
# Full model (PDE + CQE)
python run_pde_cqe_tsp.py DataSet/bays29.tsp --mode full

# Classical matched control
python run_pde_cqe_tsp.py DataSet/bays29.tsp --mode cde

# Enable CQE marked-set diagnostics (slower; for mechanism plots)
python run_pde_cqe_tsp.py DataSet/bays29.tsp --mode full --cqe-mech-diag
```

## Default settings

Aligned with the manuscript Methods defaults:

- Generations \(T=40\), population \(N_p=20\), 128 shots  
- \(\alpha_0=0.20\), soft-blend \(\beta=0.12\), bit confidence \(c=0.80\)  
- CQE interval \(\Delta G=8\), neighborhood size \(K_{nb}=8\), one AA iteration  
- Local bound \(M=8\) (no local 2-opt inside PDE–CQE)  

## Notes

- Quantum circuits are executed with the **Qiskit Aer** classical simulator.  
- Proprietary industrial platform components are **not** included and are not required to run these experiments.  
