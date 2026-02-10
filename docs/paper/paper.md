---
title: 'PyAFV: A Python package for active finite-Voronoi simulations of nonconfluent tissues'
tags:
  - Python
  - biophysics
  - active matter
  - Voronoi model
  - tissue mechanics
authors:
  - name: Wei Wang
    orcid: 0000-0002-0053-1069
    # corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
  - name: Brian A. Camley
    orcid: 0000-0002-0765-6956
    # corresponding: true # (This is how to denote the corresponding author)
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: Department of Physics and Astronomy, Johns Hopkins University, Baltimore, United States
   index: 1
   ror: "00za53h95"
 - name: Department of Biophysics, Johns Hopkins University, Baltimore, United States
   index: 2
   ror: "00za53h95"
date: 12 February 2026
bibliography: paper.bib
---

# Summary

Collective cell behavior is commonly modeled using frameworks that balance geometric realism against computational efficiency.
Self-propelled particle models are computationally efficient but lack explicit cell boundaries and interfacial mechanics [@fily2012athermal; @levine2000self; @wang2025controlling], while phase-field models resolve interfaces in detail at substantially higher computational cost for large systems [@chiang2024intercellular; @chiang2024multiphase]. Voronoi- and vertex-type models occupy an intermediate regime, retaining cell-shape-dependent mechanics with comparatively low overhead [@bi2016motility; @park2015unjamming; @henkes2020dense].

The active finite Voronoi (AFV) model extends conventional confluent Voronoi models to nonconfluent settings by introducing a finite radius scale $\ell$ around each cell center [@teomy2018confluent; @huang2023bridging].
This construction allows cell-cell contacts to terminate naturally when Voronoi edges extend beyond $\ell$, enabling gap opening and cell detachment within the same geometric framework, which is essential for studies of tissue cohesion, detachment, and fracture-like events.
In the AFV model, each cell boundary is composed of straight Voronoi contact segments and circular free-boundary arcs of radius $\ell$, and forces are derived from area-perimeter mechanics with distinct contacting and non-contacting interfacial tensions.

`pyafv` is an open-source Python package implementing this framework in two dimensions.
It provides high-level APIs for geometry/force computation, active dynamics workflows, visualization, and diagnostics such as cell-cell connectivity and contacting/non-contacting boundary length.
The package also includes calibration tools that connect AFV behavior to a deformable-polygon (DP) doublet model [@wang2026divergence; @lv2024active; @boromand2018jamming], enabling practical parameter tuning for near-detachment mechanics.

# Statement of need

Computational studies of epithelial systems frequently require modeling regimes that are neither strictly confluent nor fully dilute.
In these regimes, researchers need methods that simultaneously capture (i) cell-shape-dependent mechanics, (ii) changing neighbor topology, and (iii) opening/closing of intercellular gaps.
Standard confluent Voronoi implementations are often optimized for tiling tissues without free boundaries [@bi2015density], whereas more detailed interface-resolving methods can substantially reduce throughput in large parameter scans [@wang2025confinement].

`pyafv` addresses this need by packaging AFV simulations in a reproducible Python workflow aimed at theoretical and computational biophysics studies of tissue cohesion, detachment, and fragmentation.
The software is designed for groups that need a practical compromise between geometric detail and computational efficiency, especially when studying transitions between cohesive and fragmented tissue states.
It provides explicit access to intermediate geometry and force diagnostics, making it suitable not only for production simulations but also for method validation and mechanistic interpretation.

A second need is transparent calibration.
Near-detachment behavior in finite-Voronoi geometries can be sensitive to regularization and parameter choices, leading to ambiguous or model-dependent detachment forces if calibration is not performed explicitly [@wang2026divergence].
`pyafv` includes a dedicated calibration submodule that formalizes this process and enables reproducible comparison between AFV simulations and related polygonal models.

# Methods and implementation

## AFV geometry and dynamics

Let $\{\mathbf r_i\}$ be cell-center coordinates.
In the standard Voronoi construction, the Voronoi region of cell $i$ is the set of points closer to $\mathbf r_i$ than to any other center.
AFV defines the finite cell domain as the intersection of the standard Voronoi region with a disk of radius $\ell$ centered at $\mathbf r_i$, explicitly truncating long Voronoi edges and introducing free boundaries.
Consequently, boundaries contain straight cell-cell interfaces (where Voronoi edges are retained) and circular cell-medium interfaces (where truncation by radius $\ell$ applies).

`pyafv` computes mechanics from area and perimeter terms with a separate contribution for non-contacting perimeter length, i.e., 

$$
E=\sum_i K_A(A_i-A_0)^2 + K_P(P_i-P_0)^2 + \Lambda P_i^{(n)},
$$

where $K_A$ and $K_P$ are elastic moduli for cell area $A_i$ and perimeter $P_i$, respectively, and $A_0$ and $P_0$ are the preferred area and perimeter, respectively. $P_i^{(n)}$ is the non-contacting perimeter length and $\Lambda$ measures the tension difference between contacting and non-contacting edges [@teomy2018confluent; @huang2023bridging; @wang2026divergence].
Cell centers evolve in overdamped active dynamics,

$$
\dot{\mathbf r}_i = -\mu\nabla_i E + v_0\mathbf n_i,
$$

with optional rotational diffusion for polarity direction.
This overdamped active dynamics is implemented explicitly in the package examples, making modeling assumptions and update rules transparent in user scripts.

## Software interface

The central public object is `FiniteVoronoiSimulator`, initialized with point coordinates and a `PhysicalParams` dataclass.
`sim.build()` returns a diagnostics dictionary containing conservative forces, geometric data, and inferred connectivity.
The returned diagnostics are designed to facilitate inspection and debugging of geometric and mechanical contributions, which is particularly important near contact-breaking transitions.
This API supports common workflows: relaxation to mechanical equilibrium, active dynamics, custom visualization, and extraction of contact networks.

The package also supports heterogeneous preferred areas through per-cell updates, enabling mixed-population or perturbation studies without custom forks of the core geometry code.

## Numerical design

For $N \geqslant 3$, `pyafv` uses SciPy Voronoi tessellations only as an initial geometric scaffold [@scipy2020]. Unbounded ridges arising under open boundaries are detected and resolved by explicitly introducing extension vertices and updating region membership, enabling finite-Voronoi cell geometries to be constructed without imposing artificial confinement or periodic boundaries.
Special handling is included for small-$N$ edge cases.
Geometry routines then decompose each cell into polygonal and arc contributions to compute area/perimeter quantities and the corresponding force derivatives.

To balance speed and portability, `pyafv` uses a hybrid backend strategy: a Cython implementation is selected when available, otherwise a pure-Python fallback is used automatically.
Users can also force backend selection for debugging.

## Calibration workflow

Calibration is essential for interpreting AFV detachment forces in terms of underlying mechanical parameters.
The `pyafv.calibrate` module provides tools to align AFV near-detachment behavior with a DP cell-doublet model [@wang2026divergence].
The core automated workflow, implemented in `pyafv.calibrate.auto_calibrate()`, proceeds as follows:

1. determines steady-state doublet geometry,
2. probes DP detachment under increasing external pulling,
3. infers an AFV contact truncation threshold that reproduces the target detachment force.

This makes calibration a documented and integral part of the simulation setup rather than an *ad hoc* post-processing step.

# Example

The snippet below illustrates a minimal AFV workflow for geometry and force computation:

```python
import numpy as np
import pyafv as afv

N = 100
pts = np.random.rand(N, 2) * 10
phys = afv.PhysicalParams()
sim = afv.FiniteVoronoiSimulator(pts, phys)

diag = sim.build()
forces = diag["forces"]
connections = diag["connections"]
```

To visualize the constructed finite-Voronoi diagram, users can run `sim.plot_2d(show=True)`.
With the same interface, users can run relaxation trajectories, add active self-propulsion, extract connectivity networks, and update preferred areas per cell.
Additional examples and notebooks in the project demonstrate these workflows, including custom plotting and periodic-boundary visualization. The full API and documentation are available for further details [@wang2026pyafv].

# AI usage disclosure

During development of this software, generative AI tools were used for limited assistance. GitHub Copilot was used within the GitHub interface to assist with code review and minor refactoring suggestions during pull-request review. In addition, OpenAI Codex-based tools were occasionally used to suggest code edits and documentation phrasing.
All AI-assisted suggestions were reviewed, edited, and validated by the human authors. The core model design, algorithms, implementation decisions, validation, and interpretation were made by the authors, who take full responsibility for the correctness, originality, licensing, and compliance of the submitted software and paper.

# Acknowledgements

The authors acknowledge support from NIH Grant No. R35GM142847.
This work was carried out at the Advanced Research Computing at Hopkins (ARCH) core facility, which is supported by the National Science Foundation (NSF) Grant No. OAC1920103.

# References
