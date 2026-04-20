---
title: 'PyAFV: A Python package for active finite Voronoi simulations of nonconfluent tissues'
tags:
  - Python
  - biophysics
  - active matter
  - Voronoi model
  - tissue mechanics
authors:
  - name: Wei Wang
    orcid: 0000-0002-0053-1069
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
  - name: Brian A. Camley
    orcid: 0000-0002-0765-6956
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: Department of Physics and Astronomy, Johns Hopkins University, Baltimore, United States
   index: 1
   ror: "00za53h95"
 - name: Department of Biophysics, Johns Hopkins University, Baltimore, United States
   index: 2
   ror: "00za53h95"
date: 22 April 2026
bibliography: paper.bib
---

# Summary

Biological tissues can exist as confluent monolayers or nonconfluent clusters with intercellular gaps. While Voronoi models have been widely used to study confluent tissues, the active finite Voronoi (AFV) model [@teomy2018confluent; @huang2023bridging] extends these to nonconfluent tissues by introducing a finite radius $\ell$ around each cell center.
`PyAFV` is an open-source Python package implementing the two-dimensional AFV model, with APIs for geometry and force computation, active dynamics, visualization, and connectivity diagnostics, as well as a calibration module that connects near-detachment AFV behavior to a deformable polygon (DP) model [@wang2026divergence]. It is aimed at researchers in biophysics and soft matter who need a computationally efficient, reproducible tool for simulating nonconfluent tissue mechanics.

# Statement of need

Studies of tissues often require a model that is neither strictly confluent nor fully dilute—capturing cell-shape-dependent mechanics, changing neighbor topology, and gap opening simultaneously. Self-propelled particle models are efficient but discard cell geometry entirely, and may also crystallize [@levine2000self; @fily2012athermal; @wang2026controlling]. Phase-field models capture cell shapes more accurately but are computationally demanding [@chiang2024intercellular; @chiang2024multiphase; @wang2025confinement]. Voronoi- and vertex-type models offer a middle ground but generally fill all space by construction, and are not suited for nonconfluent tissues [@bi2015density; @bi2016motility; @park2015unjamming; @henkes2020dense]. `PyAFV` provides a Python package for the active finite Voronoi (AFV) model [@teomy2018confluent; @huang2023bridging]—an extension of the standard Voronoi model for nonconfluent tissues—where each cell's domain is the intersection of its Voronoi region with a disk of radius $\ell$, so that cell boundaries consist of straight cell-cell contact edges and circular cell-medium arcs of radius $\ell$.

A second need is reliable calibration. Detachment forces in the AFV model diverge as cells approach full separation, so the rupture timescale depends strongly on the simulation time step without truncation [@wang2026divergence]. `PyAFV` includes a calibration module that determines a consistent truncation threshold by matching AFV near-detachment forces to those of a deformable polygon (DP) model, making calibration an explicit, reproducible step.

# State of the field

Prior AFV implementations are either unreleased or not packaged as reusable libraries. The MATLAB implementation in Ref. [@huang2023bridging] lacks integrated calibration, open-boundary handling, and a library interface. `PyAFV` provides a Python library interface for the AFV model; benchmarks show it scales as $\mathcal{O}(N)$ with system size, while the MATLAB code scales as $\mathcal{O}(N^{3/2})$, giving `PyAFV` substantial speedups for $N \gtrsim 10^3$.

# Software design

**Open boundary handling.** For $N \geq 3$, `PyAFV` uses SciPy's Voronoi routine only for the initial combinatorial structure [@scipy2020]. Unbounded ridges—arising when a cell has no neighbor on one side—are resolved by introducing explicit extension vertices and updating region membership, without imposing artificial confinement or periodic boundary conditions. Small-$N$ edge cases are handled separately.

**Energy and force computation.** For given cell center coordinates $\{\mathbf{r}_i\}$, `PyAFV` implements

$$
E = \sum_i K_A(A_i - A_0)^2 + K_P(P_i - P_0)^2 + \Lambda P_i^{(n)},
$$

where $K_A$, $K_P$ are elastic moduli, $A_0$, $P_0$ are preferred area and perimeter, $P_i^{(n)}$ is the non-contacting arc length, and $\Lambda$ is the tension difference between non-contacting and contacting edges [@teomy2018confluent; @huang2023bridging; @wang2026divergence]. Each cell is decomposed into polygons and circular arc segments; forces on each cell center are computed as $-\nabla_i E$.

**Hybrid backend.** Geometry routines for the finite Voronoi construction are implemented in both Cython and pure Python. The Cython backend is selected automatically when available; otherwise the pure-Python fallback is used with no change to the public API. Users can force backend selection for debugging.

**API and transparency.** The central class is `FiniteVoronoiSimulator`, initialized with cell-center coordinates and a `PhysicalParams` dataclass. The `build()` method of this class returns a diagnostics dictionary of forces, geometric quantities, and contact connectivity. Cell dynamics are implemented in user scripts rather than inside the library, keeping modeling assumptions explicit. Per-cell heterogeneous preferred areas are supported for mixed-population studies.

**Calibration module.** Because detachment forces diverge as cells approach full separation [@wang2026divergence], calibration is essential for interpreting tissue fractures. `pyafv.calibrate.auto_calibrate()` finds the steady-state of a cell doublet in the deformable polygon (DP) model [@boromand2018jamming; @lv2024active], probes its detachment force under external pulling, and determines the AFV truncation threshold $\delta$ that matches the target force.

# Example

The snippet below illustrates a minimal AFV workflow for geometry and force computation:

```python
import numpy as np
import pyafv as afv

N = 100                                          # number of cells
pts = np.random.rand(N, 2) * 10                  # initial positions
phys = afv.PhysicalParams(r=1.0)                 # use default parameter values
sim = afv.FiniteVoronoiSimulator(pts, phys)      # initialize the simulator

diag = sim.build()
forces = diag["forces"]
connections = diag["connections"]
```

To visualize the constructed finite Voronoi diagram, users can run `sim.plot_2d(show=True)`. Additional examples and notebooks in the project demonstrate relaxation trajectories, active dynamics, connectivity extraction, and custom visualization. The full API and documentation are available for further details [@wang2026pyafv].

# Research impact statement

`PyAFV` was used to produce all the finite Voronoi simulation results of our previous work [@wang2026divergence], and its calculated forces were validated against the MATLAB implementation [@huang2023bridging] for identical configurations. Its faster scaling compared to the MATLAB code, integrated calibration, and Python library interface make it well-suited for systematic studies of nonconfluent tissue mechanics.

# AI usage disclosure

Generative AI tools were used for limited assistance during development. GitHub Copilot assisted with code review in pull requests; OpenAI Codex and Anthropic Claude Code were used to suggest code edits and documentation phrasing. All suggestions were reviewed and validated by the authors, who are responsible for the correctness and integrity of the software and paper.

# Acknowledgements

The authors acknowledge support from NIH Grant No. R35GM142847.
This work was also carried out at the Advanced Research Computing at Hopkins (ARCH) core facility, which is supported by the National Science Foundation (NSF) Grant No. OAC1920103.

# References
