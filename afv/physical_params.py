import numpy as np
from dataclasses import dataclass


@dataclass
class PhysicalParams:
    r: float = 1.0                       # Radius (maximal) of the Voronoi cells
    A0: float = np.pi                    # Preferred area of the Voronoi cells
    P0: float = 4.8                      # Preferred perimeter of the Voronoi cells
    KA: float = 1.0                      # Area elasticity
    KP: float = 1.0                      # Perimeter elasticity
    lambda_tension: float = 0.2          # Tension difference