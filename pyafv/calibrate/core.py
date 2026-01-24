import numpy as np
from typing import Iterable, TypeVar

from ..physical_params import PhysicalParams, target_delta
from .deformable_polygon import DeformablePolygonSimulator


T = TypeVar("T") # Declare the type variable

def _in_notebook() -> bool:             # pragma: no cover
    """Check if the code is running in a Jupyter notebook environment."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        return ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False

def _tqdm(it: Iterable[T], desc: str = "") -> tuple[Iterable[T], bool]:            # pragma: no cover
    """Wrap an iterable with tqdm if enabled and available; otherwise return it unchanged."""
    try:
        if _in_notebook():
            from tqdm.notebook import tqdm  # type: ignore
        else:
            from tqdm import tqdm  # type: ignore
        return tqdm(it, desc=desc), True
    except ImportError:
        return it, False


def auto_calibrate(phys: PhysicalParams, ext_forces: np.ndarray | None = None,
                   dt: float = 1e-3, nsteps: int = 50_000, show: bool = False) -> tuple[float, PhysicalParams]:
    """
    Auto-calibrate the parameters *phys* against the deformable-polygon (DP) model.

    In this calibration, we simulate an initially steady-state cell doublet under increasing external force dipoles using the DP model; the external force starts from 0 to max(*ext_forces*).
    We identify the detachment force as the first external force at which detachment occurs.
    We then search for the :py:attr:`delta` value in the finite-Voronoi (FV) model to match this detachment force.

    Args:
        phys: The initial physical parameters.
        ext_forces: An array of external forces to apply during calibration;
            defaults to None, which uses ``np.linspace(0, 10, 101)[1:]``;
            should start from a small positive value.
        dt: Time step for each simulation step.
        nsteps: Number of simulation steps to run for each external force.
        show: Whether to print progress information; no need to use it if **tqdm** is installed.
    
    Raises:
        TypeError: If *phys* is not an instance of *PhysicalParams*.
    
    Returns:
        A *tuple* containing the detachment force and the calibrated *PhysicalParams*.
        If detachment does not occur within the given force range, return a *NaN* force.

    .. warning::
        This function may take some time to run, depending on the parameters.
        (If **tqdm** is installed, a progress bar will be shown automatically.)
        Do not change defaults unless you understand the implications.

        If you only need a rough or faster calibration, you may change the external force range or interval.
        Adjusting *dt* and *nsteps* may also speed up simulations, but may affect accuracy; test the
        :py:class:`DeformablePolygonSimulator` model separately to ensure accuracy is acceptable.
    """

    sim = DeformablePolygonSimulator(phys)

    if sim.detached: # already detached at zero force (steady state)
        detachment_force = 0.0
        return float(detachment_force), sim.phys.replace(delta=0.0)
    else:
        if ext_forces is None:           # pragma: no cover
            ext_forces = np.linspace(0, 10, 101)[1:]
        pbar, has_tqdm = _tqdm(ext_forces, desc="Calibrating")

        for ext_force in pbar:
            if has_tqdm:    # with tqdm                     # pragma: no cover
                pbar.set_description(f"Applying F={ext_force:.1f}")
            elif show:  # no tqdm, but show is True         # pragma: no cover
                print(f"Applying F={ext_force:.1f}")

            sim.simulate(ext_force, dt, nsteps)
            if sim.detached:
                detachment_force = ext_force
                return float(detachment_force), sim.phys.replace(delta=target_delta(sim.phys, detachment_force))
        
        # did not detach within given forces
        return float('nan'), sim.phys.replace(delta=0.45*sim.phys.r)    # pragma: no cover
