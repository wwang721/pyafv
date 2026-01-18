import pyafv as afv
import pyafv.calibrate as cal
import numpy as np


def test_optimize_calibration():
    params = afv.PhysicalParams(lambda_tension=0.0)
    df, params_cal = cal.auto_calibrate(params)
    assert np.abs(df - 0.0) < 1e-8, "Calibration error is too high."

    params = afv.PhysicalParams(lambda_tension=0.01)
    df, params_cal = cal.auto_calibrate(params, ext_forces=[0.5, 0.6, 0.7])
    assert np.abs(df - 0.6) < 1e-8, "Calibration error is too high."

def test_simulator(phys):
    sim = cal.DeformablePolygonSimulator(phys)
    sim.plot_2d()

def test_centroid(phys):
    sim = cal.DeformablePolygonSimulator(phys)
    pts1 = sim.pts1
    pts2 = sim.pts2
    centroid1 = cal.polygon_centroid(pts1)
    centroid2 = cal.polygon_centroid(pts2)
    distance = np.linalg.norm(centroid1 - centroid2)

    # compute centroid distance using analytical formula
    l, d = 0.8703157358228049, 0.7503482672171494     # pre-computed reference value
    epsilon = l - d

    phi = np.arctan2(np.sqrt(l**2 - (l - epsilon)**2), l - epsilon)
    Acap = l**2 * (phi - np.sin(phi) * np.cos(phi))
    xcap = 4.*l * np.sin(phi)**3/(3.*(2*phi - np.sin(2*phi)))
    Delta = Acap * xcap / (np.pi * l**2 - Acap)
    centroid_distance_ref = 2 * d + 2 * Delta
    
    assert np.abs(distance - centroid_distance_ref) < 1e-3, "Centroid distance calculation is incorrect."