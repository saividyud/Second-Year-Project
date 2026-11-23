#%% Importing the necessary modules
from PLsim.otf import OTF
from PLsim.scene import Scene, SceneProjector
from PLsim.system.device import Device
from PLsim.system.lantern import generate_unitary_matrix

import hcipy as hc
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

import pandas as pd
from tqdm import tqdm

# Custom modules
import functions
from PhotonicLanternDevice import PhotonicLanternDevice
from PhotonicLanternFitting import PhotonicLanternFitting

import multiprocessing as mp

# Formatting the plots
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['figure.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.title_fontsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['mathtext.fontset'] = 'cm'

#%% Defining the constants
# Constants
MAS_TO_RAD = 1e-3/206265 # [rad/mas]

# Photonic lantern attributes
N_DIM = 65 # Number of pixels in pupil grid
DIAMETER = 10 # [m], diameter of telescope aperture for pupil
FOCAL_RESOLUTION = 1e-6 # [m], resolution of focal grid (make sure this samples LP modes enough but also not truncate the outer parts)
FOCAL_LENGTH = 40 # [m], focal length of telescope (initial value, OTF.from_fiber_params will find an optimal focal ratio)

# Fiber attributes
N_CORE = 1.444 # Index of refraction of the core
N_CLAD = 1.444 - 5.5e-3 # Index of refraction of the cladding
R_CORE = 10e-6 # [m], core radius

# Wavelengths
N_SPEC = 150 # Number of wavelength channels
WAV_MIDDLE = 1282.8 # [nm], Paschen beta line
WAV_WIDTH = 0.1 # [nm]
WAV_BANDWIDTH = 10*WAV_WIDTH # [nm]
WAV_RANGE = np.linspace(WAV_MIDDLE - WAV_BANDWIDTH/2, WAV_MIDDLE + WAV_BANDWIDTH/2, N_SPEC)

#%% Defining the binary (same as fitting.ipynb)
# Masses of the stars
M_1 = 1 # [M_1]
M_2 = 0.9 # [M_1]

# Continuum fluxes of the stars
F_1 = 1
F_2 = F_1 / functions.M_L_relation(M_1, M_2)

# Line properties (depth, center [nm], width [nm])
A_LINE_1, WAV_MIDDLE_1, WAV_WIDTH_1 = 0.9, WAV_MIDDLE, WAV_WIDTH
A_LINE_2, WAV_MIDDLE_2, WAV_WIDTH_2 = 0.94, WAV_MIDDLE - 0.05, WAV_WIDTH + 0.01

STAR1_SPECTRUM = functions.create_spectrum(F_1, A_LINE_1, WAV_MIDDLE_1, WAV_WIDTH_1, WAV_RANGE)
STAR2_SPECTRUM = functions.create_spectrum(F_2, A_LINE_2, WAV_MIDDLE_2, WAV_WIDTH_2, WAV_RANGE)

# Star 1 is fixed at the origin
STAR1_POSITION = np.array([0, 0]) # [mas]

# Spectrum shape of star 3 per unit continuum flux (flat continuum; its flux in units of F_1 is what the survey searches over)
STAR3_SHAPE = np.ones(N_SPEC)

# Line guesses for stage 3 of the full pipeline
LINE_GUESS = np.array([
    [0.8, WAV_MIDDLE + 0.1, WAV_WIDTH - 0.04],
    [0.7, WAV_MIDDLE - 0.05, WAV_WIDTH + 0.03]
])

# Parameter names of the binary fit
PARAM_NAMES = ['x1', 'y1', 'x2', 'y2', 'F1', 'a1', 'c1', 'w1', 'M2', 'a2', 'c2', 'w2']

#%% Defining parameters of the survey
# Positions for star 2
r2s = np.array([15, 30, 45, 60]) # [mas]
theta2s = np.array([0, 45, 90, 135, 180]) # [deg]

# Number of grid points along each axis for the star 3 positions
N_GRID_3 = 20

# Bracket of star 3 continuum fluxes searched by the bisection [F_1]
F3_BRACKET = (1e-6, 1)

# Number of bisection steps (each halves the bracket in log space; 7 steps narrows 6 decades to ~11%)
N_BISECTION = 7

# Measurement noise (fractional, per output and wavelength), as passed to calculate_photonic_lantern_outputs
NOISE = 1e-3

# A parameter has failed when its bias exceeds K_SIGMA of its noise-limited uncertainty
K_SIGMA = 3

# Star 3 is detectable when the chi^2 of the best binary fit rises by more than K_DETECT standard deviations of the chi^2 distribution
K_DETECT = 3

#%% Survey functions
def true_binary_params(star2_position : np.ndarray) -> np.ndarray:
    '''
    True binary parameters (x1, y1, x2, y2, F1, a1, c1, w1, M2, a2, c2, w2) for a given star 2 position.
    '''
    return np.array([*STAR1_POSITION, *star2_position, F_1, A_LINE_1, WAV_MIDDLE_1, WAV_WIDTH_1, M_2, A_LINE_2, WAV_MIDDLE_2, WAV_WIDTH_2])

def simulate_triple(lantern : PhotonicLanternDevice, star2_position : np.ndarray, star3_position : np.ndarray, F3 : float, noise : float | None = None, seed : int = 0) -> tuple[np.ndarray, np.ndarray]:
    '''
    Simulates an observation of the binary plus a third star.

    Returns
    -------
    fractional_intensity : np.ndarray
        (N_SPEC x N_PORTS) observed fractional intensities
    total_intensity : np.ndarray
        (N_SPEC,) observed total intensity
    '''
    positions = np.array([STAR1_POSITION, star2_position, star3_position])
    spectra = np.array([STAR1_SPECTRUM, STAR2_SPECTRUM, F3 * STAR3_SHAPE])

    outputs = lantern.calculate_photonic_lantern_outputs(positions, spectra, noise=noise, seed=seed)
    total_intensity = np.sum(outputs, axis=1)

    return outputs / total_intensity[:, None], total_intensity

def numerical_jacobian(fitter : PhotonicLanternFitting, params : np.ndarray, observed_fractional_intensities : np.ndarray, observed_total : np.ndarray) -> np.ndarray:
    '''
    Central difference Jacobian of `fitter._binary_objective` with respect to the 12 binary parameters (N_DATA x 12).
    '''
    steps = 1e-4 * np.array([*[fitter.position_scale] * 4, 0.1 * params[4], *fitter.line_scale, fitter.mass_scale, *fitter.line_scale])

    J = np.zeros((N_SPEC * fitter.lantern.N_PORTS, len(params)))

    for i, h in enumerate(steps):
        plus, minus = params.copy(), params.copy()
        plus[i] += h
        minus[i] -= h

        J[:, i] = (fitter._binary_objective(plus, observed_fractional_intensities, observed_total, M_1) - fitter._binary_objective(minus, observed_fractional_intensities, observed_total, M_1)) / (2 * h)

    return J

def linear_estimate(fitter : PhotonicLanternFitting, star2_position : np.ndarray, star3_positions : np.ndarray) -> dict:
    '''
    Linearized (no fitting) estimate of how a faint third star biases the stage 4 binary fit.

    Parameters
    ----------
    fitter : PhotonicLanternFitting
        Fitter whose stage 4 objective is linearized
    star2_position : np.ndarray
        Position of star 2 [mas] (2,)
    star3_positions : np.ndarray
        Positions of star 3 [mas] (N_3 x 2)

    Returns
    -------
    result : dict
        sigma : (12,) noise-limited uncertainty of each parameter
        bias_per_flux : (N_3 x 12) bias of each parameter per unit star 3 flux
        F3_crit_param : (N_3 x 12) star 3 flux at which each parameter is biased by K_SIGMA sigma
        F3_crit : (N_3,) star 3 flux at which the first parameter fails
        limiting_param : (N_3,) index of the first parameter to fail
        F3_detect : (N_3,) star 3 flux at which the chi^2 of the binary fit rises detectably
    '''
    lantern = fitter.lantern
    truth = true_binary_params(star2_position)

    # Noiseless binary observation (the expansion point)
    binary_outputs = lantern.calculate_photonic_lantern_outputs(np.array([STAR1_POSITION, star2_position]), np.array([STAR1_SPECTRUM, STAR2_SPECTRUM]))
    binary_total = np.sum(binary_outputs, axis=1) # (N_SPEC,)
    binary_fractional = binary_outputs / binary_total[:, None] # (N_SPEC, N_PORTS)

    # Jacobian of the stage 4 residuals at the true binary
    J = numerical_jacobian(fitter, truth, binary_fractional, binary_total) # (N_DATA, 12)
    J_pinv = np.linalg.pinv(J) # (12, N_DATA), equal to (J^T J)^-1 J^T

    # Noise of each stage 4 residual: the noise is multiplicative per output, so it is NOISE times the fractional intensity
    noise_std = NOISE * binary_fractional.ravel() # (N_DATA,)

    # Covariance of an unweighted least squares fit with this noise: C = J^+ Sigma J^+^T
    covariance = (J_pinv * noise_std**2) @ J_pinv.T
    sigma = np.sqrt(np.diag(covariance))

    # Star 3 outputs per unit continuum flux, all positions in one pass (outputs are linear in the spectrum at each wavelength)
    star3_outputs = lantern.calculate_flat_outputs_per_body(star3_positions) * STAR3_SHAPE[None, :, None] # (N_3, N_SPEC, N_PORTS)

    # Residual change per unit star 3 flux: r = (model - observed) / observed_total, so adding F3 * u3 to the observation gives r3 = -u3 / T_binary
    r3 = -(star3_outputs / binary_total[None, :, None]).reshape(len(star3_positions), -1) # (N_3, N_DATA)

    # Linearized fit: minimizing |F3 r3 + J delta|^2 gives delta = -F3 J^+ r3
    bias_per_flux = -r3 @ J_pinv.T # (N_3, 12)

    # Flux at which each parameter's bias reaches K_SIGMA sigma
    with np.errstate(divide='ignore'):
        F3_crit_param = K_SIGMA * sigma[None, :] / np.abs(bias_per_flux)

    # Residual left after the binary absorbs what it can: F3 (I - J J^+) r3, and its chi^2 per unit flux squared
    unabsorbed = r3 + bias_per_flux @ J.T # (N_3, N_DATA)
    chi2_per_flux2 = np.sum((unabsorbed / noise_std[None, :])**2, axis=1) # (N_3,)

    n_dof = J.shape[0] - J.shape[1]
    F3_detect = np.sqrt(K_DETECT * np.sqrt(2 * n_dof) / chi2_per_flux2)

    return {
        'sigma': sigma,
        'bias_per_flux': bias_per_flux,
        'F3_crit_param': F3_crit_param,
        'F3_crit': np.min(F3_crit_param, axis=1),
        'limiting_param': np.argmin(F3_crit_param, axis=1),
        'F3_detect': F3_detect,
    }

def warm_start_fit(fitter : PhotonicLanternFitting, fractional_intensity : np.ndarray, total_intensity : np.ndarray, truth : np.ndarray) -> np.ndarray:
    '''
    Stage 4 only, started from the true binary. Measures the local bias of the binary fit.
    '''
    return fitter.binary_fitting_warm_start(fractional_intensity, total_intensity, truth, M_1).x

def full_fit(fitter : PhotonicLanternFitting, fractional_intensity : np.ndarray, total_intensity : np.ndarray, truth : np.ndarray, cpu : int | None = None) -> np.ndarray:
    '''
    Full binary pipeline (stages 1 to 4). Also captures the fit locking onto the wrong pair of stars.
    '''
    result = fitter.binary_fitting(fractional_intensity, total_intensity, LINE_GUESS, M_1, refine=True, cpu=cpu)

    # The pipeline returns the stage 1 result when a single star already fits, which counts as a failure to fit the binary
    if not isinstance(result, np.ndarray):
        return np.full_like(truth, np.nan)

    return result

def bisect_critical_flux(fit_function, fitter : PhotonicLanternFitting, star2_position : np.ndarray, star3_position : np.ndarray, sigma : np.ndarray, noise : float | None = None, seed : int = 0, **fit_kwargs) -> tuple[float, int]:
    '''
    Bisects in log flux for the star 3 flux at which any binary parameter is biased by more than K_SIGMA sigma.

    Returns
    -------
    F3_crit : float
        Critical star 3 flux [F_1] (the bracket edge if the fit never fails or always fails)
    limiting_param : int
        Index of the parameter with the largest bias in sigma at the first failing flux (-1 if the fit never fails)
    '''
    truth = true_binary_params(star2_position)

    def failed_parameter(F3):
        fractional_intensity, total_intensity = simulate_triple(fitter.lantern, star2_position, star3_position, F3, noise=noise, seed=seed)
        fit = fit_function(fitter, fractional_intensity, total_intensity, truth, **fit_kwargs)

        bias_in_sigma = np.abs(fit - truth) / sigma
        if np.any(np.isnan(bias_in_sigma)) or np.any(bias_in_sigma > K_SIGMA):
            return int(np.nanargmax(np.nan_to_num(bias_in_sigma, nan=np.inf)))
        return None

    lo, hi = F3_BRACKET

    limiting_param = failed_parameter(hi)
    if limiting_param is None:
        return hi, -1

    for _ in range(N_BISECTION):
        mid = np.sqrt(lo * hi)
        param = failed_parameter(mid)

        if param is None:
            lo = mid
        else:
            hi, limiting_param = mid, param

    return np.sqrt(lo * hi), limiting_param

#%% Running the survey
if __name__ == '__main__':
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--level', choices=['linear', 'warm', 'full'], default='linear', help='Linear estimate (no fitting), warm started stage 4, or the full pipeline')
    parser.add_argument('--r2', type=float, default=None, help='Projected distance between star 1 and star 2 in mas (default: all of r2s)')
    parser.add_argument('--theta2', type=float, default=None, help='Angle between star 2 and the x-axis in degrees (default: all of theta2s)')
    parser.add_argument('--stride', type=int, default=4, help='Fit every stride-th star 3 grid point along each axis (warm and full levels only)')
    parser.add_argument('--cpu', type=int, default=6, help='Worker processes for stage 2 of the full pipeline')
    args = parser.parse_args()

    # Creating the photonic lantern model
    photonic_lantern_device = PhotonicLanternDevice(
        N_DIM=N_DIM,
        DIAMETER=DIAMETER,
        FOCAL_RESOLUTION=FOCAL_RESOLUTION,
        WAV_MIDDLE=WAV_MIDDLE,
        N_CORE=N_CORE,
        N_CLAD=N_CLAD,
        R_CORE=R_CORE,
        FOCAL_LENGTH=FOCAL_LENGTH,
        N_SPEC=N_SPEC
    )

    fitter = PhotonicLanternFitting(photonic_lantern_device, WAV_RANGE)

    # Grid of positions for star 3
    x3s = np.linspace(-1*photonic_lantern_device.angular_extent, 1*photonic_lantern_device.angular_extent, N_GRID_3)
    y3s = np.linspace(-1*photonic_lantern_device.angular_extent, 1*photonic_lantern_device.angular_extent, N_GRID_3)

    if args.level != 'linear':
        x3s, y3s = x3s[::args.stride], y3s[::args.stride]

    X3S, Y3S = np.meshgrid(x3s, y3s) # [mas], indexed [y, x]
    star3_positions = np.column_stack([X3S.ravel(), Y3S.ravel()])

    r2_list = r2s if args.r2 is None else np.array([args.r2])
    theta2_list = theta2s if args.theta2 is None else np.array([args.theta2])

    F3_crit = np.zeros((len(r2_list), len(theta2_list), len(y3s), len(x3s)))
    limiting_param = np.zeros_like(F3_crit, dtype=int)
    F3_detect = np.full_like(F3_crit, np.nan)
    sigmas = np.zeros((len(r2_list), len(theta2_list), len(PARAM_NAMES)))

    for i, r2 in enumerate(r2_list):
        for j, theta2 in enumerate(theta2_list):
            star2_position = r2 * np.array([np.cos(np.deg2rad(theta2)), np.sin(np.deg2rad(theta2))])

            # The linear estimate also supplies the noise-limited sigma used as the failure threshold by the fitting levels
            linear = linear_estimate(fitter, star2_position, star3_positions)
            sigmas[i, j] = linear['sigma']

            if args.level == 'linear':
                F3_crit[i, j] = linear['F3_crit'].reshape(X3S.shape)
                limiting_param[i, j] = linear['limiting_param'].reshape(X3S.shape)
                F3_detect[i, j] = linear['F3_detect'].reshape(X3S.shape)
                continue

            fit_function = warm_start_fit if args.level == 'warm' else full_fit
            fit_kwargs = {'cpu': args.cpu} if args.level == 'full' else {}

            for k, star3_position in enumerate(tqdm(star3_positions, desc=f'{args.level}: r2 = {r2} mas, theta2 = {theta2} deg')):
                F3, param = bisect_critical_flux(fit_function, fitter, star2_position, star3_position, linear['sigma'], **fit_kwargs)

                F3_crit[i, j].flat[k] = F3
                limiting_param[i, j].flat[k] = param

    output_file = f'datafiles/triple_survey_{args.level}.npz'
    np.savez(output_file, r2s=r2_list, theta2s=theta2_list, x3s=x3s, y3s=y3s, F3_crit=F3_crit, limiting_param=limiting_param, F3_detect=F3_detect,
             sigmas=sigmas, param_names=PARAM_NAMES, K_SIGMA=K_SIGMA, K_DETECT=K_DETECT, NOISE=NOISE)
    print(f'Saved {output_file}')
