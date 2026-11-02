import sys
import subprocess

if sys.platform == 'darwin':
    # Starting caffeinate in the background for the duration of the script
    proc = subprocess.Popen(['caffeinate', '-i'])
    print('Caffeinate started for MacOS')
else:
    proc = None
    print('Caffeinate not started for non-MacOS')

# Importing the ncessary modules
from PLsim.otf import OTF
from PLsim.scene import Scene, SceneProjector
from PLsim.system.device import Device

import hcipy as hc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

import pandas as pd
from tqdm import tqdm
import pickle

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

#%% Input parameters and constants
THETA_MIN = -100 # [mas]
THETA_MAX = 100 # [mas]
THETA_NUM = 100
THETA_RANGE = np.linspace(THETA_MIN, THETA_MAX, THETA_NUM)

M2_MIN = 0.01 # [M_1]
M2_MAX = 1 # [M_1]
M2_NUM = 99
M2_RANGE = np.linspace(M2_MIN, M2_MAX, M2_NUM)

MAS_TO_RAD = 1e-3/206265 # [rad/mas]
DIVISOR_OUTPUT = 0

N_SPEC = 75 # Number of wavelength channels
WAV_MIDDLE = 1282.8 # [nm]
WAV_WIDTH = 0.1 # [nm]
WAV_BANDWIDTH = 10*WAV_WIDTH # [nm]
WAV_RANGE = np.linspace(WAV_MIDDLE - WAV_BANDWIDTH/2, WAV_MIDDLE + WAV_BANDWIDTH/2, N_SPEC)

#%% Defining functions

# Defining the mass-luminosity relation
def M_L_relation(M_1, M_2):
    if M_1 < 2:
        return (M_1/M_2)**5
    else:
        return (M_1/M_2)**3

# Creating function to calculate the normalized intensities of each output given photonic lantern outputs and divisor output
def calculate_normalized_intensities(outputs, divisor_output):
    '''
    Calculates the normalized intensities of each output given photonic lantern outputs and divisor output.as_integer_ratio

    Parameters
    ----------
    outputs : np.ndarray 
        Output from `device.calculate_outputs()` [number of wavelength channels, number of outputs]
    divisor_output : int
        Index of the output to normalize by

    Returns
    -------
    normalized_intensities : np.ndarray
        Normalized intensities of each output [number of outputs]
    '''
    # Initializing array to store the total normalized intensities
    tot_norm_outputs = []

    # Iterating through each photonic lantern output
    for i in range(outputs.shape[1]):
        # Calculating the continuum intensity for the output (basically far from the spectral features)
        continuum_intensity = (outputs[0, i] + outputs[1, i]) / 2

        # Calculating the total normalized intensity for the output
        integral = np.abs(np.sum( (outputs[:, i] - continuum_intensity)/outputs[:, divisor_output] * (WAV_RANGE[1] - WAV_RANGE[0]) ))

        tot_norm_outputs.append(integral)

    return np.array(tot_norm_outputs)

#%% Creating photonic lantern model

# Pixels in pupil grid (65 is enough for a 6-port PL)
ndim = 65

# Diameter of telescope aperture for pupil
diameter = 10 # [m]

# Create the pupil grid
pupil_grid = hc.make_pupil_grid(ndim, diameter)
print(pupil_grid)

# Resolution of focal grid (make sure this samples LP modes enough but also not truncate the outer parts)
focal_plane_resolution = 1e-6 # [m]

# Focal length of telescope (initial value, OTF.from_fiber_params will find an optimal focal ratio)
focal_length = 40 # [m]

# Create the focal grid
focal_grid = hc.make_pupil_grid(ndim, focal_plane_resolution * ndim)
print(focal_grid)

# Index of refraction of the core
n_core = 1.444

# Index of refraction of the cladding
n_clad = 1.444 - 5.5e-3

# Core radius
r_core = 10e-6 # [m]

# Reference wavelength of light
ref_wavelength = 1282.8e-9 # [m]
# ref_wavelength = 1.55e-6 # [m]

print(f'n_core: {n_core}')
print(f'n_clad: {n_clad}')
print(f'r_core: {r_core}')
print(f'ref_wavelength: {ref_wavelength}')

# Telescope aperture (circular) (a binary mask that has ones inside the aperture and zeros outside)
aperture = hc.make_circular_aperture(diameter)(pupil_grid)
print(aperture)

# Creating the optical transfer function (OTF)
otf = OTF.from_fiber_params(
    n_core, n_clad, r_core, ref_wavelength,
    focal_grid, pupil_grid,
    focal_length, aperture,
    optimize_focal_length=False
)
print(otf)

# Defining number of outputs of photonic lantern
N_OUTPUTS = 10

# Computing OTF
otf.compute()

# Importing the function for generating a unitary matrix
from PLsim.system.lantern import generate_unitary_matrix

# Initializing matrix (complex)
matrix = generate_unitary_matrix(N_OUTPUTS)[np.newaxis, :, :].repeat(N_SPEC, axis=0)
print(matrix.shape)

# Initializing device with the unitary matrix
device = Device(matrix)

# Defining scene
scene = Scene(pupil_grid, ref_wavelength)

# Creating a projection onto the OTF
scene_projector = SceneProjector(otf, scene)

#%% Creating the input scene for a binary star

# Mass of the stars
M_1 = 1 # [M_sun]
M_2 = M2_RANGE # [M_sun]

# Flux of the continuum of star 1
F_1 = 1

# Flux of star 2
F_2 = F_1 / M_L_relation(M_1, M_2).reshape(-1, 1)
print(f'Flux of star 2 shape: {F_2.shape}')

# Flux multiplier of the line
a_line_1 = 0.9 # Paschen beta line is 90% of the continuum normally...
a_line_2 = 0.9 # Paschen beta line is 90% of the continuum normally...

# Flux of the line of star 1
F_line_1 = a_line_1 * F_1

# Flux of the line of star 2
F_line_2 = a_line_2 * F_2

# Spectrum of star 1
star1_spectrum = F_1 * np.ones_like(WAV_RANGE)
star1_spectrum -= (F_1 - F_line_1) * np.exp(-0.5 * (WAV_RANGE - WAV_MIDDLE)**2 / WAV_WIDTH**2)

# Spectrum of star 2
star2_spectrum = np.dot(F_2, np.ones_like(WAV_RANGE).reshape(1, -1))
star2_spectrum -= (F_2 - F_line_2) * np.exp(-0.5 * (WAV_RANGE - WAV_MIDDLE)**2 / WAV_WIDTH**2).reshape(1, -1)
print(f'Spectrum of star 2 shape: {star2_spectrum.shape}')

#%% Passing the different star positions and spectra into photonic lantern

# Converting angular separations into radians
theta_range_rad = THETA_RANGE * MAS_TO_RAD # [rad]

# Star 1 position
star1_x = np.zeros_like(theta_range_rad)
star1_y = np.zeros_like(theta_range_rad)

# Star 2 position
star2_x = theta_range_rad
star2_y = theta_range_rad

combined_total_normalized_intensities = []

# Iterating through different masses
for i in tqdm(range(M2_NUM)):
    constant_mass_2 = []

    # Iterating through different star x positions
    for j in range(len(star2_x)):
        constant_x_position = []

        for k in range(len(star2_y)):
            # Projecting the star onto the OTF
            star1_projected = scene_projector.compute_point(star1_x[j], star1_y[k]).repeat(N_SPEC, axis=2) * star1_spectrum[np.newaxis, np.newaxis]
            star2_projected = scene_projector.compute_point(star2_x[j], star2_y[k]).repeat(N_SPEC, axis=2) * star2_spectrum[i, :][np.newaxis, np.newaxis]

            # Combining the projected star fields
            combined_projected = star1_projected + star2_projected

            # Passing the combined projected star through the device
            combined_outputs = device.calculate_outputs(combined_projected.transpose(2, 0, 1))

            # Storing the total intensities for each output for the combined star field
            constant_x_position.append(calculate_normalized_intensities(combined_outputs, DIVISOR_OUTPUT))

        constant_mass_2.append(constant_x_position)

    combined_total_normalized_intensities.append(constant_mass_2)

combined_total_normalized_intensities = np.array(combined_total_normalized_intensities)

print('(mass, x, y, output)')
print(combined_total_normalized_intensities.shape)

# Saving the results to a pickle file
with open('binary_results.pkl', 'wb') as f:
    pickle.dump(combined_total_normalized_intensities, f)

#%% Plotting the results
log_intensities = np.log10(combined_total_normalized_intensities)
finite_values = log_intensities[np.isfinite(log_intensities)]
vmin, vmax = finite_values.min(), finite_values.max()

images = []

fig = plt.figure(figsize=(24, 10))
fig.set_constrained_layout(True)
axes = fig.subplots(ncols=5, nrows=2, sharex=True, sharey=True)
axes = axes.flatten()

# Plotting first frame
fig.suptitle(f'Total Normalized Intensities of Outputs, $M_2 = {M2_RANGE[0]:.2f} M_1$')

for i in range(N_OUTPUTS):
    image = axes[i].imshow(log_intensities[0, :, :, i], extent=[THETA_RANGE[0], THETA_RANGE[-1], THETA_RANGE[0], THETA_RANGE[-1]], aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    images.append(image)
    axes[i].set_title(f'Output {i+1}')
    axes[i].set_xlim(THETA_RANGE[0], THETA_RANGE[-1])
    axes[i].set_ylim(THETA_RANGE[0], THETA_RANGE[-1])

    # Plotting diffraction limit
    diffraction_limit_patch = patches.Circle((0, 0), 1.22 * ref_wavelength / diameter / MAS_TO_RAD, color='k', fill=False, linestyle='--', linewidth=1)
    axes[i].add_patch(diffraction_limit_patch)

    if i in [5, 6, 7, 8, 9]:
        axes[i].set_xlabel('$M_2/M_1$')

    if i in [0, 5]:
        axes[i].set_ylabel('Separation [mas]')

    axes[i].set_aspect('equal')

# Plotting colorbar
fig.colorbar(axes[0].imshow(log_intensities[0, :, :, 0], extent=[THETA_RANGE[0], THETA_RANGE[-1], THETA_RANGE[0], THETA_RANGE[-1]], aspect='auto', origin='lower', vmin=vmin, vmax=vmax), ax=axes, location='right', label=r'$\log_{10}$(Normalized Intensity)', pad=0.01, shrink=0.89)

# Defining the update function for the animation
def update(frame):
    new_data = log_intensities[frame, :, :, :]
    
    for i in range(N_OUTPUTS):
        images[i].set_data(new_data[:, :, i])
        axes[i].set_aspect('equal')
    
    fig.suptitle(f'Total Normalized Intensities of Outputs, $M_2 = {M2_RANGE[frame]:.2f} M_1$')
    
    return images

# Creating the animation object
ani = animation.FuncAnimation(fig, update, frames=M2_NUM, interval=100, blit=True)

# Saving the animation
ani.save('binary_animation.mp4', writer='ffmpeg', fps=1)

# plt.show()