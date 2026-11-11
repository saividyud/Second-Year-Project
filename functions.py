# Importing the necessary modules
import numpy as np

from PLsim.otf import OTF
from PLsim.scene import Scene, SceneProjector
from PLsim.system.device import Device
from PLsim.system.lantern import generate_unitary_matrix

import hcipy as hc

# Defining the mass-luminosity relation
def M_L_relation(M_1, M_2) -> float:
    if M_1 < 2:
        return (M_1/M_2)**5
    else:
        return (M_1/M_2)**3

# Function to define photonic lantern device
def create_photonic_lantern_device(
    N_DIM,
    DIAMETER,
    FOCAL_RESOLUTION,
    WAV_MIDDLE,
    N_CORE,
    N_CLAD,
    R_CORE,
    FOCAL_LENGTH,
    N_SPEC) -> tuple[
        hc.CartesianGrid, 
        hc.CartesianGrid, 
        hc.Field, 
        OTF,
        int,
        np.ndarray, 
        Device, 
        Scene, 
        SceneProjector]:
    '''
    Creates a photonic lantern device with the given parameters.

    Parameters
    ----------
    N_DIM : int
        Number of pixels in the pupil grid
    DIAMETER : float
        Diameter of the telescope aperture for pupil grid [m]
    FOCAL_RESOLUTION : int
        Resolution of the focal grid (make sure this samples LP modes enough but also not truncate the outer parts) [m]
    WAV_MIDDLE : float
        Middle wavelength of the wavelength range [nm]
    N_CORE : float
        Index of refraction of the core
    N_CLAD : float
        Index of refraction of the cladding
    R_CORE : float
        Core radius [m]
    FOCAL_LENGTH : float
        Focal length of the telescope (initial value, OTF.from_fiber_params will find an optimal focal ratio) [m]
    N_SPEC : int
        Number of wavelength channels

    Returns
    -------
    pupil_grid : Grid
        Pupil grid
    focal_grid : Grid
        Focal grid
    aperture : BinaryMask
        Aperture of the fiber
    otf : OTF
        Optical transfer function
    N_OUTPUTS : int
        Number of photonic lantern outputs
    matrix : np.ndarray
        Unitary matrix
    device : Device
        Photonic lantern device
    scene : Scene
        Scene
    scene_projector : SceneProjector
        Projection onto the optical transfer function
    '''
    # Create the pupil grid
    pupil_grid = hc.make_pupil_grid(N_DIM, DIAMETER)
    print('Created pupil grid')

    # Create the focal grid
    focal_grid = hc.make_pupil_grid(N_DIM, FOCAL_RESOLUTION * N_DIM)
    print('Created focal grid\n')

    # Reference wavelength of light
    ref_wavelength = WAV_MIDDLE * 1e-9 # [m]

    print('Fiber details:')
    print(f'n_core: {N_CORE}')
    print(f'n_clad: {N_CLAD}')
    print(f'r_core: {R_CORE} m')
    print(f'ref_wavelength: {ref_wavelength} m\n')

    # Telescope aperture (circular) (a binary mask that has ones inside the aperture and zeros outside)
    aperture = hc.make_circular_aperture(DIAMETER)(pupil_grid)
    print('Created aperture')

    # Creating the optical transfer function (OTF)
    otf = OTF.from_fiber_params(
        N_CORE, N_CLAD, R_CORE, ref_wavelength,
        focal_grid, pupil_grid,
        FOCAL_LENGTH, aperture,
        optimize_focal_length=False
    )
    N_OUTPUTS = otf.focal_modes.shape[0]
    print('Created optical transfer function\n')

    # Computing OTF
    otf.compute()
    print('Computed optical transfer function\n')

    # Initializing matrix (complex)
    matrix = generate_unitary_matrix(N_OUTPUTS)[np.newaxis, :, :].repeat(N_SPEC, axis=0)
    print(f'Generated unitary matrix of shape: (N_SPEC x N_OUTPUTS x N_OUTPUTS) -> {matrix.shape}\n')

    # Initializing device with the unitary matrix
    device = Device(matrix)
    print('Initialized device with the unitary matrix')

    # Defining scene
    scene = Scene(pupil_grid, ref_wavelength)
    print('\nDefined scene')

    # Creating a projection onto the OTF
    scene_projector = SceneProjector(otf, scene)
    print('Projected scene onto optical transfer function\n')

    return pupil_grid, focal_grid, aperture, otf, N_OUTPUTS, matrix, device, scene, scene_projector

# Creating function to calculate the normalized intensities of each output given photonic lantern outputs and divisor output
def calculate_normalized_intensities(outputs, divisor_output, wav_range) -> np.ndarray:
    '''
    Calculates the normalized intensities of each output given photonic lantern outputs and divisor output.as_integer_ratio

    Parameters
    ----------
    outputs : np.ndarray 
        Output from `device.calculate_outputs()` [number of wavelength channels, number of outputs]
    divisor_output : int
        Index of the output to normalize by
    wav_range : np.ndarray
        Wavelength range [number of wavelength channels]

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
        integral = np.abs(np.sum( (outputs[:, i] - continuum_intensity)/outputs[:, divisor_output] * (wav_range[1] - wav_range[0]) ))

        tot_norm_outputs.append(integral)

    return np.array(tot_norm_outputs)

# Function to calculate the spectroastrometric signal according to option 2
def calculate_signal(outputs, wav_range) -> np.ndarray:
    '''
    Calculates the signal of each output given photonic lantern outputs and divisor output.

    Parameters
    ----------
    outputs : np.ndarray 
        Output from `device.calculate_outputs()` [number of wavelength channels, number of outputs]
    wav_range : np.ndarray
        Wavelength range [number of wavelength channels]

    Returns
    -------
    signals : np.ndarray
        Signals of each output [number of outputs]
    '''
    # Initializing array to store the signals
    signals = []

    # Finding the total intensity at each wavelength channel for all the outputs
    total_intensity = np.sum(outputs, axis=1)

    # Iterating through each photonic lantern output
    for i in range(outputs.shape[1]):
        # Calculating fractional intensity of the port (port intensity/total intensity)
        fractional_intensity = outputs[:, i] / total_intensity

        # Calculating the continuum fractional intensity for the output (basically far from the spectral features)
        continuum_fractional_intensity = (fractional_intensity[0] + fractional_intensity[-1]) / 2

        # Calculating the signal for the output
        integral = np.sum( np.abs((fractional_intensity - continuum_fractional_intensity) * (wav_range[1] - wav_range[0])) )

        signals.append(integral)

    return np.array(signals)

# Creating a spectrum generating function
def create_spectrum(cont_flux, line_depth, line_middle, line_width, wavelength_range) -> np.ndarray:
    '''
    Creates a Gaussian absorption or emission line spectrum with the given parameters.

    Parameters
    ----------
    cont_flux : float
        Flux of the continuum [arbitrary units]
    line_depth : float
        Depth of the line (fraction of continuum flux)
    line_middle : float
        Middle wavelength of the line [length units]
    line_width : float
        Width of the line (FWHM) [length units]
    wavelength_range : np.ndarray
        Wavelength range [length units]

    Returns
    -------
    spectrum : np.ndarray
        Spectrum [arbitrary units] (same length as wavelength_range)
    '''
    # Creating the spectrum
    spectrum = cont_flux * np.ones_like(wavelength_range)
    spectrum -= cont_flux * (1 - line_depth) * np.exp(-0.5 * (wavelength_range - line_middle)**2 / line_width**2)

    return spectrum