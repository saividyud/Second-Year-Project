# Importing the necessary modules
import numpy as np

from PLsim.otf import OTF
from PLsim.scene import Scene, SceneProjector
from PLsim.system.device import Device
from PLsim.system.lantern import generate_unitary_matrix

import hcipy as hc

# Defining the mass-luminosity relation
def M_L_relation(M_1, M_2):
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
    N_OUTPUTS, N_SPEC) -> tuple[
        hc.CartesianGrid, 
        hc.CartesianGrid, 
        hc.Field, 
        OTF, 
        np.ndarray, 
        Device, 
        Scene, 
        SceneProjector]:
    '''
    Creates a photonic lantern device with the given parameters.

    Parameters
    ----------
    N_DIM : int
        Number of dimensions of the pupil grid
    DIAMETER : float
        Diameter of the pupil grid
    FOCAL_RESOLUTION : int
        Resolution of the focal grid
    WAV_MIDDLE : float
        Middle wavelength of the wavelength range
    N_CORE : float
        Core index of the fiber
    N_CLAD : float
        Clad index of the fiber
    R_CORE : float
        Core radius of the fiber
    FOCAL_LENGTH : float
        Focal length of the fiber
    N_OUTPUTS : int
        Number of outputs of the photonic lantern
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

    return pupil_grid, focal_grid, aperture, otf, matrix, device, scene, scene_projector

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