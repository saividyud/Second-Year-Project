# Importing the necessary modules
import numpy as np

from PLsim.otf import OTF
from PLsim.scene import Scene, SceneProjector
from PLsim.system.device import Device
from PLsim.system.lantern import generate_unitary_matrix

import hcipy as hc

# Custom libraries
import functions

class PhotonicLanternDevice:
    '''
    Class to create a photonic lantern device with the given parameters with helper functions.

    Parameters
    ----------
    N_DIM : int
        Number of pixels in pupil grid
    DIAMETER : float
        Diameter of the telescope aperture for pupil grid [m]
    FOCAL_RESOLUTION : float
        Resolution of the focal grid (make sure this samples LP modes enough but also not truncate the outer parts) [m]
    FOCAL_LENGTH : float
        Focal length of the telescope (initial value, OTF.from_fiber_params will find an optimal focal ratio) [m]
    N_CORE : float
        Index of refraction of the core
    N_CLAD : float
        Index of refraction of the cladding
    R_CORE : float
        Core radius [m]
    N_SPEC : int
        Number of wavelength channels
    WAV_MIDDLE : float
        Middle wavelength of the wavelength range [nm]
    '''
    def __init__(self, N_DIM=65, DIAMETER=10, FOCAL_RESOLUTION=1e-6, FOCAL_LENGTH=40,
                N_CORE=1.444, N_CLAD=1.444-5.5e-3, R_CORE=10e-6, 
                N_SPEC=150, WAV_MIDDLE=1282.8):
        
        # Initializing the attributes
        self.N_DIM = N_DIM
        self.DIAMETER = DIAMETER
        self.FOCAL_RESOLUTION = FOCAL_RESOLUTION
        self.FOCAL_LENGTH = FOCAL_LENGTH
        self.N_CORE = N_CORE
        self.N_CLAD = N_CLAD
        self.R_CORE = R_CORE
        self.N_SPEC = N_SPEC
        self.WAV_MIDDLE = WAV_MIDDLE

        # Defining some useful constants
        self.MAS_TO_RAD = 1e-3/206265 # [rad/mas]

        # Creating the photonic lantern device
        (self.pupil_grid, 
        self.focal_grid, 
        self.aperture, 
        self.otf, 
        self.N_PORTS,
        self.matrix, 
        self.device, 
        self.scene, 
        self.scene_projector) = self.create_photonic_lantern_device(
            N_DIM=self.N_DIM,
            DIAMETER=self.DIAMETER,
            FOCAL_RESOLUTION=self.FOCAL_RESOLUTION,
            WAV_MIDDLE=self.WAV_MIDDLE,
            N_CORE=self.N_CORE,
            N_CLAD=self.N_CLAD,
            R_CORE=self.R_CORE,
            FOCAL_LENGTH=self.FOCAL_LENGTH,
            N_SPEC=self.N_SPEC)

        # Calculating the angular extent of the photonic lantern device
        self.diffraction_limit = 1.22 * self.WAV_MIDDLE * 1e-9 / self.DIAMETER / self.MAS_TO_RAD # [mas]
        self.angular_extent = self.R_CORE / self.FOCAL_LENGTH / self.MAS_TO_RAD + self.diffraction_limit # [mas]

    def create_photonic_lantern_device(
        self,
        N_DIM : int,
        DIAMETER : float,
        FOCAL_RESOLUTION : float,
        WAV_MIDDLE : float,
        N_CORE : float,
        N_CLAD : float,
        R_CORE : float,
        FOCAL_LENGTH : float,
        N_SPEC : int) -> tuple[
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
        N_PORTS : int
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
        return functions.create_photonic_lantern_device(
            N_DIM=N_DIM,
            DIAMETER=DIAMETER,
            FOCAL_RESOLUTION=FOCAL_RESOLUTION,
            WAV_MIDDLE=WAV_MIDDLE,
            N_CORE=N_CORE,
            N_CLAD=N_CLAD,
            R_CORE=R_CORE,
            FOCAL_LENGTH=FOCAL_LENGTH,
            N_SPEC=N_SPEC)

    def calculate_photonic_lantern_outputs(self, positions : np.ndarray, spectra : np.ndarray, noise : float | None = None, seed : float | int = 0) -> np.ndarray:
        '''
        Parameters
        ----------
        positions : np.ndarray
            Array of positions [mas] (N_BODIES x 2)
        spectra : np.ndarray
            Array of spectra over the same wavelength range [arbitrary units] (N_BODIES x N_SPEC)

        Returns
        -------
        outputs : np.ndarray
            Array of output spectra from each photonic lantern port (N_SPEC x N_PORTS)
        '''
        # Determining the number of bodies in the scene
        N_BODIES = len(positions)

        # Initializing list of projected bodies
        combined_projected = np.zeros((self.N_PORTS, self.N_PORTS, self.N_SPEC), dtype=np.complex128)

        # If a noise level is provided, initializing the random number generator
        rng = np.random.default_rng(seed)

        if noise is not None:
            NOISE = noise
        else:
            NOISE = 0

        # Iterating through each body in the scene
        for i in range(N_BODIES):
            # Projecting the body onto the OTF
            body_projected = self.scene_projector.compute_point(positions[i, 0] * self.MAS_TO_RAD, positions[i, 1] * self.MAS_TO_RAD).repeat(self.N_SPEC, axis=2) * spectra[i, :][np.newaxis, np.newaxis]

            # Adding the projected body to the list
            combined_projected += body_projected

        # Passing the combined projected bodies 
        outputs = self.device.calculate_outputs(combined_projected.transpose(2, 0, 1))
        outputs *= 1 + NOISE * rng.standard_normal(outputs.shape)

        return outputs
