# Importing the necessary modules
import numpy as np
import scipy.optimize as optimize
import pandas as pd

from PLsim.otf import OTF
from PLsim.scene import Scene, SceneProjector
from PLsim.system.device import Device
from PLsim.system.lantern import generate_unitary_matrix

import hcipy as hc

from tqdm import tqdm
import time as t

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

    def _single_star_objective(self, x : tuple, observed_fractional_intensities : np.ndarray) -> float:
        '''
        Calculates the objective function for a single star fit to observed photonic lantern output spectra.

        Parameters
        ----------
        x : tuple
            Tuple of (x, y) position for the first star in mas
        observed_fractional_intensities : np.ndarray
            (N_SPEC x N_PORTS) matrix of observed fractional intensities from photonic lantern

        Returns
        -------
        delta : float
            Error norm of the errors of each port given by the integral over wavelengths of the difference between binary and single fractional intensities
        '''
        star_x, star_y = x
        star_position = np.array([star_x, star_y])
        star_flat_spectrum = np.ones(self.N_SPEC) # [arbitrary units]

        # Calculating the outputs of the star
        star_outputs = self.calculate_photonic_lantern_outputs(np.array([star_position]), np.array([star_flat_spectrum]))

        # Computing the total intensities across all outputs as a function of wavelength
        single_total_intensities = np.sum(star_outputs, axis=1)

        # Computing fractional intensities for the outputs
        single_fractional_intensities = star_outputs / single_total_intensities.reshape(-1, 1)

        # Calculating residuals
        residuals = observed_fractional_intensities - single_fractional_intensities # (N_SPEC, N_PORTS)
        delta = np.sum(residuals**2)

        return delta

    def single_fitting(self, observed_fractional_intensities : np.ndarray, bounded_edge : float | None = None) -> optimize.OptimizeResult:
        '''
        Minimizes the objective function `self._single_star_objective` for a single star fit to observed photonic lantern output spectra.

        Parameters
        ----------
        observed_fractional_intensities : np.ndarray
            (N_SPEC x N_PORTS) matrix of observed fractional intensities from photonic lantern
        bounded_edge : float, optional
            Edge of the bounded region [mas], default None

        Returns
        -------
        result : optimize.OptimizeResult
            Result of the minimization
        '''
        # Defining the bounds of the minimization
        if bounded_edge is not None:
            bounds = [(-bounded_edge, bounded_edge), (-bounded_edge, bounded_edge)]
        else:
            bounds = [(-self.angular_extent, self.angular_extent), (-self.angular_extent, self.angular_extent)]

        # Executing the minimization
        result = optimize.differential_evolution(
            lambda x: self._single_star_objective(x, observed_fractional_intensities),
            bounds)

        return result

    def _solve_spectra(self, star1_flat_output : np.ndarray, star2_flat_output : np.ndarray, observed_fractional_intensities : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''
        Solves the spectra using linear least squares.

        Parameters
        ----------
        star1_flat_output : np.ndarray
            (N_SPEC x N_OUTPUTS) matrix of outputs from the first star given a flat spectrum of ones
        star2_flat_output : np.ndarray
            (N_SPEC x N_OUTPUTS) matrix of outputs from the second star given a flat spectrum of ones
        observed_fractional_intensities : np.ndarray
            (N_SPEC x N_OUTPUTS) matrix of observed fractional intensities from photonic lantern

        Returns
        -------
        spectra : np.ndarray
            (N_SPEC x 2) matrix of spectra for the stars
        model_intensities : np.ndarray
            (N_SPEC x N_OUTPUTS) matrix of model fractional intensities from photonic lantern
        '''
        # Calculating weight matrix A (strength of the photonic lantern for the different positions and wavelengths)
        A = np.stack([star1_flat_output, star2_flat_output], axis=2) # (N_SPEC, N_OUTPUTS, 2)
        
        # Calculating the covariance matrix M = A^T * A
        M = np.einsum('lni,lnj->lij', A, A) # (N_SPEC, 2, 2)

        # Calculating the right hand side vector v = A^T * f
        v = np.einsum('lni,ln->li', A, observed_fractional_intensities) # (N_SPEC, 2)

        # Solving the system of equations M * c = v, where c is the vector of spectra of the stars
        spectra = np.linalg.solve(M, v[..., None])[..., 0] # (N_SPEC, 2)

        # Calculating the model intensities
        model_intensities = np.einsum('lni,li->ln', A, spectra) # (N_SPEC, N_OUTPUTS)

        return spectra, model_intensities

    def _position_residuals(self, x : np.ndarray, observed_fractional_intensities : np.ndarray) -> np.ndarray:
        '''
        Flattened resodial vector for position fitting with the spectra solved linearly within.

        Parameters
        ----------
        x : np.ndarray
            Array of (x, y) positions of both the stars [mas] (x1, y1, x2, y2)
        observed_fractional_intensities : np.ndarray
            (N_SPEC x N_OUTPUTS) matrix of observed fractional intensities from photonic lantern

        Returns
        -------
        residuals : np.ndarray
            Flattened residual vector of the differences between the observed and model fractional intensities
        '''
        # Extracting positions of the stars
        star1_x, star1_y, star2_x, star2_y = x
        star1_position = np.array([star1_x, star1_y])
        star2_position = np.array([star2_x, star2_y])

        # Creating a flat spectrum for the stars
        flat_spectrum = np.ones(self.N_SPEC) # [arbitrary units]

        # Calculating the outputs of the star positions
        star1_flat_output = self.calculate_photonic_lantern_outputs(np.array([star1_position]), np.array([flat_spectrum]))
        star2_flat_output = self.calculate_photonic_lantern_outputs(np.array([star2_position]), np.array([flat_spectrum]))

        # Solving the spectra using linear least squares
        spectra, model_intensities = self._solve_spectra(star1_flat_output, star2_flat_output, observed_fractional_intensities)

        # Calculating the residuals
        residuals = observed_fractional_intensities - model_intensities
        
        return residuals.ravel()

    def _binary_fitting_stage2(self, stage1 : optimize.OptimizeResult, observed_fractional_intensities : np.ndarray, bounded_edge : float | None = None):
        '''
        Uses the fitted single star position and a grid of Star 2 positions to find local minima and solve for spectra using linear least squares.

        Parameters
        ----------
        stage1 : optimize.OptimizeResult
            Result of the single star fit
        observed_fractional_intensities : np.ndarray
            (N_SPEC x N_OUTPUTS) matrix of observed fractional intensities from photonic lantern
        bounded_edge : float, optional
            Edge of the bounded region [mas], default None

        Returns
        -------
        stage2 : optimize.OptimizeResult
        '''
        # Defining a grid of Star 2 positions
        if bounded_edge is None:
            bounded_edge = self.angular_extent

        x2_grid = np.linspace(-bounded_edge, bounded_edge, 10)
        y2_grid = np.linspace(-bounded_edge, bounded_edge, 10)

        # Iterating through each Star 2 position and finding the best fit
        best_fit = None

        with tqdm(total=len(x2_grid) * len(y2_grid), desc='Binary fitting stage 2 (position fitting)') as pbar:
            for x2 in x2_grid:
                for y2 in y2_grid:
                    position = np.array([stage1.x[0], stage1.x[1], x2, y2])

                    fit = optimize.least_squares(
                        lambda x: self._position_residuals(x, observed_fractional_intensities),
                        x0=position,
                        bounds=([-bounded_edge] * 4, [bounded_edge] * 4),
                        x_scale=10 # Sets the scale of the position fitting in mas
                    )

                    if best_fit is None or fit.cost < best_fit.cost:
                        best_fit = fit

                    pbar.update(1)

        return best_fit

    def _binary_fitting_stage3(self, stage2 : optimize.OptimizeResult, observed_fractional_intensities : np.ndarray, observed_total : np.ndarray, wavelength_range : np.ndarray, guess : np.ndarray) -> tuple[optimize.OptimizeResult, optimize.OptimizeResult, np.ndarray]:
        '''
        Calculates the spectra and fits the line properties of both stars, as well as calculates the mass ratio.

        Parameters
        ----------
        stage2 : optimize.OptimizeResult
            Result of the binary fitting stage 2
        observed_fractional_intensities : np.ndarray
            (N_SPEC x N_OUTPUTS) matrix of observed fractional intensities from photonic lantern
        observed_total : np.ndarray
            (N_SPEC,) total observed intensity summed over all outputs (before normalizing to fractions)
        wavelength_range : np.ndarray
            Wavelength range [nm]
        guess : np.ndarray
            Guess for the line properties of Star 1 and Star 2 (depth, center, width) (2 x 3)

        Returns
        -------
        line1_result : optimize.OptimizeResult
            Line fit of Star 1 (the brighter star)
        line2_result : optimize.OptimizeResult
            Line fit of Star 2
        best_positions : np.ndarray
            Positions of the stars ordered to match the line fits (x1, y1, x2, y2) [mas]
        '''
        # Extracting the spectra of the stars
        best_positions = stage2.x

        flat_spectrum = np.ones(self.N_SPEC) # [arbitrary units]

        star1_flat_output = self.calculate_photonic_lantern_outputs(np.array([[best_positions[0], best_positions[1]]]), np.array([flat_spectrum]))
        star2_flat_output = self.calculate_photonic_lantern_outputs(np.array([[best_positions[2], best_positions[3]]]), np.array([flat_spectrum]))

        spectra, model_intensities = self._solve_spectra(star1_flat_output, star2_flat_output, observed_fractional_intensities)

        # Finding the brighter of the two stars to call it "Star 1"
        if np.median(spectra[:, 1]) > np.median(spectra[:, 0]):
            spectra = spectra[:, ::-1]
            best_positions = np.r_[best_positions[2:], best_positions[:2]]

        # Calculating the recovered spectra
        self.recovered_s1 = spectra[:, 0] * observed_total # (N_SPEC,)
        self.recovered_s2 = spectra[:, 1] * observed_total # (N_SPEC,)

        # Constructing guesses
        guess1 = (np.median(self.recovered_s1), guess[0, 0], guess[0, 1], guess[0, 2])
        guess2 = (np.median(self.recovered_s2), guess[1, 0], guess[1, 1], guess[1, 2])

        # Fitting the line properties of Star 1 and Star 2
        line1_result = self._fit_line(self.recovered_s1, guess1, wavelength_range)
        line2_result = self._fit_line(self.recovered_s2, guess2, wavelength_range)

        return line1_result, line2_result, best_positions

    def _fit_line(self, spectrum : np.ndarray, guess : tuple[float, float, float, float], wavelength_range : np.ndarray) -> optimize.OptimizeResult:
        '''
        Fits the line properties of a star to a recovered spectrum.

        Parameters
        ----------
        spectrum : np.ndarray
            Recovered spectrum of the star [arbitrary units]
        guess : tuple[float, float, float, float]
            Guess for the line properties of the star (continuum flux, depth, center, width)
        wavelength_range : np.ndarray
            Wavelength range [nm]

        Returns
        -------
        result : optimize.OptimizeResult
            Result of the minimization
        '''
        return optimize.least_squares(
            fun=lambda q: functions.create_spectrum(q[0], q[1], q[2], q[3], wavelength_range) - spectrum, # (continuum flux, depth, center, width)
            x0=guess,
            bounds=([0, 0, wavelength_range[0], 1e-6], [np.inf, 2, wavelength_range[-1], 1]) # (continuum flux, depth, center, width)
        )

    def _binary_fitting_stage4(self, stage3 : tuple[optimize.OptimizeResult, optimize.OptimizeResult, np.ndarray], observed_fractional_intensities : np.ndarray, observed_total : np.ndarray, M1 : float, wavelength_range : np.ndarray, bounded_edge : float | None = None) -> optimize.OptimizeResult:
        '''
        Refines all the parameters of the binary (positions and both line profiles) using nonlinear least squares.

        Parameters
        ----------
        stage3 : tuple[optimize.OptimizeResult, optimize.OptimizeResult, np.ndarray]
            Results of the binary fitting stage 3 (line fit of Star 1, line fit of Star 2, ordered positions)
        observed_fractional_intensities : np.ndarray
            (N_SPEC x N_OUTPUTS) matrix of observed fractional intensities from photonic lantern
        observed_total : np.ndarray
            (N_SPEC,) total observed intensity summed over all outputs (before normalizing to fractions)
        M1 : float
            Mass of Star 1
        wavelength_range : np.ndarray
            Wavelength range [nm]
        bounded_edge : float, optional
            Edge of the bounded region [mas], default None

        Returns
        -------
        stage4 : optimize.OptimizeResult
            Result of the minimization with parameters (x1, y1, x2, y2, F1, a1, c1, w1, M2, a2, c2, w2)
        '''
        # Extracting the line fits and positions of the stars
        line1_result, line2_result, best_positions = stage3

        # Guess for M2
        if M1 < 2:
            fitted_M2 = M1 * (line2_result.x[0] / line1_result.x[0])**(1/5) # (M_2)
        else:
            fitted_M2 = M1 * (line2_result.x[0] / line1_result.x[0])**(1/3) # (M_2)

        F1_guess = line1_result.x[0]

        if bounded_edge is None:
            bounded_edge = self.angular_extent

        # Performing the least squares fit
        stage4 = optimize.least_squares(
            fun=lambda params: self._binary_objective(params, observed_fractional_intensities, observed_total, M1, wavelength_range),
            x0=(
                # [x1, y1, x2, y2, F1, depth of line 1, center of line 1, width of line 1, M2, depth of line 2, center of line 2, width of line 2]
                *best_positions, F1_guess, *line1_result.x[1:], fitted_M2, *line2_result.x[1:]
            ),
            bounds=(
                [-bounded_edge, -bounded_edge, -bounded_edge, -bounded_edge, 0, 0, wavelength_range[0], 1e-6, 0, 0, wavelength_range[0], 1e-6],
                [bounded_edge, bounded_edge, bounded_edge, bounded_edge, np.inf, 2, wavelength_range[-1], 1, np.inf, 2, wavelength_range[-1], 1]
            ),
            x_scale=[10, 10, 10, 10, 0.1 * F1_guess, 0.5, 0.05, 0.05, 0.5, 0.5, 0.05, 0.05],
        )

        return stage4

    def _binary_objective(self, params : np.ndarray, observed_fractional_intensities : np.ndarray, observed_total : np.ndarray, M1 : float, wavelength_range : np.ndarray) -> np.ndarray:
        '''
        Calculates the objective function for the binary fitting.

        Parameters
        ----------
        params : np.ndarray
            Parameters of the binary fitting (x1, y1, x2, y2, F1, a1, c1, w1, M2, a2, c2, w2)
        observed_fractional_intensities : np.ndarray
            (N_SPEC x N_OUTPUTS) matrix of observed fractional intensities from photonic lantern
        observed_total : np.ndarray
            (N_SPEC,) total observed intensity summed over all outputs (before normalizing to fractions)
        M1 : float
            Mass of Star 1
        wavelength_range : np.ndarray
            Wavelength range [nm]

        Returns
        -------
        residuals : np.ndarray
            Flattened residual vector of the differences between the observed and model intensities, normalized by the observed total intensity
        '''
        # Extracting the parameters
        x1, y1, x2, y2, F_1, depth1, center1, width1, M2, depth2, center2, width2 = params

        # Creating the spectra of the stars
        F_2 = F_1 / functions.M_L_relation(M1, M2)

        star1_spectrum = functions.create_spectrum(F_1, depth1, center1, width1, wavelength_range)
        star2_spectrum = functions.create_spectrum(F_2, depth2, center2, width2, wavelength_range)

        # Calculating the outputs of the star positions
        model_intensities = self.calculate_photonic_lantern_outputs(np.array([[x1, y1], [x2, y2]]), np.array([star1_spectrum, star2_spectrum]))

        # Comparing absolute intensities (not fractions) so that the overall flux scale, and therefore Star 1's spectrum, is constrained.
        # Dividing by the observed total keeps the residuals on the same scale as the fractional intensities.
        residuals = model_intensities / observed_total[:, None] - observed_fractional_intensities

        return residuals.ravel()

    def binary_fitting(self, observed_fractional_intensities : np.ndarray, observed_total : np.ndarray, guess : np.ndarray, M1 : float, wavelength_range : np.ndarray, bounded_edge : float | None = None, single_star_tolerance : float = 1e-3, refine : bool = True, true_values : list[float] = None) -> np.ndarray:
        '''
        Fits a binary signal to the observed photonic lantern outputs.T

        Parameters
        ----------
        observed_fractional_intensities : np.ndarray
            (N_SPEC x N_OUTPUTS) matrix of observed fractional intensities from photonic lantern
        observed_total : np.ndarray
            (N_SPEC,) total observed intensity summed over all outputs (before normalizing to fractions)
        guess : np.ndarray
            Guess for the line properties of Star 1 and Star 2 (depth, center, width) (2 x 3)
        M1 : float
            Mass of Star 1 (to compute the mass ratio)
        wavelength_range : np.ndarray
            Wavelength range [nm]
        bounded_edge : float, optional
            Edge of the bounded region [mas], default None
        refine : bool, optional
            Whether to refine the fit after the binary fitting, default True
        true_values : list[float], optional
            True values of the parameters (x1, y1, x2, y2, F1, depth of line 1, center of line 1, width of line 1, M2, depth of line 2, center of line 2, width of line 2), default None

        Returns
        -------
        final_results : np.ndarray
            Final results of the binary fitting [x1, y1, x2, y2, F1, depth of line 1, center of line 1, width of line 1, M2, depth of line 2, center of line 2, width of line 2]
        '''
        # Stage 1: Fit a single star to get a starting point for Star 1:
        init_time = t.time()
        stage1 = self.single_fitting(observed_fractional_intensities, bounded_edge)
        print(f'Stage 1:    Position = {np.round(stage1.x, 3)} mas')
        print(f'            SSR = {stage1.fun:.3e}')
        print(f'            Time taken: {t.time() - init_time:.3f} seconds\n')

        # If the single star fit is within tolerance, return the result and do not continue with the binary fitting
        if stage1.fun < single_star_tolerance:
            print(f'Single star fit result is within tolerance of {single_star_tolerance}, returning result...\n')
            return stage1
        else:
            print(f'Single star fit result is not within tolerance of {single_star_tolerance}, continuing with binary fitting...\n')

        # Stage 2: Use fitted single star position and grid of Star 2 positions to find local minima and solve for spectra using linear least squares
        init_time = t.time()
        stage2 = self._binary_fitting_stage2(stage1, observed_fractional_intensities, bounded_edge)
        print(f'Stage 2:    Star 1 position = {np.round(stage2.x[0:2], 3)} mas')
        print(f'            Star 2 position = {np.round(stage2.x[2:4], 3)} mas')
        print(f'            SSR = {stage2.cost:.3e}')
        print(f'            Time taken: {t.time() - init_time:.3f} seconds\n')

        # Stage 3: Recover spectra and fit the line and mass of Star 2
        init_time = t.time()
        stage3 = self._binary_fitting_stage3(stage2, observed_fractional_intensities, observed_total, wavelength_range, guess)
        line1_result, line2_result, best_positions = stage3

        # Fitting the mass ratio of Star 1 and Star 2 using mass-luminosity relation
        if M1 < 2:
            fitted_M2 = M1 * (line2_result.x[0] / line1_result.x[0])**(1/5) # (M_2)
        else:
            fitted_M2 = M1 * (line2_result.x[0] / line1_result.x[0])**(1/3) # (M_2)

        print(f'Stage 3:    Star 1 line (continuum flux, depth, center, width) = {np.round(line1_result.x, 4)}')
        print(f'            Star 2 line (continuum flux, depth, center, width) = {np.round(line2_result.x, 4)}')
        print(f'            M_2 = {fitted_M2:.4f}')
        print(f'            Time taken: {t.time() - init_time:.3f} seconds\n')

        final_results = np.array([
            best_positions[0], best_positions[1], best_positions[2], best_positions[3], line1_result.x[0], line1_result.x[1], line1_result.x[2], line1_result.x[3], fitted_M2, line2_result.x[1], line2_result.x[2], line2_result.x[3]
        ])

        if refine:
            init_time = t.time()
            stage4 = self._binary_fitting_stage4(stage3, observed_fractional_intensities, observed_total, M1, wavelength_range, bounded_edge)
            print(f'Stage 4:    Star 1 position = {np.round(stage4.x[0:2], 3)} mas')
            print(f'            Star 2 position = {np.round(stage4.x[2:4], 3)} mas')
            print(f'            Star 1 line (continuum flux, depth, center, width) = {np.round(stage4.x[4:8], 4)}')
            print(f'            Star 2 line (depth, center, width) = {np.round(stage4.x[9:12], 4)}')
            print(f'            M_2 = {stage4.x[8]:.4f}')
            print(f'            SSR = {stage4.cost:.3e}')
            print(f'            Time taken: {t.time() - init_time:.3f} seconds\n')

            # Computing the correlation matrix and other statistics if true values are provided
            if true_values is not None:
                # Finding number of degrees of freedom
                n_data, n_params = stage4.fun.size, stage4.x.size

                # Finding the variance of the residuals
                residual_variance = 2 * stage4.cost / (n_data - n_params)

                # Finding the covariance matrix from the Jacobian: C = (J^T J)^-1 * (residual variance)
                covariance = np.linalg.inv(stage4.jac.T @ stage4.jac) * residual_variance

                # Finding the uncertainties and correlation matrix
                uncertainties = np.sqrt(np.diag(covariance))
                correlation = covariance / np.outer(uncertainties, uncertainties)

                # Creating a DataFrame of the parameters, uncertainties, and correlation matrix
                param_names = [
                    'x1 [mas]', 'y1 [mas]', 'x2 [mas]', 'y2 [mas]',
                    'continuum flux of star 1', 'depth of line 1', 'center of line 1 [nm]', 'width of line 1 [nm]',
                    'M2 [M_1]', 'depth of line 2', 'center of line 2 [nm]', 'width of line 2 [nm]'
                ]
                df = pd.DataFrame({'fit': stage4.x, 'uncertainty': uncertainties, 'true': true_values}, index=param_names)

                print(df.to_string(float_format='%.6g'))
                
                print('\nCorrelation matrix:')
                print(pd.DataFrame(correlation, index=param_names, columns=param_names).to_string(float_format='%.2f'))

            final_results = stage4.x

        return final_results