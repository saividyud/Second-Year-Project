# Importing the necessary modules
import numpy as np
import scipy.optimize as optimize
import pandas as pd

from tqdm import tqdm
import time as t

# Custom libraries
import functions
from PhotonicLanternDevice import PhotonicLanternDevice

class PhotonicLanternFitting:
    '''
    Class to fit single and binary star signals to observed photonic lantern outputs.

    Parameters
    ----------
    lantern : PhotonicLanternDevice
        Photonic lantern device used to model the outputs of trial scenes
    wavelength_range : np.ndarray
        Wavelength range [nm] (must have N_SPEC entries to match the lantern)
    bounded_edge : float, optional
        Edge of the bounded region for the star positions [mas], default None (uses the lantern's angular extent)
    n_grid : int, optional
        Number of grid points along each axis for the Star 2 starting positions in stage 2, default 10
    position_scale : float, optional
        Characteristic scale of the star positions for the least squares fits [mas], default 10
    line_scale : tuple[float, float, float], optional
        Characteristic scales of the line properties (depth, center [nm], width [nm]) for the refinement, default (0.5, 0.05, 0.05)
    mass_scale : float, optional
        Characteristic scale of the mass of Star 2 for the refinement [M_1], default 0.5
    '''
    def __init__(self, lantern : PhotonicLanternDevice, wavelength_range : np.ndarray, bounded_edge : float | None = None, n_grid : int = 10,
                 position_scale : float = 10, line_scale : tuple[float, float, float] = (0.5, 0.05, 0.05), mass_scale : float = 0.5):

        if len(wavelength_range) != lantern.N_SPEC:
            raise ValueError(f'wavelength_range has {len(wavelength_range)} entries but the lantern has N_SPEC = {lantern.N_SPEC}')

        self.lantern = lantern
        self.wavelength_range = wavelength_range
        self.bounded_edge = lantern.angular_extent if bounded_edge is None else bounded_edge
        self.n_grid = n_grid
        self.position_scale = position_scale
        self.line_scale = line_scale
        self.mass_scale = mass_scale

    def single_fitting(self, observed_fractional_intensities : np.ndarray) -> optimize.OptimizeResult:
        '''
        Minimizes the objective function `self._single_star_objective` for a single star fit to observed photonic lantern output spectra.

        Parameters
        ----------
        observed_fractional_intensities : np.ndarray
            (N_SPEC x N_PORTS) matrix of observed fractional intensities from photonic lantern

        Returns
        -------
        result : optimize.OptimizeResult
            Result of the minimization
        '''
        # Defining the bounds of the minimization
        bounds = [(-self.bounded_edge, self.bounded_edge), (-self.bounded_edge, self.bounded_edge)]

        # Executing the minimization
        result = optimize.differential_evolution(
            lambda x: self._single_star_objective(x, observed_fractional_intensities),
            bounds)

        return result

    def binary_fitting(self, observed_fractional_intensities : np.ndarray, observed_total : np.ndarray, guess : np.ndarray, M1 : float, single_star_tolerance : float = 1e-3, refine : bool = True, true_values : list[float] = None) -> np.ndarray:
        '''
        Fits a binary signal to the observed photonic lantern outputs.

        The observation (`observed_fractional_intensities`, `observed_total`, `M1`) and the result of each stage (`stage1` to `stage4`) are stored as attributes.

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
        single_star_tolerance : float, optional
            SSR below which the single star fit is accepted and the binary fitting is skipped, default 1e-3
        refine : bool, optional
            Whether to refine the fit after the binary fitting, default True
        true_values : list[float], optional
            True values of the parameters (x1, y1, x2, y2, F1, depth of line 1, center of line 1, width of line 1, M2, depth of line 2, center of line 2, width of line 2), default None

        Returns
        -------
        final_results : np.ndarray
            Final results of the binary fitting [x1, y1, x2, y2, F1, depth of line 1, center of line 1, width of line 1, M2, depth of line 2, center of line 2, width of line 2]
        '''
        # Storing the observation being fitted
        self.observed_fractional_intensities = observed_fractional_intensities
        self.observed_total = observed_total
        self.M1 = M1

        # Stage 1: Fit a single star to get a starting point for Star 1:
        init_time = t.time()
        self.stage1 = self.single_fitting(self.observed_fractional_intensities)
        print(f'Stage 1:    Position = {np.round(self.stage1.x, 3)} mas')
        print(f'            SSR = {self.stage1.fun:.3e}')
        print(f'            Time taken: {t.time() - init_time:.3f} seconds\n')

        # If the single star fit is within tolerance, return the result and do not continue with the binary fitting
        if self.stage1.fun < single_star_tolerance:
            print(f'Single star fit result is within tolerance of {single_star_tolerance}, returning result...\n')
            return self.stage1
        else:
            print(f'Single star fit result is not within tolerance of {single_star_tolerance}, continuing with binary fitting...\n')

        # Stage 2: Use fitted single star position and grid of Star 2 positions to find local minima and solve for spectra using linear least squares
        init_time = t.time()
        self.stage2 = self._binary_fitting_stage2(self.stage1)
        print(f'Stage 2:    Star 1 position = {np.round(self.stage2.x[0:2], 3)} mas')
        print(f'            Star 2 position = {np.round(self.stage2.x[2:4], 3)} mas')
        print(f'            SSR = {self.stage2.cost:.3e}')
        print(f'            Time taken: {t.time() - init_time:.3f} seconds\n')

        # Stage 3: Recover spectra and fit the line and mass of Star 2
        init_time = t.time()
        self.stage3 = self._binary_fitting_stage3(self.stage2, guess)
        line1_result, line2_result, best_positions = self.stage3

        # Fitting the mass ratio of Star 1 and Star 2 using mass-luminosity relation
        if self.M1 < 2:
            fitted_M2 = self.M1 * (line2_result.x[0] / line1_result.x[0])**(1/5) # (M_2)
        else:
            fitted_M2 = self.M1 * (line2_result.x[0] / line1_result.x[0])**(1/3) # (M_2)

        print(f'Stage 3:    Star 1 line (continuum flux, depth, center, width) = {np.round(line1_result.x, 4)}')
        print(f'            Star 2 line (continuum flux, depth, center, width) = {np.round(line2_result.x, 4)}')
        print(f'            M_2 = {fitted_M2:.4f}')
        print(f'            Time taken: {t.time() - init_time:.3f} seconds\n')

        final_results = np.array([
            best_positions[0], best_positions[1], best_positions[2], best_positions[3], line1_result.x[0], line1_result.x[1], line1_result.x[2], line1_result.x[3], fitted_M2, line2_result.x[1], line2_result.x[2], line2_result.x[3]
        ])

        if refine:
            init_time = t.time()
            self.stage4 = self._binary_fitting_stage4(self.stage3)
            print(f'Stage 4:    Star 1 position = {np.round(self.stage4.x[0:2], 3)} mas')
            print(f'            Star 2 position = {np.round(self.stage4.x[2:4], 3)} mas')
            print(f'            Star 1 line (continuum flux, depth, center, width) = {np.round(self.stage4.x[4:8], 4)}')
            print(f'            Star 2 line (depth, center, width) = {np.round(self.stage4.x[9:12], 4)}')
            print(f'            M_2 = {self.stage4.x[8]:.4f}')
            print(f'            SSR = {self.stage4.cost:.3e}')
            print(f'            Time taken: {t.time() - init_time:.3f} seconds\n')

            # Computing the correlation matrix and other statistics if true values are provided
            if true_values is not None:
                # Finding number of degrees of freedom
                n_data, n_params = self.stage4.fun.size, self.stage4.x.size

                # Finding the variance of the residuals
                residual_variance = 2 * self.stage4.cost / (n_data - n_params)

                # Finding the covariance matrix from the Jacobian: C = (J^T J)^-1 * (residual variance)
                covariance = np.linalg.inv(self.stage4.jac.T @ self.stage4.jac) * residual_variance

                # Finding the uncertainties and correlation matrix
                uncertainties = np.sqrt(np.diag(covariance))
                correlation = covariance / np.outer(uncertainties, uncertainties)

                # Creating a DataFrame of the parameters, uncertainties, and correlation matrix
                param_names = [
                    'x1 [mas]', 'y1 [mas]', 'x2 [mas]', 'y2 [mas]',
                    'continuum flux of star 1', 'depth of line 1', 'center of line 1 [nm]', 'width of line 1 [nm]',
                    'M2 [M_1]', 'depth of line 2', 'center of line 2 [nm]', 'width of line 2 [nm]'
                ]
                df = pd.DataFrame({'fit': self.stage4.x, 'uncertainty': uncertainties, 'true': true_values}, index=param_names)

                print(df.to_string(float_format='%.6g'))
                
                print('\nCorrelation matrix:')
                print(pd.DataFrame(correlation, index=param_names, columns=param_names).to_string(float_format='%.2f'))

            final_results = self.stage4.x

        return final_results

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
        star_flat_spectrum = np.ones(self.lantern.N_SPEC) # [arbitrary units]

        # Calculating the outputs of the star
        star_outputs = self.lantern.calculate_photonic_lantern_outputs(np.array([star_position]), np.array([star_flat_spectrum]))

        # Computing the total intensities across all outputs as a function of wavelength
        single_total_intensities = np.sum(star_outputs, axis=1)

        # Computing fractional intensities for the outputs
        single_fractional_intensities = star_outputs / single_total_intensities.reshape(-1, 1)

        # Calculating residuals
        residuals = observed_fractional_intensities - single_fractional_intensities # (N_SPEC, N_PORTS)
        delta = np.sum(residuals**2)

        return delta

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
        flat_spectrum = np.ones(self.lantern.N_SPEC) # [arbitrary units]

        # Calculating the outputs of the star positions
        star1_flat_output = self.lantern.calculate_photonic_lantern_outputs(np.array([star1_position]), np.array([flat_spectrum]))
        star2_flat_output = self.lantern.calculate_photonic_lantern_outputs(np.array([star2_position]), np.array([flat_spectrum]))

        # Solving the spectra using linear least squares
        spectra, model_intensities = self._solve_spectra(star1_flat_output, star2_flat_output, observed_fractional_intensities)

        # Calculating the residuals
        residuals = observed_fractional_intensities - model_intensities
        
        return residuals.ravel()

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

    def _binary_fitting_stage2(self, stage1 : optimize.OptimizeResult) -> optimize.OptimizeResult:
        '''
        Uses the fitted single star position and a grid of Star 2 positions to find local minima and solve for spectra using linear least squares.

        Parameters
        ----------
        stage1 : optimize.OptimizeResult
            Result of the single star fit

        Returns
        -------
        stage2 : optimize.OptimizeResult
        '''
        # Defining a grid of Star 2 positions
        x2_grid = np.linspace(-self.bounded_edge, self.bounded_edge, self.n_grid)
        y2_grid = np.linspace(-self.bounded_edge, self.bounded_edge, self.n_grid)

        # Iterating through each Star 2 position and finding the best fit
        best_fit = None

        with tqdm(total=len(x2_grid) * len(y2_grid), desc='Binary fitting stage 2 (position fitting)') as pbar:
            for x2 in x2_grid:
                for y2 in y2_grid:
                    position = np.array([stage1.x[0], stage1.x[1], x2, y2])

                    fit = optimize.least_squares(
                        lambda x: self._position_residuals(x, self.observed_fractional_intensities),
                        x0=position,
                        bounds=([-self.bounded_edge] * 4, [self.bounded_edge] * 4),
                        x_scale=self.position_scale
                    )

                    if best_fit is None or fit.cost < best_fit.cost:
                        best_fit = fit

                    pbar.update(1)

        return best_fit

    def _binary_fitting_stage3(self, stage2 : optimize.OptimizeResult, guess : np.ndarray) -> tuple[optimize.OptimizeResult, optimize.OptimizeResult, np.ndarray]:
        '''
        Calculates the spectra and fits the line properties of both stars, as well as calculates the mass ratio.

        Parameters
        ----------
        stage2 : optimize.OptimizeResult
            Result of the binary fitting stage 2
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

        flat_spectrum = np.ones(self.lantern.N_SPEC) # [arbitrary units]

        star1_flat_output = self.lantern.calculate_photonic_lantern_outputs(np.array([[best_positions[0], best_positions[1]]]), np.array([flat_spectrum]))
        star2_flat_output = self.lantern.calculate_photonic_lantern_outputs(np.array([[best_positions[2], best_positions[3]]]), np.array([flat_spectrum]))

        spectra, model_intensities = self._solve_spectra(star1_flat_output, star2_flat_output, self.observed_fractional_intensities)

        # Finding the brighter of the two stars to call it "Star 1"
        if np.median(spectra[:, 1]) > np.median(spectra[:, 0]):
            spectra = spectra[:, ::-1]
            best_positions = np.r_[best_positions[2:], best_positions[:2]]

        # Calculating the recovered spectra
        self.recovered_s1 = spectra[:, 0] * self.observed_total # (N_SPEC,)
        self.recovered_s2 = spectra[:, 1] * self.observed_total # (N_SPEC,)

        # Constructing guesses
        guess1 = (np.median(self.recovered_s1), guess[0, 0], guess[0, 1], guess[0, 2])
        guess2 = (np.median(self.recovered_s2), guess[1, 0], guess[1, 1], guess[1, 2])

        # Fitting the line properties of Star 1 and Star 2
        line1_result = self._fit_line(self.recovered_s1, guess1)
        line2_result = self._fit_line(self.recovered_s2, guess2)

        return line1_result, line2_result, best_positions

    def _fit_line(self, spectrum : np.ndarray, guess : tuple[float, float, float, float]) -> optimize.OptimizeResult:
        '''
        Fits the line properties of a star to a recovered spectrum.

        Parameters
        ----------
        spectrum : np.ndarray
            Recovered spectrum of the star [arbitrary units]
        guess : tuple[float, float, float, float]
            Guess for the line properties of the star (continuum flux, depth, center, width)

        Returns
        -------
        result : optimize.OptimizeResult
            Result of the minimization
        '''
        return optimize.least_squares(
            fun=lambda q: functions.create_spectrum(q[0], q[1], q[2], q[3], self.wavelength_range) - spectrum, # (continuum flux, depth, center, width)
            x0=guess,
            bounds=([0, 0, self.wavelength_range[0], 1e-6], [np.inf, 2, self.wavelength_range[-1], 1]) # (continuum flux, depth, center, width)
        )

    def _binary_fitting_stage4(self, stage3 : tuple[optimize.OptimizeResult, optimize.OptimizeResult, np.ndarray]) -> optimize.OptimizeResult:
        '''
        Refines all the parameters of the binary (positions and both line profiles) using nonlinear least squares.

        Parameters
        ----------
        stage3 : tuple[optimize.OptimizeResult, optimize.OptimizeResult, np.ndarray]
            Results of the binary fitting stage 3 (line fit of Star 1, line fit of Star 2, ordered positions)

        Returns
        -------
        stage4 : optimize.OptimizeResult
            Result of the minimization with parameters (x1, y1, x2, y2, F1, a1, c1, w1, M2, a2, c2, w2)
        '''
        # Extracting the line fits and positions of the stars
        line1_result, line2_result, best_positions = stage3

        # Guess for M2
        if self.M1 < 2:
            fitted_M2 = self.M1 * (line2_result.x[0] / line1_result.x[0])**(1/5) # (M_2)
        else:
            fitted_M2 = self.M1 * (line2_result.x[0] / line1_result.x[0])**(1/3) # (M_2)

        F1_guess = line1_result.x[0]

        edge = self.bounded_edge
        wav_min, wav_max = self.wavelength_range[0], self.wavelength_range[-1]

        # Performing the least squares fit
        stage4 = optimize.least_squares(
            fun=lambda params: self._binary_objective(params, self.observed_fractional_intensities, self.observed_total, self.M1),
            x0=(
                # [x1, y1, x2, y2, F1, depth of line 1, center of line 1, width of line 1, M2, depth of line 2, center of line 2, width of line 2]
                *best_positions, F1_guess, *line1_result.x[1:], fitted_M2, *line2_result.x[1:]
            ),
            bounds=(
                [-edge, -edge, -edge, -edge, 0, 0, wav_min, 1e-6, 0, 0, wav_min, 1e-6],
                [edge, edge, edge, edge, np.inf, 2, wav_max, 1, np.inf, 2, wav_max, 1]
            ),
            x_scale=[*[self.position_scale] * 4, 0.1 * F1_guess, *self.line_scale, self.mass_scale, *self.line_scale],
        )

        return stage4

    def _binary_objective(self, params : np.ndarray, observed_fractional_intensities : np.ndarray, observed_total : np.ndarray, M1 : float) -> np.ndarray:
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

        Returns
        -------
        residuals : np.ndarray
            Flattened residual vector of the differences between the observed and model intensities, normalized by the observed total intensity
        '''
        # Extracting the parameters
        x1, y1, x2, y2, F_1, depth1, center1, width1, M2, depth2, center2, width2 = params

        # Creating the spectra of the stars
        F_2 = F_1 / functions.M_L_relation(M1, M2)

        star1_spectrum = functions.create_spectrum(F_1, depth1, center1, width1, self.wavelength_range)
        star2_spectrum = functions.create_spectrum(F_2, depth2, center2, width2, self.wavelength_range)

        # Calculating the outputs of the star positions
        model_intensities = self.lantern.calculate_photonic_lantern_outputs(np.array([[x1, y1], [x2, y2]]), np.array([star1_spectrum, star2_spectrum]))

        # Comparing absolute intensities (not fractions) so that the overall flux scale, and therefore Star 1's spectrum, is constrained.
        # Dividing by the observed total keeps the residuals on the same scale as the fractional intensities.
        residuals = model_intensities / observed_total[:, None] - observed_fractional_intensities

        return residuals.ravel()
