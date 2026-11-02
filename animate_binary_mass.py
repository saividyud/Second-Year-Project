"""Animate photonic-lantern output maps while the secondary-star mass changes.

This script extends the two notebook sweeps into three dimensions: the
secondary star's x position, y position, and mass. Position-dependent device
responses are computed once and then reused for every mass frame.

Example:
    .venv/bin/python animate_binary_mass.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hcipy as hc
import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PLsim.otf import OTF
from PLsim.scene import Scene, SceneProjector
from PLsim.system.device import Device
from PLsim.system.lantern import generate_unitary_matrix
from tqdm import tqdm


N_OUTPUTS = 10
N_SPEC = 100
DIAMETER_M = 10.0
WAVELENGTH_NM = 1282.8
LINE_WIDTH_NM = 0.1
LINE_DEPTH = 0.9
MAS_TO_RAD = 1e-3 / 206265


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate 5x2 photonic-lantern maps as secondary mass changes."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("binary_mass_animation.mp4"),
        help="Output .mp4 or .gif path (default: binary_mass_animation.mp4).",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=50,
        help="Samples along each position axis (default: 50).",
    )
    parser.add_argument(
        "--mass-frames",
        type=int,
        default=30,
        help="Number of secondary-mass frames (default: 30).",
    )
    parser.add_argument("--mass-min", type=float, default=0.01)
    parser.add_argument("--mass-max", type=float, default=1.0)
    parser.add_argument(
        "--position-limit",
        type=float,
        default=100.0,
        help="Plot positions from -limit to +limit mas (default: 100).",
    )
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the random unitary lantern matrix (default: 0).",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("binary_position_responses.npz"),
        help="Position-response cache (default: binary_position_responses.npz).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not load or save a position-response cache.",
    )
    return parser.parse_args()


def mass_luminosity_ratio(primary_mass: float, secondary_mass: np.ndarray) -> np.ndarray:
    """Return F1/F2 using the same relation as binaries.ipynb."""
    exponent = 5 if primary_mass < 2 else 3
    return (primary_mass / secondary_mass) ** exponent


def create_optical_model(seed: int) -> tuple[SceneProjector, Device]:
    """Recreate the photonic-lantern model used in binaries.ipynb."""
    np.random.seed(seed)

    ndim = 65
    pupil_grid = hc.make_pupil_grid(ndim, DIAMETER_M)
    focal_plane_resolution = 1e-6
    focal_grid = hc.make_pupil_grid(ndim, focal_plane_resolution * ndim)
    aperture = hc.make_circular_aperture(DIAMETER_M)(pupil_grid)

    otf = OTF.from_fiber_params(
        1.444,
        1.444 - 5.5e-3,
        10e-6,
        WAVELENGTH_NM * 1e-9,
        focal_grid,
        pupil_grid,
        40,
        aperture,
        optimize_focal_length=False,
    )
    otf.compute()

    matrix = generate_unitary_matrix(N_OUTPUTS)[np.newaxis, :, :].repeat(
        N_SPEC, axis=0
    )
    device = Device(matrix, verbose=False)
    scene = Scene(pupil_grid, WAVELENGTH_NM * 1e-9)
    return SceneProjector(otf, scene), device


def point_response(
    scene_projector: SceneProjector, device: Device, x_rad: float, y_rad: float
) -> np.ndarray:
    """Calculate the wavelength-independent device response at one position."""
    projection = np.asarray(scene_projector.compute_point(x_rad, y_rad)).squeeze()
    # The same unitary matrix is used at every wavelength, so one result is
    # sufficient. Device broadcasts this 2-D coherence matrix over wavelength.
    return device.calculate_outputs(projection)[0]


def load_or_compute_responses(
    theta_mas: np.ndarray,
    scene_projector: SceneProjector,
    device: Device,
    cache_path: Path | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return responses for the on-axis primary and all secondary positions."""
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        if (
            np.array_equal(cached["theta_mas"], theta_mas)
            and int(cached["seed"]) == seed
        ):
            print(f"Loading position responses from {cache_path}")
            return cached["primary_response"], cached["secondary_responses"]
        print(f"Ignoring incompatible cache at {cache_path}")

    primary_response = point_response(scene_projector, device, 0.0, 0.0)
    theta_rad = theta_mas * MAS_TO_RAD
    secondary_responses = np.empty(
        (theta_mas.size, theta_mas.size, N_OUTPUTS), dtype=float
    )

    positions = (
        (iy, ix, y_rad, x_rad)
        for iy, y_rad in enumerate(theta_rad)
        for ix, x_rad in enumerate(theta_rad)
    )
    for iy, ix, y_rad, x_rad in tqdm(
        positions,
        total=theta_mas.size**2,
        desc="Computing x-y responses",
    ):
        secondary_responses[iy, ix] = point_response(
            scene_projector, device, x_rad, y_rad
        )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            theta_mas=theta_mas,
            seed=seed,
            primary_response=primary_response,
            secondary_responses=secondary_responses,
        )
        print(f"Saved position responses to {cache_path}")

    return primary_response, secondary_responses


def normalization_scale() -> float:
    """Spectral factor from the notebook's integrated normalization scheme."""
    band_width = 10 * LINE_WIDTH_NM
    wavelengths = np.linspace(
        WAVELENGTH_NM - band_width / 2,
        WAVELENGTH_NM + band_width / 2,
        N_SPEC,
    )
    profile = np.ones_like(wavelengths)
    profile -= (1 - LINE_DEPTH) * np.exp(
        -0.5 * (wavelengths - WAVELENGTH_NM) ** 2 / LINE_WIDTH_NM**2
    )
    continuum = profile[:2].mean()
    delta_wavelength = wavelengths[1] - wavelengths[0]
    return abs(np.sum((profile - continuum) / profile) * delta_wavelength)


def calculate_log_maps(
    primary_response: np.ndarray,
    secondary_responses: np.ndarray,
    secondary_masses: np.ndarray,
) -> np.ndarray:
    """Calculate log-normalized maps for every mass and output."""
    primary_mass = 1.0
    secondary_fluxes = 1.0 / mass_luminosity_ratio(
        primary_mass, secondary_masses
    )
    scale = normalization_scale()
    maps = np.empty(
        (secondary_masses.size, *secondary_responses.shape), dtype=float
    )

    for frame, flux in enumerate(secondary_fluxes):
        combined_response = primary_response + flux * secondary_responses
        divisor = combined_response[..., 0, np.newaxis]
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = scale * np.abs(combined_response / divisor)
            maps[frame] = np.where(normalized > 0, np.log10(normalized), np.nan)

    return maps


def create_animation(
    log_maps: np.ndarray,
    theta_mas: np.ndarray,
    secondary_masses: np.ndarray,
    output_path: Path,
    fps: int,
    dpi: int,
) -> None:
    """Create and save the shared-color-scale 5x2 animation."""
    finite = log_maps[np.isfinite(log_maps)]
    if finite.size == 0:
        raise ValueError("No finite intensities were produced.")
    vmin, vmax = finite.min(), finite.max()

    fig, axes_2d = plt.subplots(
        2, 5, figsize=(18, 8), constrained_layout=True, sharex=True, sharey=True
    )
    axes = axes_2d.ravel()
    extent = [
        theta_mas[0],
        theta_mas[-1],
        theta_mas[0],
        theta_mas[-1],
    ]
    diffraction_limit = (
        1.22 * WAVELENGTH_NM * 1e-9 / DIAMETER_M * 206265 * 1000
    )

    images = []
    for output_index, ax in enumerate(axes):
        image = ax.imshow(
            log_maps[0, :, :, output_index],
            extent=extent,
            aspect="equal",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        images.append(image)
        ax.set_title(f"Output {output_index + 1}")
        ax.add_patch(
            patches.Circle(
                (0, 0),
                diffraction_limit,
                color="k",
                fill=False,
                linestyle="--",
                linewidth=1,
            )
        )
        if output_index >= 5:
            ax.set_xlabel("X [mas]")
        if output_index in (0, 5):
            ax.set_ylabel("Y [mas]")

    title = fig.suptitle(
        rf"Secondary mass: $M_2={secondary_masses[0]:.3f}\,M_\odot$"
    )
    fig.colorbar(
        images[0],
        ax=axes,
        location="right",
        label=r"$\log_{10}$(Normalized Intensity)",
        pad=0.01,
        shrink=0.9,
    )

    def update(frame: int):
        for output_index, image in enumerate(images):
            image.set_data(log_maps[frame, :, :, output_index])
        title.set_text(
            rf"Secondary mass: $M_2={secondary_masses[frame]:.3f}\,M_\odot$"
        )
        return [*images, title]

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=secondary_masses.size,
        interval=1000 / fps,
        blit=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".gif":
        writer = animation.PillowWriter(fps=fps)
    else:
        writer = animation.FFMpegWriter(fps=fps, codec="h264", bitrate=2400)
    movie.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Saved animation to {output_path}")


def main() -> None:
    args = parse_args()
    if args.grid_size < 2:
        raise ValueError("--grid-size must be at least 2.")
    if args.mass_frames < 2:
        raise ValueError("--mass-frames must be at least 2.")
    if not 0 < args.mass_min <= args.mass_max:
        raise ValueError("Mass limits must satisfy 0 < mass-min <= mass-max.")

    cache_path = None if args.no_cache else args.cache
    theta_mas = np.linspace(
        -args.position_limit, args.position_limit, args.grid_size
    )
    secondary_masses = np.linspace(
        args.mass_min, args.mass_max, args.mass_frames
    )

    scene_projector, device = create_optical_model(args.seed)
    primary_response, secondary_responses = load_or_compute_responses(
        theta_mas, scene_projector, device, cache_path, args.seed
    )
    log_maps = calculate_log_maps(
        primary_response, secondary_responses, secondary_masses
    )
    create_animation(
        log_maps,
        theta_mas,
        secondary_masses,
        args.output,
        args.fps,
        args.dpi,
    )


if __name__ == "__main__":
    main()
