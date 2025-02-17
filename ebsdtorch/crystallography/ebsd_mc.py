import numpy as np
from typing import Dict, Any
import os
import torch
import numpy as np
from typing import Dict, Optional, Tuple
from ebsdtorch.crystallography.unit_cell import Cell

# Set the PYOPENCL_CTX environment variable
os.environ["PYOPENCL_CTX"] = "0"

try:
    import pyopencl as cl
    import pyopencl.array
except ImportError:
    raise ImportError("pyopencl required for Monte Carlo simulations")

EBSD_MC_KERNEL = """
#define PI 3.14159265359f

typedef struct {
    float x;
    float y;
} LambertStruct;

LambertStruct rosca_lambert(float4 pt) {
    LambertStruct ret;
    float factor = sqrt(fmax(2.0f * (1.0f - fabs(pt.z)), 0.0f));
    float big = fabs(pt.y) <= fabs(pt.x) ? pt.x : pt.y;
    float sml = fabs(pt.y) <= fabs(pt.x) ? pt.y : pt.x;
    float simpler_term = (big < 0 ? -1.0f : 1.0f) * factor * (2.0f / sqrt(8.0f));
    float arctan_term = (big < 0 ? -1.0f : 1.0f) * factor * atan2(sml * (big < 0 ? -1.0f : 1.0f), fabs(big)) * (2.0f * sqrt(2.0f) / PI);
    ret.x = fabs(pt.y) <= fabs(pt.x) ? simpler_term : arctan_term;
    ret.y = fabs(pt.y) <= fabs(pt.x) ? arctan_term : simpler_term;
    return ret;
}

uint4 lfsr113_Bits(uint4 z) {
    uint b;
    b  = ((z.x << 6) ^ z.x) >> 13;
    z.x = ((z.x & 4294967294U) << 18) ^ b;
    b  = ((z.y << 2) ^ z.y) >> 27;
    z.y = ((z.y & 4294967288U) << 2) ^ b;
    b  = ((z.z << 13) ^ z.z) >> 21;
    z.z = ((z.z & 4294967280U) << 7) ^ b;
    b  = ((z.w << 3) ^ z.w) >> 12;
    z.w = ((z.w & 4294967168U) << 13) ^ b;
    return z;
}

__kernel void bse_sim_mc(
    __global uint* accumulator,
    __global uint4* seeds,
    const float n_trials_per_electron,
    const float starting_E_keV,
    const int n_exit_energy_bins,
    const int n_exit_direction_bins,
    const int n_exit_depth_bins,
    const float binsize_exit_energy,
    const float binsize_exit_depth,
    const float atom_num,
    const float unit_cell_density_rho,
    const float atomic_weight_A,
    const int n_max_steps,
    const float sigma,
    const float omega,
    const int depth_mode  // 0 for linear, 1 for logscale
) {
    int gid = get_global_id(0);

    float sigma_rad = sigma * PI / 180.0f;
    float omega_rad = omega * PI / 180.0f;
    float mean_ionization_pot_J = ((9.76f * atom_num) + (58.5f * pow(atom_num, -0.19f))) * 1.0e-3f;

    // Precompute constants
    const float const_0_00785 = -0.00785f * (atom_num / atomic_weight_A);
    const float const_5_21 = 5.21f * 602.2f * pow(atom_num, 2.0f);
    const float const_3_4e_3 = 3.4e-3f * pow(atom_num, 0.66667f);
    const float const_1e7_over_rho = 1.0e7f * atomic_weight_A / unit_cell_density_rho;

    // Updated random number generation factor
    const float rand_factor = 2.32830643708079737543146996187e-10f;

    uint4 seed = seeds[gid];

    for (int i = 0; i < n_trials_per_electron; i++) {
        float4 current_direction = (float4)(
            sin(sigma_rad) * cos(omega_rad),
            sin(sigma_rad) * sin(omega_rad),
            cos(sigma_rad),
            0.0f  // Initialize depth to 0
        );
        current_direction.xyz /= sqrt(dot(current_direction.xyz, current_direction.xyz));

        float energy = starting_E_keV;

        for (int step = 0; step < n_max_steps; step++) {
            float energy_inv = 1.0f / energy;
            float alpha = const_3_4e_3 * energy_inv;
            float sigma_E = const_5_21 * energy_inv * energy_inv * 
                            (4.0f * PI / (alpha * (1.0f + alpha))) * 
                            (((511.0f + energy) / (1024.0f + energy)) * ((511.0f + energy) / (1024.0f + energy)));
            float mean_free_path_nm = const_1e7_over_rho / sigma_E;

            seed = lfsr113_Bits(seed);
            float rand_step = (float)(seed.x ^ seed.y ^ seed.z ^ seed.w) * rand_factor;
            float step_nm = -mean_free_path_nm * log(rand_step);

            float de_ds = const_0_00785 * energy_inv * log((1.166f * energy / mean_ionization_pot_J) + 0.9911f);

            seed = lfsr113_Bits(seed);
            float rand_phi = (float)(seed.x ^ seed.y ^ seed.z ^ seed.w) * rand_factor;
            float phi = acos(1.0f - ((2.0f * alpha * rand_phi) / (1.0f + alpha - rand_phi)));

            seed = lfsr113_Bits(seed);
            float rand_psi = (float)(seed.x ^ seed.y ^ seed.z ^ seed.w) * rand_factor;
            float psi = 2.0f * PI * rand_psi;

            float4 c_old = current_direction;
            float4 c_new;

            float cos_phi = cos(phi);
            float sin_phi = sin(phi);
            if (fabs(c_old.z) > 0.99999f) {
                float cos_psi = cos(psi);
                float sin_psi = sin(psi);
                c_new = (float4)(
                    sin_phi * cos_psi,
                    sin_phi * sin_psi,
                    (c_old.z > 0 ? 1.0f : -1.0f) * cos_phi,
                    c_old.w  // Preserve depth
                );
            } else {
                float dsq = sqrt(1.0f - c_old.z * c_old.z);
                float dsqi = 1.0f / dsq;
                float cos_psi = cos(psi);
                float sin_psi = sin(psi);
                c_new = (float4)(
                    sin_phi * (c_old.x * c_old.z * cos_psi - c_old.y * sin_psi) * dsqi + c_old.x * cos_phi,
                    sin_phi * (c_old.y * c_old.z * cos_psi + c_old.x * sin_psi) * dsqi + c_old.y * cos_phi,
                    -sin_phi * cos_psi * dsq + c_old.z * cos_phi,
                    c_old.w  // Preserve depth
                );
            }

            c_new.xyz /= sqrt(dot(c_new.xyz, c_new.xyz));
            
            float escape_depth = fabs(c_new.w / c_new.z);
            c_new.w += step_nm * c_new.z;  // Update depth
            energy += step_nm * unit_cell_density_rho * de_ds;

            current_direction = c_new;

            // Ensure energy doesn't go negative
            if (energy <= 0) {
                energy = 0;
                break;
            } else {
                if (current_direction.w < 0) {
                    int exit_depth_index;
                    if (depth_mode == 0) {  // linear
                        exit_depth_index = (int)(escape_depth / binsize_exit_depth);
                    } else {  // logscale
                        exit_depth_index = (int)(log(escape_depth + 1) / binsize_exit_depth);
                    }
                    int exit_energy_index = (int)((starting_E_keV - energy) * (1.0f / binsize_exit_energy));
                    LambertStruct lambert = rosca_lambert(current_direction);
                    int2 exit_direction_index = (int2)(
                        (int)((lambert.x * 0.499999f + 0.5f) * n_exit_direction_bins),
                        (int)((lambert.y * 0.499999f + 0.5f) * n_exit_direction_bins)
                    );

                    if (exit_energy_index >= 0 && exit_energy_index < n_exit_energy_bins &&
                        exit_depth_index >= 0 && exit_depth_index < n_exit_depth_bins &&
                        exit_direction_index.x >= 0 && exit_direction_index.x < n_exit_direction_bins &&
                        exit_direction_index.y >= 0 && exit_direction_index.y < n_exit_direction_bins) {
                        int index = exit_energy_index * (n_exit_depth_bins * n_exit_direction_bins * n_exit_direction_bins) +
                                    exit_depth_index * (n_exit_direction_bins * n_exit_direction_bins) +
                                    exit_direction_index.x * n_exit_direction_bins +
                                    exit_direction_index.y;
                        atomic_inc(&accumulator[index]);
                    }
                    break;
                }
            }
        }
    }

    seeds[gid] = seed;
}
"""


class OpenCLMonteCarlo:
    def __init__(self, params: Dict[str, Any]):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.program = cl.Program(self.ctx, EBSD_MC_KERNEL).build()

        self.n_electrons = params["n_electrons"]

        # Allocate memory on the GPU
        histogram_shape = (
            params["n_exit_energy_bins"],
            params["n_exit_depth_bins"],
            params["n_exit_direction_bins"],
            params["n_exit_direction_bins"],
        )
        self.accumulator = cl.array.zeros(self.queue, histogram_shape, dtype=np.uint32)
        self.seeds = cl.array.to_device(
            self.queue,
            np.random.randint(0, 2**32, size=(self.n_electrons, 4), dtype=np.uint32),
        )

    def run_simulation(self, params: Dict[str, Any]) -> np.ndarray:
        self.program.bse_sim_mc(
            self.queue,
            (self.n_electrons,),
            None,
            self.accumulator.data,
            self.seeds.data,
            np.float32(params["n_trials_per_electron"]),
            np.float32(params["starting_E_keV"]),
            np.int32(params["n_exit_energy_bins"]),
            np.int32(params["n_exit_direction_bins"]),
            np.int32(params["n_exit_depth_bins"]),
            np.float32(params["binsize_exit_energy"]),
            np.float32(params["binsize_exit_depth"]),
            np.float32(params["atom_num"]),
            np.float32(params["unit_cell_density_rho"]),
            np.float32(params["atomic_weight_A"]),
            np.int32(params["n_max_steps"]),
            np.float32(params["sigma"]),
            np.float32(params["omega"]),
            np.int32(0 if params["depth_mode"] == "linear" else 1),
        )

        return self.accumulator.get()


def run_monte_carlo_simulation(
    material_params: Dict[str, Any], simulation_params: Dict[str, Any]
) -> np.ndarray:
    params = {**material_params, **simulation_params}
    mc = OpenCLMonteCarlo(params)
    histogram = mc.run_simulation(params)
    return histogram


class MonteCarloSimulator:
    """A class to run Monte Carlo simulations for electron backscatter."""

    def __init__(
        self,
        cell: Cell,
        kV: float,
        incidence_angle: float,
        binsize_exit_energy: float = 0.5,
        binsize_exit_depth: float = 0.1,
        n_depth_bins: int = 80,
        depth_mode: str = "log",
        n_direction_bins: int = 101,
        n_energy_bins_extra: int = 4,
        find_steps_fraction: float = 1 / 100,
        min_histogram_mass: int = 20_000_000,
        omega: float = 0.0,
        n_max_steps_override: Optional[int] = None,
    ):
        """
        Initialize the Monte Carlo simulator.

        Args:
            cell: Unit cell containing crystal structure information
            kV: Accelerating voltage in keV
            incidence_angle: Angle of incidence in degrees
            binsize_exit_energy: Energy bin size in keV
            binsize_exit_depth: Depth bin size in nm
            n_depth_bins: Number of depth bins
            n_direction_bins: Number of direction bins
            n_energy_bins_extra: Number of extra energy bins
            find_steps_fraction: Fraction of electrons to use when finding required steps
            min_histogram_mass: Minimum number of electrons for histogram
            omega: Azimuthal angle in degrees
        """
        self.cell = cell
        self.kV = kV
        self.incidence_angle = incidence_angle
        self.binsize_exit_energy = binsize_exit_energy
        self.binsize_exit_depth = binsize_exit_depth
        self.n_depth_bins = n_depth_bins
        self.depth_mode = depth_mode
        self.n_direction_bins = n_direction_bins
        self.n_energy_bins_extra = n_energy_bins_extra
        self.find_steps_fraction = find_steps_fraction
        self.min_histogram_mass = min_histogram_mass
        self.omega = omega

        # Compute number of energy bins based on kV
        self.n_fillable_bins = int(kV // binsize_exit_energy)
        self.n_energy_bins = self.n_fillable_bins + n_energy_bins_extra

        # Set up material parameters from cell
        self.material_params = {
            "atom_num": cell.get_average_atomic_number.cpu().item(),
            "unit_cell_density_rho": cell.get_density.cpu().item(),
            "atomic_weight_A": cell.get_average_atomic_weight.cpu().item(),
            "residual": 0.0,  # Not using residuals in this implementation
        }

        # Initialize simulation parameters
        self.base_simulation_params = {
            "n_electrons": 1_000_000,  # Default value, can be changed
            "n_trials_per_electron": 10,
            "starting_E_keV": self.kV
            + self.n_energy_bins_extra * self.binsize_exit_energy,
            "n_exit_energy_bins": self.n_energy_bins,
            "n_exit_depth_bins": self.n_depth_bins,
            "n_exit_direction_bins": self.n_direction_bins,
            "depth_mode": self.depth_mode,
            "sigma": self.incidence_angle,
            "omega": self.omega,
            "binsize_exit_energy": self.binsize_exit_energy,
            "binsize_exit_depth": self.binsize_exit_depth,
        }

        # Find required number of steps if requested
        if n_max_steps_override is None:
            self.n_max_steps = self._find_required_steps()
        elif isinstance(n_max_steps_override, int):
            self.n_max_steps = n_max_steps_override
        else:
            raise ValueError("n_max_steps_override must be an int or None")
        print(f"Required steps: {self.n_max_steps}")

    def _find_required_steps(self, threshold: float = 0.0) -> int:
        """
        Find the required number of steps to fill the energy marginal.

        Args:
            threshold: Minimum acceptable value in energy marginal bins

        Returns:
            Required number of steps
        """
        step_sizes = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]

        # Use a reduced number of electrons for step finding
        test_simulation_params = self.base_simulation_params.copy()
        test_simulation_params["n_electrons"] = int(
            self.base_simulation_params["n_electrons"] * self.find_steps_fraction
        )

        for n_steps in step_sizes:
            test_simulation_params["n_max_steps"] = n_steps

            # Run a test simulation
            histogram = run_monte_carlo_simulation(
                self.material_params, test_simulation_params
            )

            # Convert to tensor and process
            histogram = torch.tensor(histogram).float()
            histogram = histogram[self.n_energy_bins_extra :]

            if self.omega == 0.0:
                histogram = (histogram + histogram.flip(3)) / 2

            # Check energy marginal
            energy_marginal = histogram.sum(dim=(1, 2, 3))
            energy_marginal /= energy_marginal.sum()
            energy_marginal = energy_marginal[: self.n_fillable_bins]

            if torch.all(energy_marginal > threshold):
                return n_steps

        return max(step_sizes)

    def _interp(
        self,
        x: torch.Tensor,
        xp: torch.Tensor,
        fp: torch.Tensor,
        dim: int = -1,
        extrapolate: str = "const",
    ) -> torch.Tensor:
        """One-dimensional linear interpolation between monotonically increasing sample points.

        Args:
            x: The x-coordinates at which to evaluate the interpolated values
            xp: The x-coordinates of the data points, must be increasing
            fp: The y-coordinates of the data points, same shape as xp
            dim: Dimension across which to interpolate
            extrapolate: How to handle values outside range ('linear' or 'const' or 'error')

        Returns:
            The interpolated values, same size as x
        """
        # Move interpolation dimension to last axis
        x = x.movedim(dim, -1)
        xp = xp.movedim(dim, -1)
        fp = fp.movedim(dim, -1)

        # Calculate slope and offset
        m = torch.diff(fp) / torch.diff(xp)
        b = fp[..., :-1] - m * xp[..., :-1]

        # Find indices
        indices = torch.searchsorted(xp.contiguous(), x.contiguous(), right=False)

        if extrapolate == "error":
            # Check for out-of-range indices
            if torch.any(indices < 0) or torch.any(indices >= m.shape[-1]):
                raise ValueError("(some) interpolation indices out of range")
        elif extrapolate == "const":
            # Pad for constant values outside range
            m = torch.cat(
                [torch.zeros_like(m)[..., :1], m, torch.zeros_like(m)[..., :1]], dim=-1
            )
            b = torch.cat([fp[..., :1], b, fp[..., -1:]], dim=-1)
        else:  # linear extrapolation
            indices = torch.clamp(indices - 1, 0, m.shape[-1] - 1)

        values = m.gather(-1, indices) * x + b.gather(-1, indices)
        return values.movedim(-1, dim)

    def _interpquad(
        self,
        x: torch.Tensor,
        xp: torch.Tensor,
        fp: torch.Tensor,
        dim: int = -1,
        extrapolate: str = "const",
    ) -> torch.Tensor:
        """Quadratic interpolation between sample points using local fitting.

        Args:
            x: The x-coordinates at which to evaluate the interpolated values
            xp: The x-coordinates of the data points, must be increasing
            fp: The y-coordinates of the data points, same shape as xp
            dim: Dimension across which to interpolate
            extrapolate: How to handle values outside range ('quadratic' or 'const')

        Returns:
            The interpolated values, same size as x
        """
        # Move interpolation dimension to last axis
        x = x.movedim(dim, -1)
        xp = xp.movedim(dim, -1)
        fp = fp.movedim(dim, -1)

        # Find indices for the middle points of quadratic interpolation
        indices = torch.searchsorted(xp.contiguous(), x.contiguous(), right=False)
        indices = torch.clamp(indices, 1, xp.shape[-1] - 2)

        # Get three points for quadratic interpolation
        x0 = xp.gather(-1, indices - 1)
        x1 = xp.gather(-1, indices)
        x2 = xp.gather(-1, indices + 1)

        y0 = fp.gather(-1, indices - 1)
        y1 = fp.gather(-1, indices)
        y2 = fp.gather(-1, indices + 1)

        # Compute quadratic coefficients
        # Using Lagrange basis polynomials for quadratic interpolation
        l0 = ((x - x1) * (x - x2)) / ((x0 - x1) * (x0 - x2))
        l1 = ((x - x0) * (x - x2)) / ((x1 - x0) * (x1 - x2))
        l2 = ((x - x0) * (x - x1)) / ((x2 - x0) * (x2 - x1))

        # Compute interpolated values
        values = y0 * l0 + y1 * l1 + y2 * l2

        if extrapolate == "const":
            # Use constant values for extrapolation
            below_range = x < xp[..., 0]
            above_range = x > xp[..., -1]
            values = torch.where(below_range, fp[..., 0], values)
            values = torch.where(above_range, fp[..., -1], values)

        return values.movedim(-1, dim)

    def resample_to_linear(
        self,
        histogram: torch.Tensor,
        n_linear_bins: int = 200,
        linear_binsize: float = 1.0,
        interpolation: str = "quadratic",
        normalize: bool = False,
    ) -> torch.Tensor:
        """Resample a log-scale depth histogram to linear scale.

        Args:
            histogram: 4D histogram tensor (energy, depth, vertical, horizontal)
            n_linear_bins: Number of bins for linear scale
            linear_binsize: Bin size for linear scale in nm

        Returns:
            Resampled histogram with linear depth scale
        """
        if self.base_simulation_params["depth_mode"] != "log":
            raise ValueError("Can only resample from log to linear scale")

        # Calculate log-scale bin edges and centers
        bin_edges = (
            torch.exp(
                torch.arange(
                    0,
                    self.n_depth_bins * self.binsize_exit_depth
                    + self.binsize_exit_depth,
                    self.binsize_exit_depth,
                    device=histogram.device,
                    dtype=torch.float32,
                )
            )
            - 1.0
        )
        bin_markers = (bin_edges[1:] + bin_edges[:-1]) / 2
        bin_sizes = torch.diff(bin_edges)

        # Normalize histogram by bin sizes
        histogram_linearized = histogram / bin_sizes[None, :, None, None]

        # Create linear bin centers
        bin_centers_linear = torch.arange(
            linear_binsize / 2.0,
            n_linear_bins * linear_binsize,
            linear_binsize,
            device=histogram.device,
            dtype=torch.float32,
        )

        # Repeat dimensions for broadcasting
        bin_centers_linear = bin_centers_linear[None, :, None, None].repeat(
            self.n_fillable_bins, 1, self.n_direction_bins, self.n_direction_bins
        )
        bin_markers = bin_markers[None, :, None, None].repeat(
            self.n_fillable_bins, 1, self.n_direction_bins, self.n_direction_bins
        )

        # Resample histogram
        if interpolation == "linear":
            resampled_hist = self._interp(
                bin_centers_linear,
                bin_markers,
                histogram_linearized,
                dim=1,
                extrapolate="error",
            )
        elif interpolation == "quadratic":
            resampled_hist = self._interpquad(
                bin_centers_linear,
                bin_markers,
                histogram_linearized,
                dim=1,
                extrapolate="error",
            )
        else:
            raise ValueError(
                f"Invalid interpolation method: {interpolation}. Use 'linear' or 'quadratic'."
            )
        resampled_hist = torch.clamp(resampled_hist, min=0)

        # Normalize
        if normalize:
            resampled_hist = resampled_hist / resampled_hist.sum()

        return resampled_hist

    def run_simulation(
        self,
        n_electrons: Optional[int] = None,
        n_trials_per_electron: int = 10,
    ) -> Tuple[torch.Tensor, int]:
        """
        Run the Monte Carlo simulation.

        Args:
            n_electrons: Number of electrons to simulate (uses default if None)
            n_trials_per_electron: Number of trials per electron

        Returns:
            Tuple containing:
                - 4D histogram tensor (energy, depth, vertical, horizontal)
                - Total number of simulated electrons
        """
        simulation_params = self.base_simulation_params.copy()
        if n_electrons is not None:
            simulation_params["n_electrons"] = n_electrons
        simulation_params["n_trials_per_electron"] = n_trials_per_electron
        simulation_params["n_max_steps"] = self.n_max_steps

        total_bse = 0
        total_sim = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Run initial simulation
        histogram = run_monte_carlo_simulation(self.material_params, simulation_params)

        total_sim += (
            simulation_params["n_electrons"]
            * simulation_params["n_trials_per_electron"]
        )

        histogram = torch.tensor(histogram, device=device).float()
        histogram = histogram[self.n_energy_bins_extra :]

        if self.omega == 0.0:
            histogram = (histogram + histogram.flip(3)) / 2

        total_bse += histogram.sum().item()

        # Run additional simulations if needed to reach minimum mass
        while total_bse < self.min_histogram_mass:
            histogram_new = run_monte_carlo_simulation(
                self.material_params, simulation_params
            )

            histogram_new = torch.tensor(histogram_new, device=device).float()
            histogram_new = histogram_new[self.n_energy_bins_extra :]

            if self.omega == 0.0:
                histogram_new = (histogram_new + histogram_new.flip(3)) / 2

            total_bse += histogram_new.sum().item()
            total_sim += (
                simulation_params["n_electrons"]
                * simulation_params["n_trials_per_electron"]
            )

            histogram += histogram_new

        return histogram, total_sim


if __name__ == "__main__":
    # Nickel
    cell = Cell(
        sg_num=225,
        atom_data=[(28, 0.0, 0.0, 0.0, 1.0, 0.0035)],
        abc=(0.35236,) * 3,
        abc_units="nm",
        angles=(90.0, 90.0, 90.0),
        angles_units="deg",
    )

    # # NaCl
    # cell = Cell(
    #     sg_num=225,
    #     atom_data=[
    #         (11, 0.0, 0.0, 0.0, 1.0, 0.005),
    #         (17, 0.0, 0.0, 0.5, 1.0, 0.005),
    #     ],
    #     abc=(0.559, 0.559, 0.559),
    #     abc_units="nm",
    #     angles=(90.0, 90.0, 90.0),
    #     angles_units="deg",
    # )

    # # Regular pyrite (https://next-gen.materialsproject.org/materials/mp-226)
    # # has the 3-fold along the (x, x, x) diagonal so it doesn't give the wrong answer
    # cell = Cell(
    #     sg_num=205,
    #     atom_data=[
    #         (26, 0.0, 0.5, 0.5, 1.0, 0.005),  # 4a Wyckoff position
    #         (16, 0.38538, 0.11462, 0.88538, 1.0, 0.005),  # 16c Wyckoff position
    #     ],
    #     abc=(0.540, 0.540, 0.540),
    #     abc_units="nm",
    #     angles=(90.0, 90.0, 90.0),
    #     angles_units="deg",
    # )

    # Create simulator with log-scale depth bins
    simulator = MonteCarloSimulator(
        cell=cell,
        kV=30.0,
        incidence_angle=70.0,
        binsize_exit_energy=0.5,
        n_energy_bins_extra=4,
        # binsize_exit_depth=0.5,
        # n_depth_bins=200,
        # depth_mode="linear",
        binsize_exit_depth=0.1,
        n_depth_bins=80,
        depth_mode="log",
        min_histogram_mass=10_000_000,
        n_max_steps_override=1000,
    )

    # Run simulation
    histogram, total_electrons = simulator.run_simulation()

    bsize = 0.5
    histogram = simulator.resample_to_linear(
        histogram,
        n_linear_bins=200,
        linear_binsize=bsize,
    )
    histogram = histogram.cpu().numpy()

    print(f"histogram shape: {histogram.shape}")

    # visualize each marginal
    import matplotlib.pyplot as plt

    # # do 4 panel with each 1D marginal and then 6 panel with each 2D marginal
    # fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # labels = ["Energy", "Depth", "Vertical", "Horizontal"]

    # # 1D marginals
    # for i, ax in enumerate(axes.ravel()):
    #     ax.plot(histogram.sum(tuple([j for j in range(4) if j != i])))
    #     ax.set_title(f"{labels[i]} marginal")

    # plt.show()

    # fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # # do 4 choose 2
    # from itertools import combinations

    # for i, (a, b) in enumerate(combinations(range(4), 2)):
    #     ax = axes.ravel()[i]
    #     ax.imshow(
    #         histogram.sum(axis=tuple([j for j in range(4) if j != a and j != b])),
    #     )
    #     ax.set_title(f"{labels[a]} vs {labels[b]}")

    # plt.show()

    marginal_energy_depth = histogram.sum(axis=(2, 3))

    # normalize overall
    marginal_energy_depth /= marginal_energy_depth.sum()

    # normalize per energy
    marginal_energy_depth /= marginal_energy_depth.sum(axis=1)[:, None]

    cdf_energy_depth = marginal_energy_depth.cumsum(axis=1)

    # for each energy, find the depth at which 95% of the electrons have escaped
    depths_95 = (cdf_energy_depth > 0.95).argmax(axis=1) * bsize

    # plot energy on x axis and 95% depth cutoff on y axis for 20 kV through 30 kV in 1 keV steps
    plt.plot(depths_95[:21][::-1])
    plt.xlabel("Energy (keV)")
    # label the ticks from 20 to 30 at every other keV
    plt.xticks(range(0, 21, 2), range(20, 31))
    plt.ylabel("Depth (nm)")
    plt.xlim(0, 20)
    plt.ylim(0, 25)
    plt.show()
    print(depths_95)

    # plot line plots of depth pdf for each energy on a single plot
    counts = histogram.sum(axis=(2, 3))
    for i in range(10):
        marginal = counts[i]
        plt.plot(np.linspace(0.5, 99.5, 200), marginal, label=f"{30 - i}")
    plt.xlabel("Depth (nm)")
    # limit view from 0 to 50 nm and set ticks accordingly
    plt.xlim(0, 100)  # 0.5 sized bins
    plt.xticks(range(0, 101, 20), range(0, 51, 10))
    plt.ylabel("Probability")
    plt.legend()
    plt.show()
