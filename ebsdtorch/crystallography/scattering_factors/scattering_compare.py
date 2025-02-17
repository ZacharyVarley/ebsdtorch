import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

# Assuming the provided code is imported as:
from scattering_factors_WK import scatter_factor_WK
from scattering_factors_LVD_TCB import scatter_factor_LVD


def compare_scattering_factors(
    elements: List[Tuple[str, int, float]],  # [(symbol, Z, DWF), ...]
    voltage_kV: float = 100.0,
    g_range: Tuple[float, float] = (0.01, 4.0),
    num_points: int = 100,
    g_scale: str = "logspace",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # device: torch.device = torch.device("cpu"),
) -> None:
    """
    Compare scattering factors from WK and Thomas methods.

    Args:
        elements: List of (symbol, atomic number, DWF) tuples
        voltage_kV: Accelerating voltage in kV
        s_range: (min, max) s values in Å⁻¹
        num_points: Number of s points to evaluate
        device: 'cuda' or 'cpu'
    """
    # Set up device and dtype
    device = torch.device(device)
    dtype = torch.float64

    # Create logarithmically spaced s values
    if g_scale == "logspace":
        g_vals = torch.logspace(
            np.log10(g_range[0]),
            np.log10(g_range[1]),
            num_points,
            device=device,
            dtype=dtype,
        )  # in Å⁻¹
    elif g_scale == "linspace":
        g_vals = torch.linspace(
            g_range[0], g_range[1], num_points, device=device, dtype=dtype
        )  # in Å⁻¹

    # Prepare inputs for both methods
    Z_vals = torch.tensor([Z for _, Z, _ in elements], device=device, dtype=torch.int64)
    DWF_vals = torch.tensor([dwf for _, _, dwf in elements], device=device, dtype=dtype)

    # Calculate scattering factors using both methods
    wk_factors_lack_core = scatter_factor_WK(
        g_vals,  # in Å⁻¹
        Z_vals,  # atomic numbers
        torch.sqrt(DWF_vals / (8 * torch.pi**2)),  # <u> in Å
        voltage_kV,
        include_core=False,
        # return_type="ITC",
    )[
        0
    ]  # Take first voltage slice

    print(f"WK factors shape: {wk_factors_lack_core.shape}")

    # Calculate scattering factors using both methods
    wk_factors_with_core = scatter_factor_WK(
        g_vals,  # in Å⁻¹
        Z_vals,  # atomic numbers
        torch.sqrt(DWF_vals / (8 * torch.pi**2)),  # in Å^2
        voltage_kV,
        include_core=True,
    )[
        0
    ]  # Take first voltage slice

    print(f"WK factors shape: {wk_factors_with_core.shape}")

    thomas_factors = scatter_factor_LVD(
        g_vals,  # in Å⁻¹
        Z_vals,  # atomic numbers
        torch.sqrt(DWF_vals / (8 * torch.pi**2)),  # <u> in Å
        voltage_kV,
        intervals=256,
        interval_range=(0.0, 1e5),
        sampling_type="logspace",
    )[
        0
    ]  # Take first voltage slice

    print(f"Thomas factors shape: {thomas_factors.shape}")

    # Set up plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Color palette
    colors = sns.color_palette("husl", len(elements))

    # Plot real parts
    for i, ((symbol, _, _), color) in enumerate(zip(elements, colors)):
        ax1.plot(
            g_vals.cpu().numpy(),
            wk_factors_lack_core.cpu().numpy()[i].real,
            label=f"{symbol} (WK)",
            color=color,
            linestyle="solid",
        )
        ax1.plot(
            g_vals.cpu().numpy(),
            thomas_factors.cpu().numpy()[i].real,
            label=f"{symbol} (LVD)",
            color=color,
            linestyle="dashdot",
        )

        # cu_s_vals = np.array(
        #     [
        #         0.00,
        #         0.01,
        #         0.02,
        #         0.03,
        #         0.04,
        #         0.05,
        #         0.06,
        #         0.07,
        #         0.08,
        #         0.09,
        #         0.10,
        #         0.11,
        #         0.12,
        #         0.13,
        #         0.14,
        #         0.15,
        #         0.16,
        #         0.17,
        #         0.18,
        #         0.19,
        #         0.20,
        #         0.30,
        #         0.40,
        #         0.50,
        #         0.55,
        #         0.60,
        #         0.65,
        #         0.70,
        #         1.0,
        #         1.5,
        #         2.0,
        #     ]
        # )
        # cu_f_vals = np.array(
        #     [
        #         5.6,
        #         5.587,
        #         5.547,
        #         5.482,
        #         5.395,
        #         5.287,
        #         5.165,
        #         5.029,
        #         4.886,
        #         4.737,
        #         4.585,
        #         4.434,
        #         4.285,
        #         4.139,
        #         3.998,
        #         3.862,
        #         3.731,
        #         3.607,
        #         3.488,
        #         3.375,
        #         3.267,
        #         2.428,
        #         1.868,
        #         1.464,
        #         1.303,
        #         1.163,
        #         1.041,
        #         0.935,
        #         0.523,
        #         0.252,
        #         0.150,
        #     ]
        # )

        # ax1.plot(
        #     cu_s_vals,
        #     cu_f_vals,
        #     label=f"{symbol} (RHF)",
        #     color=color,
        #     linestyle="dashed",
        # )

    # if g_scale == "logspace":
    #     ax1.set_xscale("log")
    #     ax1.set_yscale("log")
    #     ax1.set_ylim(1e-7, 1e2)
    # ax1.set_yscale("log")
    ax1.set_xlabel("g (Å⁻¹)")
    ax1.set_ylabel("Real Part (Å)")
    # ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.set_title(f"Real Part of Scattering Factors at {voltage_kV} kV")

    # Plot imaginary parts
    for i, ((symbol, _, _), color) in enumerate(zip(elements, colors)):
        ax2.plot(
            g_vals.cpu().numpy(),
            wk_factors_lack_core.cpu().numpy()[i].imag,
            label=f"{symbol} (WK w/o core)",
            color=color,
            linestyle="solid",
        )
        ax2.plot(
            g_vals.cpu().numpy(),
            wk_factors_with_core.cpu().numpy()[i].imag,
            label=f"{symbol} (WK w/ core)",
            color=color,
            linestyle="dashed",
        )
        ax2.plot(
            g_vals.cpu().numpy(),
            thomas_factors.cpu().numpy()[i].imag,
            label=f"{symbol} LVD Integration",
            color=color,
            linestyle="dashdot",
        )

    # if g_scale == "logspace":
    #     ax2.set_xscale("log")
    # ax2.set_yscale("log")
    ax2.set_xlabel("g (Å⁻¹)")
    ax2.set_ylabel("Imaginary Part (Å)")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.set_title(f"Imaginary Part of Scattering Factors at {voltage_kV} kV")

    plt.tight_layout()
    plt.show()


# Run comparison
elements = [
    ("Li", 3, 4.4318),
    ("Al", 13, 0.7194),
    ("Ni", 28, 0.3280),
    ("Cu", 29, 0.5073),
    ("Au", 79, 0.5714),
    ("Pb", 82, 1.9627),
]

# # Run comparison (Debugging)
# elements = [
#     ("He", 2, 0.5),
#     ("Li", 3, 0.5),
#     ("Be", 4, 0.5),
#     ("B", 5, 0.5),
# ]

compare_scattering_factors(
    elements,
    voltage_kV=20.0,
    g_range=(0.01, 10.0),
    num_points=100,
    # g_scale="linspace",
    g_scale="logspace",
)
