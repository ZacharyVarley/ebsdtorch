"""
This module contains the Cell class for defining crystal structures. It keeps
diffraction computations separate from the crystal structure definition. Even
though many methods are written in torch, (e.g. closure of space groups), they
should almost always use the CPU instead of the GPU.

"""

import torch
from torch import Tensor
from typing import List, Tuple
from ebsdtorch.crystallography.groups import (
    sg_operators,
    sg_xt_beg,
    sg_pg_beg,
    pg_to_laue,
    sg_two_settings,
    sg_names,
)

# Define ATOM_weights
atomic_weights_dict = atomic_weights = torch.tensor(
    [
        v
        for v in {
            "H": 1.00794,
            "He": 4.002602,
            "Li": 6.941,
            "Be": 9.012182,
            "B": 10.811,
            "C": 12.0107,
            "N": 14.0067,
            "O": 15.9994,
            "F": 18.9984032,
            "Ne": 20.1797,
            "Na": 22.98976928,
            "Mg": 24.3050,
            "Al": 26.9815386,
            "Si": 28.0855,
            "P": 30.973762,
            "S": 32.065,
            "Cl": 35.453,
            "Ar": 39.948,
            "K": 39.0983,
            "Ca": 40.078,
            "Sc": 44.955912,
            "Ti": 47.867,
            "V": 50.9415,
            "Cr": 51.9961,
            "Mn": 54.938045,
            "Fe": 55.845,
            "Co": 58.933195,
            "Ni": 58.6934,
            "Cu": 63.546,
            "Zn": 65.38,
            "Ga": 69.723,
            "Ge": 72.64,
            "As": 74.92160,
            "Se": 78.96,
            "Br": 79.904,
            "Kr": 83.798,
            "Rb": 85.4678,
            "Sr": 87.62,
            "Y": 88.90585,
            "Zr": 91.224,
            "Nb": 92.90638,
            "Mo": 95.96,
            "Tc": 98.9062,
            "Ru": 101.07,
            "Rh": 102.90550,
            "Pd": 106.42,
            "Ag": 107.8682,
            "Cd": 112.411,
            "In": 114.818,
            "Sn": 118.710,
            "Sb": 121.760,
            "Te": 127.60,
            "I": 126.90447,
            "Xe": 131.293,
            "Cs": 132.9054519,
            "Ba": 137.327,
            "La": 138.90547,
            "Ce": 140.116,
            "Pr": 140.90765,
            "Nd": 144.242,
            "Pm": 145.0,
            "Sm": 150.36,
            "Eu": 151.964,
            "Gd": 157.25,
            "Tb": 158.92535,
            "Dy": 162.500,
            "Ho": 164.93032,
            "Er": 167.259,
            "Tm": 168.93421,
            "Yb": 173.054,
            "Lu": 174.9668,
            "Hf": 178.49,
            "Ta": 180.94788,
            "W": 183.84,
            "Re": 186.207,
            "Os": 190.23,
            "Ir": 192.217,
            "Pt": 195.084,
            "Au": 196.966569,
            "Hg": 200.59,
            "Tl": 204.3833,
            "Pb": 207.2,
            "Bi": 208.98040,
            "Po": 209.0,
            "At": 210.0,
            "Rn": 222.0,
            "Fr": 223.0,
            "Ra": 226.0,
            "Ac": 227.0,
            "Th": 232.03806,
            "Pa": 231.03588,
            "U": 238.02891,
            "Np": 237.0,
            "Pu": 244.0,
            "Am": 243.0,
            "Cm": 247.0,
            "Bk": 251.0,
            "Cf": 252.0,
        }.values()
    ],
    dtype=torch.float64,
)


@torch.jit.script
def unique_rows(a: Tensor, eps: float = 5e-4) -> Tensor:
    remove = torch.zeros_like(a[:, 0], dtype=torch.bool)
    for i in range(a.shape[0]):
        if not remove[i]:
            # equals = torch.all(torch.abs(a[i, :] - a[(i + 1) :]) < eps, dim=1)
            # use sum
            equals = torch.sum(torch.abs(a[i, :] - a[(i + 1) :]), dim=1) < eps
            remove[(i + 1) :] = torch.logical_or(remove[(i + 1) :], equals)
    return a[~remove]


class Cell:
    def __init__(
        self,
        sg_num: int,
        atom_data: List[Tuple[int, float, float, float, float, float]],
        abc: Tuple[float, float, float],
        abc_units: str,
        angles: Tuple[float, float, float],
        angles_units: str,
        sg_setting: int = 1,
        xtalname: str = "",
        source: str = "",
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            :abc:
                Tuple of lattice parameters (a, b, c) in nm, angstrom, or pm
            :abc_units:
                Units of lattice parameters ('nm', 'angstrom', or 'pm')
            :angles:
                Tuple of lattice angles (alpha, beta, gamma) in radians or
                degrees
            :angles_units:
                Units of lattice angles ('rad' or 'deg')
            :space_group_number:
                Space group number (1-230)
            :atom_data:
                List of tuples (atomic num, x, y, z, occupancy, Debye-Waller)
                Use 0.005 to estimate unknown Debye-Waller factors.
            :xtalname:
                Name of the crystal structure
            :source:
                Source of the Debian-Waller factors

        Note:
            Symmetrically equivalent positions are automatically generated using
            the space group number.

        """
        self.device = device

        # Check space group number
        if sg_num < 1 or sg_num > 230:
            raise ValueError("Space group number must be between 1 and 230")

        if sg_setting != 1 and sg_num not in sg_two_settings:
            raise ValueError(f"Space group {sg_num} does not have a second setting")

        self.sg_num = sg_num
        self.sg_setting = sg_setting

        # automatically find the point group for the space group
        pg_num = [i + 1 for i, sg in enumerate(sg_pg_beg) if sg_num <= sg]
        if len(pg_num) == 0:
            self.pg_num = 32
        else:
            self.pg_num = pg_num[0]

        # automatically find the crystal system for the space group
        self.xtal_system = [i + 1 for i, sg in enumerate(sg_xt_beg) if sg_num <= sg]
        if len(self.xtal_system) == 0:
            self.xtal_system = 7
        else:
            self.xtal_system = self.xtal_system[0]

        # automatically find the Laue group for the point group
        self.laue_group = pg_to_laue[self.pg_num]

        self.a, self.b, self.c = map(
            lambda x: torch.tensor(x, device=device, dtype=torch.float64), abc
        )

        if abc_units == "angstrom":
            # passed in angstrom, convert to nm
            self.a *= 0.1
            self.b *= 0.1
            self.c *= 0.1
        elif abc_units == "pm":
            # passed in pm, convert to nm
            self.a *= 1e-3
            self.b *= 1e-3
            self.c *= 1e-3
        elif abc_units == "nm":
            pass
        else:
            raise ValueError("abc_units must be 'nm', 'angstrom', or 'pm'")

        self.alpha, self.beta, self.gamma = map(
            lambda x: torch.tensor(x, device=device, dtype=torch.float64), angles
        )
        if angles_units == "deg":
            # passed in degrees, convert to radians
            self.alpha = torch.deg2rad(self.alpha)
            self.beta = torch.deg2rad(self.beta)
            self.gamma = torch.deg2rad(self.gamma)
        elif angles_units != "rad":
            raise ValueError("angles_units must be 'rad' or 'deg'")

        self.atom_types = torch.tensor(
            [a[0] for a in atom_data],
            device=device,
            dtype=torch.int64,
        )
        self.atom_types_unique = torch.unique(self.atom_types)
        self.atom_ntype = len(self.atom_types_unique)
        self.atom_data = torch.tensor(
            [a[1:] for a in atom_data],
            device=device,
            dtype=torch.float64,
        )
        self.xtalname = xtalname
        self.source = source

        # 1) direct metric tensor
        self.dmt = torch.tensor(
            [
                [
                    self.a**2,
                    self.a * self.b * torch.cos(self.gamma),
                    self.a * self.c * torch.cos(self.beta),
                ],
                [
                    self.a * self.b * torch.cos(self.gamma),
                    self.b**2,
                    self.b * self.c * torch.cos(self.alpha),
                ],
                [
                    self.a * self.c * torch.cos(self.beta),
                    self.b * self.c * torch.cos(self.alpha),
                    self.c**2,
                ],
            ],
            device=device,
        )

        # 2) volume is the root of the determinant of the metric tensor
        self.vol = torch.sqrt(torch.det(self.dmt))

        # Check volume
        if self.vol < 1e-6:
            raise ValueError(
                f"Unit cell volume of {self.vol} nm^3 is suspiciously small"
            )

        # 3) reciprocal metric tensor is the inverse of the direct metric tensor
        self.rmt = torch.linalg.inv(self.dmt)

        # 4) direct structure matrix
        self.dsm = torch.tensor(
            [
                [self.a, self.b * torch.cos(self.gamma), self.c * torch.cos(self.beta)],
                [
                    0.0,
                    self.b * torch.sin(self.gamma),
                    -self.c
                    * (
                        torch.cos(self.beta) * torch.cos(self.gamma)
                        - torch.cos(self.alpha)
                    )
                    / torch.sin(self.gamma),
                ],
                [0.0, 0.0, self.vol / (self.a * self.b * torch.sin(self.gamma))],
            ],
            device=device,
        )

        # 5) reciprocal structure matrix is the inverse of the direct structure matrix
        self.rsm = torch.linalg.inv(self.dsm)

        # 6) trigonal/rhombohedral case need a second direct structure matrix
        if self.xtal_system == 5:
            x = 0.5 / torch.cos(torch.pi * 0.5 * self.alpha)
            y = torch.sqrt(1 - x * x)
            Mx = torch.tensor([[1, 0, 0], [0, x, -y], [0, y, x]], device=device).T

            x = (
                2
                * torch.sin(torch.pi * 0.5 * self.alpha)
                / torch.sqrt(torch.tensor(3.0, device=device))
            )
            y = torch.sqrt(1 - x * x)
            My = torch.tensor([[x, 0, -y], [0, 1, 0], [y, 0, x]], device=device).T
            self.dsm2 = torch.matmul(torch.matmul(My, Mx), self.dsm)
        else:
            self.dsm2 = None

        # 7) get the space group operators: Tensor of shape (n_ops, 4, 4)
        self.sg_ops = sg_operators(self.sg_num, self.sg_setting)

        # now we compute the orbit (set of unique positions) for each atom in
        # the asymmetric unit
        self.apos = []
        # get the position, form homogeneous coordinates and apply Seitz matrices
        pos = self.atom_data[:, :3]
        pos = torch.cat([pos, torch.ones_like(pos[:, [0]])], dim=-1)
        # (1, n_ops, 4, 4) @ (n_pos, 1, 4, 1) -> (n_pos, n_ops, 4, 1) -> (n_pos, n_ops, 3)
        all_pos = torch.matmul(
            self.sg_ops[None, :, :, :], pos[:, None, :, None]
        ).squeeze(-1)[..., :3]

        # catch slightly negative positions, bring back to [0, 1), and catch again
        all_pos = torch.where(
            (all_pos).abs() < 1e-6, torch.zeros_like(all_pos), all_pos
        )
        all_pos = torch.fmod(all_pos + 100.0, 1.0)
        all_pos = torch.where(
            (all_pos).abs() < 1e-6, torch.zeros_like(all_pos), all_pos
        )

        # remove duplicates
        for i in range(all_pos.shape[0]):
            self.apos.append(unique_rows(all_pos[i], eps=1e-4))
        self.mults = torch.tensor(
            [len(a) for a in self.apos], device=device, dtype=torch.int64
        )

        # 7b) store number of unique atoms in the asymmetric unit
        self.n_s = len(self.apos)

        # 8) Compute average atomic number, weight and density
        self.average_atomic_number = (
            torch.mean(
                self.atom_types.to(torch.float64) * self.atom_data[:, 3] * self.mults
            )
            / self.mults.sum()
        )

        a_weights = atomic_weights[self.atom_types - 1].to(self.device)
        self.average_atomic_weight = (
            torch.mean(a_weights * self.atom_data[:, 3] * self.mults) / self.mults.sum()
        )

        # Density in g/cm^3
        # self.density = self.average_atomic_weight / (self.vol * 6.02214076e2)
        self.density = (a_weights * self.atom_data[:, 3] * self.mults).sum() / (
            self.vol * 6.02214076e2
        )

    def __str__(self) -> str:
        """Return a string representation of the crystal structure."""
        crystal_systems = {
            1: "Triclinic",
            2: "Monoclinic",
            3: "Orthorhombic",
            4: "Tetragonal",
            5: "Trigonal",
            6: "Hexagonal",
            7: "Cubic",
        }

        header = f"Crystal Structure: {self.xtalname if self.xtalname else 'Unnamed'}"
        system = f"{crystal_systems[self.xtal_system]}"
        spacegrp = f"Space Group: {self.sg_num}"

        # Convert angles back to degrees for display
        angles_deg = [
            torch.rad2deg(a).item() for a in (self.alpha, self.beta, self.gamma)
        ]
        cell = f"(a, b, c) = ({self.a:.5f}, {self.b:.5f}, {self.c:.5f}) nm | (α, β, γ) = ({angles_deg[0]:.2f}, {angles_deg[1]:.2f}, {angles_deg[2]:.2f})°"

        volume = f"Volume: {self.vol:.4f} nm³"
        density = f"Density: {self.density:.4f} g/cm³"

        # Format atomic positions
        atoms = []
        atoms.append(
            "Atom | Z  | (x, y, z) in xtal basis  |  #  | % Occup  | Debye-Waller (nm²)"
        )
        atoms.append("-" * 70)
        for i, (z, pos) in enumerate(zip(self.atom_types, self.atom_data)):
            atoms.append(
                f" {i + 1}   | {z:02d} | ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) | {(len(self.apos[i])):02d}  | {pos[3]:.4f}   | {pos[4]:.6f}"
            )

        return "\n".join(
            [
                header,
                spacegrp + f" ({system})" + f" setting {self.sg_setting}",
                cell,
                volume + " | " + density,
                *atoms,
            ]
        )

    def to(self, device: torch.device) -> "Cell":
        """Method to move all tensors to specified device."""
        if device == self.device:
            return self
        self.device = device
        self.a = self.a.to(device)
        self.b = self.b.to(device)
        self.c = self.c.to(device)
        self.alpha = self.alpha.to(device)
        self.beta = self.beta.to(device)
        self.gamma = self.gamma.to(device)
        self.dmt = self.dmt.to(device)
        self.rmt = self.rmt.to(device)
        self.dsm = self.dsm.to(device)
        self.rsm = self.rsm.to(device)
        if self.dsm2 is not None:
            self.dsm2 = self.dsm2.to(device)
        self.atom_types = self.atom_types.to(device)
        self.atom_types_unique = self.atom_types_unique.to(device)
        self.atom_data = self.atom_data.to(device)
        self.apos = [a.to(device) for a in self.apos]
        self.mults = self.mults.to(device)
        self.sg_ops = self.sg_ops.to(device)
        self.average_atomic_number = self.average_atomic_number.to(device)
        self.average_atomic_weight = self.average_atomic_weight.to(device)
        self.density = self.density.to(device)
        self.vol = self.vol.to(device)
        return self

    def set_fp_dtype(self, dtype: torch.dtype) -> "Cell":
        """Set dtype of tensors that are floats."""
        self.a = self.a.to(dtype)
        self.b = self.b.to(dtype)
        self.c = self.c.to(dtype)
        self.alpha = self.alpha.to(dtype)
        self.beta = self.beta.to(dtype)
        self.gamma = self.gamma.to(dtype)
        self.dmt = self.dmt.to(dtype)
        self.rmt = self.rmt.to(dtype)
        self.dsm = self.dsm.to(dtype)
        self.rsm = self.rsm.to(dtype)
        if self.dsm2 is not None:
            self.dsm2 = self.dsm2.to(dtype)
        self.atom_data = self.atom_data.to(dtype)
        self.apos = [a.to(dtype) for a in self.apos]
        self.mults = self.mults.to(dtype)
        self.sg_ops = self.sg_ops.to(dtype)
        self.average_atomic_number = self.average_atomic_number.to(dtype)
        self.average_atomic_weight = self.average_atomic_weight.to(dtype)
        self.density = self.density.to(dtype)
        self.vol = self.vol.to(dtype)
        return self

    @property
    def get_density(self) -> Tensor:
        """Returns the unit cell density"""
        return self.density

    @property
    def get_average_atomic_number(self) -> Tensor:
        """Returns the average atomic number"""
        return self.average_atomic_number

    @property
    def get_average_atomic_weight(self) -> Tensor:
        """Returns the average atomic weight"""
        return self.average_atomic_weight

    @property
    def get_volume(self) -> Tensor:
        """Returns the unit cell volume"""
        return self.vol

    @property
    def get_direct_metric_tensor(self) -> Tensor:
        """Returns the direct metric tensor"""
        return self.dmt

    @property
    def get_reciprocal_metric_tensor(self) -> Tensor:
        """Returns the reciprocal metric tensor"""
        return self.rmt

    @property
    def get_direct_structure_matrix(self) -> Tensor:
        """Returns the direct structure matrix"""
        return self.dsm

    @property
    def get_reciprocal_structure_matrix(self) -> Tensor:
        """Returns the reciprocal structure matrix"""
        return self.rsm

    def transform_space(self, t: torch.Tensor, inspace: str, outspace: str) -> Tensor:
        """
        Convert vector components from inspace to outspace.

        Args:
            t (torch.Tensor): Input vector(s) in inspace reference frame.
                            Can be of shape (3,), (N, 3), or (..., 3).
            inspace (str): Character to label input space ('d', 'r', or 'c').
            outspace (str): Character to label output space ('d', 'r', or 'c').

        Returns:
            torch.Tensor: Output vector(s) in outspace reference frame.
                        Will have the same shape as the input.
        """
        # Check if the last dimension is 3
        if t.shape[-1] != 3:
            raise ValueError("The last dimension of input tensor must be 3")

        # Intercept the case where inspace and outspace are the same
        if inspace == outspace:
            return t

        # Reshape t to (..., 3) if it's (3,)
        original_shape = t.shape
        t = t.reshape(-1, 3)

        if inspace == "d":
            if outspace == "c":
                d = torch.einsum("ij,...j->...i", self.dsm, t)
            elif outspace == "r":
                d = torch.einsum("...i,ij->...j", t, self.dmt)
        elif inspace == "r":
            if outspace == "c":
                d = torch.einsum("ij,...j->...i", self.rsm, t)
            elif outspace == "d":
                d = torch.einsum("...i,ij->...j", t, self.rmt)
        elif inspace == "c":
            if outspace == "d":
                d = torch.einsum("...i,ij->...j", t, self.rsm)
            elif outspace == "r":
                d = torch.einsum("...i,ij->...j", t, self.dsm)
        else:
            raise ValueError(f"Invalid transformation: from {inspace} to {outspace}")

        # Reshape back to original shape
        return d.reshape(original_shape)

    def transform_coordinates(
        self,
        t: torch.Tensor,
        talpha: torch.Tensor,
        space: str,
        direction: str,
    ) -> Tensor:
        """
        Convert vector components from one reference frame to another.

        This is a general coordinate transformation using the old-to-new matrix alpha.
        The details of this routine are summarized in Table 1.6, page 51, of the textbook.

        Args:
            t (torch.Tensor): Input vector(s) w.r.t. input space reference frame.
                            Can be of shape (3,), (N, 3), or (..., 3).
            talpha (torch.Tensor): Transformation matrix of shape (3, 3).
            space (str): Space in which to perform transformation ('d', 'r', or 'c').
            direction (str): Transformation direction ('on'=old-to-new, 'no'=new-to-old).

        Returns:
            torch.Tensor: Transformed vector components. Will have the same shape as the input.
        """
        # Check if the last dimension of t is 3
        if t.shape[-1] != 3:
            raise ValueError("The last dimension of input tensor must be 3")

        # Reshape t to (..., 3) if it's (3,)
        original_shape = t.shape
        t = t.reshape(-1, 3)

        if space == "d":
            if direction == "on":
                d = torch.einsum("...i,ji->...j", t, talpha)
            else:
                d = torch.einsum("...i,ij->...j", t, talpha)
        else:
            if direction == "on":
                d = torch.einsum("ij,...j->...i", talpha, t)
            else:
                d = torch.einsum("ji,...j->...i", talpha, t)

        # Reshape back to original shape
        return d.reshape(original_shape)

    def calc_dot(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        space: str,
    ) -> Tensor:
        """
        Compute the dot product between two vectors in real, reciprocal, or Cartesian space.

        Args:
            p (torch.Tensor): First input vector(s) in space reference frame.
            q (torch.Tensor): Second input vector(s).
            space (str): Space in which to compute product ('d', 'r', or 'c').

        Returns:
            torch.Tensor: Dot product p.q
        """
        if space == "d":
            return torch.einsum("...i,ij,...j->...", p, self.dmt, q)
        elif space == "r":
            return torch.einsum("...i,ij,...j->...", p, self.rmt, q)
        elif space == "c":
            return torch.einsum("...i,...i->...", p, q)
        else:
            raise ValueError(f"Invalid space: {space}")

    def calc_auto_dot(
        self,
        p: torch.Tensor,
        space: str,
    ) -> Tensor:
        """
        Compute the dot product between two vectors in real, reciprocal, or Cartesian space.

        Args:
            p (torch.Tensor): vector(s) in space reference frame.
            space (str): Space in which to compute product ('d', 'r', or 'c').

        Returns:
            torch.Tensor: Dot product p.q
        """
        if space == "d":
            return torch.einsum("...i,ij,...j->...", p, self.dmt, p)
        elif space == "r":
            return torch.einsum("...i,ij,...j->...", p, self.rmt, p)
        elif space == "c":
            return torch.einsum("...i,...i->...", p, p)
        else:
            raise ValueError(f"Invalid space: {space}")

    def norm_vec(
        self,
        p: torch.Tensor,
        space: str,
    ) -> Tensor:
        """
        Normalize vector(s) in arbitrary space.

        Args:
            p (torch.Tensor): Input/output vector components.
            space (str): Space character ('d', 'r', or 'c').

        Returns:
            torch.Tensor: Normalized vector(s).
        """
        x = self.calc_length(p, space)
        return torch.where(x[..., None] != 0, p / x[..., None], torch.zeros_like(p))

    def calc_length(
        self,
        p: torch.Tensor,
        space: str,
    ) -> Tensor:
        """
        Calculate vector length(s) in arbitrary space.

        Args:
            p (torch.Tensor): Input vector components.
            space (str): Space character ('d', 'r', or 'c').

        Returns:
            torch.Tensor: Vector length(s).
        """
        # if it is an integer dtype, convert to float
        if p.dtype in [torch.int32, torch.int64]:
            p = p.to(self.dsm.dtype)
        return torch.sqrt(self.calc_auto_dot(p, space))

    def calc_angle(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        space: str,
    ) -> Tensor:
        """
        Calculate angle between vectors in arbitrary space.

        Args:
            p (torch.Tensor): First vector components.
            q (torch.Tensor): Second vector components.
            space (str): Space of the computation ('d', 'r', or 'c').

        Returns:
            torch.Tensor: Angle(s) in radians.
        """
        x = self.calc_dot(p, q, space)
        y = self.calc_length(p, space)
        z = self.calc_length(q, space)

        if torch.any((y == 0) | (z == 0)):
            raise ValueError("Vector of zero length specified")

        t = x / (y * z)
        return torch.where(
            t >= 1.0,
            torch.tensor(0.0, device=self.device),
            torch.where(t <= -1.0, torch.pi, torch.arccos(t)),
        )

    def calc_cross(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        inspace: str,
        outspace: str,
        scale_by_volume: bool,
    ) -> Tensor:
        """
        Compute cross product between vectors in arbitrary space.

        Args:
            p (torch.Tensor): First input vector(s).
            q (torch.Tensor): Second input vector(s).
            inspace (str): Input space character ('d','r','c').
            outspace (str): Output space character ('d','r','c').
            scale_by_volume (bool): Whether to scale by unit cell volume.

        Returns:
            torch.Tensor: Cross product vector(s).
        """
        vl = self.vol if scale_by_volume else torch.tensor(1.0, device=self.device)

        if inspace == "d":
            r = vl * torch.cross(p, q, dim=-1)
            if outspace == "d":
                return torch.einsum("...i,ij->...j", r, self.rmt)
            elif outspace == "c":
                return torch.einsum("ij,...j->...i", self.rsm, r)
        elif inspace == "r":
            r = torch.cross(p, q, dim=-1) / vl
            if outspace == "r":
                return torch.einsum("...i,ij->...j", r, self.dmt)
            elif outspace == "c":
                return torch.einsum("ij,...j->...i", self.dsm, r)
        elif inspace == "c":
            return torch.cross(p, q, dim=-1)

        raise ValueError(f"Invalid space transformation: from {inspace} to {outspace}")

    @staticmethod
    def z2_percent(
        atom_types: torch.Tensor, numat: torch.Tensor, atom_pos: torch.Tensor
    ) -> Tensor:
        """
        Calculate the Z^2 percent contributions for each atom type.

        Args:
            atom_types: Atomic numbers for each atom type
            numat: Number of atoms of each type
            atom_pos: Atomic positions including occupancy factors

        Returns:
            torch.Tensor: Array of Z^2 percentages
        """
        Z2list = numat * atom_pos[:, 3] * atom_types**2
        return 100.0 * Z2list / Z2list.sum()

    def convert_from_R_to_H(self) -> None:
        """
        Convert lattice parameters and atom coordinates to the hexagonal unit cell.
        We use the obverse setting in all cases.
        """
        # Convert lattice parameters to hexagonal
        calpha = torch.cos(self.alpha * torch.pi / 180.0)
        ar = self.a

        self.alpha = torch.tensor(90.0, device=self.device)
        self.beta = torch.tensor(90.0, device=self.device)
        self.gamma = torch.tensor(120.0, device=self.device)
        self.a = ar * torch.sqrt(torch.tensor(2.0, device=self.device) - 2.0 * calpha)
        self.b = self.a
        self.c = ar * torch.sqrt(torch.tensor(3.0, device=self.device) + 6.0 * calpha)

        # Transform matrix
        M = torch.tensor(
            [
                [2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                [-1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                [-1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0],
            ],
            device=self.device,
        )

        # Transform atom coordinates
        for i in range(self.atom_ntype):
            pos = self.atom_data[i, :3].clone()
            pos = torch.matmul(M, pos)
            self.atom_data[i, :3] = pos

    def is_g_allowed(self, g: torch.Tensor) -> bool:
        """Determine whether a reflection is allowed by lattice centering.

        Args:
            g (torch.Tensor): Reciprocal lattice vector indices [h,k,l]

        Returns:
            torch.Tensor: Boolean mask indicating which reflections are allowed
        """

        # Get the lattice centering type from the name
        lc = sg_names[self.sg_num - 1][1]

        if lc == "P":  # Primitive
            return torch.ones_like(g[..., 0], dtype=torch.bool)

        elif lc == "F":  # Face centered
            # parity must be all even or all odd
            sums = torch.sum(g % 2, dim=-1)
            return (sums == 0) | (sums == 3)

        elif lc == "I":  # Body centered (sum of indices must be even)
            return (torch.sum(g, dim=-1) % 2) == 0

        elif lc == "A":  # A-centered (sum of k and l must be even)
            return (g[..., 1] + g[..., 2]) % 2 == 0

        elif lc == "B":  # B-centered (sum of h and l must be even)
            return (g[..., 0] + g[..., 2]) % 2 == 0

        elif lc == "C":  # C-centered (sum of h and k must be even)
            return (g[..., 0] + g[..., 1]) % 2 == 0

        elif lc == "R":  # Rhombohedral
            # assume hexagonal setting
            print("Assuming hexagonal setting for rhombohedral lattice.")
            return (-g[..., 0] + g[..., 1] + g[..., 2]) % 3 == 0
        else:
            raise ValueError(f"Unknown lattice centering type: {lc}")

    def get_hkl_limits(self, dmin: float) -> Tuple[int, int, int]:
        """
        Calculate the maximum h, k, and l indices for a given d-spacing.

        Args:
            dmin (float): Minimum d-spacing to consider in nm

        Returns:
            Tuple[int, int, int]: Maximum h, k, and l indices
        """
        # First find approximate max indices along principal directions
        test_range = torch.arange(1, 100, device=self.device, dtype=torch.int64)

        # Test vectors along each axis
        h_vec = torch.stack(
            [test_range, torch.zeros_like(test_range), torch.zeros_like(test_range)],
            dim=-1,
        )
        k_vec = torch.stack(
            [torch.zeros_like(test_range), test_range, torch.zeros_like(test_range)],
            dim=-1,
        )
        l_vec = torch.stack(
            [torch.zeros_like(test_range), torch.zeros_like(test_range), test_range],
            dim=-1,
        )

        # Calculate d-spacings for axial reflections
        d_h = 1.0 / self.calc_length(h_vec, "r")
        d_k = 1.0 / self.calc_length(k_vec, "r")
        d_l = 1.0 / self.calc_length(l_vec, "r")

        # Find maximum index needed for each axis
        max_h = (
            torch.nonzero(d_h < dmin)[0].item()
            if len(torch.nonzero(d_h < dmin)) > 0
            else len(test_range)
        )
        max_k = (
            torch.nonzero(d_k < dmin)[0].item()
            if len(torch.nonzero(d_k < dmin)) > 0
            else len(test_range)
        )
        max_l = (
            torch.nonzero(d_l < dmin)[0].item()
            if len(torch.nonzero(d_l < dmin)) > 0
            else len(test_range)
        )

        return max_h, max_k, max_l

    def get_reflections(
        self,
        dmin: float,
        difference_table: bool,
        filter_centering: bool = True,
    ) -> Tensor:
        """
        Compute the range of reflections for the lookup table using a meshgrid approach.
        Maximum test value is automatically determined by checking axial reflections.

        Args:
            dmin (float): Minimum d-spacing to consider in nm
            difference_table (bool): Whether to return a table of valid reflection
                differences (True) or a table of valid reflections (False)

        Returns:
            Tensor: (N, 3) tensor of hkl indices that adhere to the dmin
        """

        # difference tables are not quarter tables and only filter by centering
        if difference_table:
            filter_dmin = False
        else:
            filter_dmin = True

        # First find approximate max indices along principal directions
        test_range = torch.arange(1, 100, device=self.device, dtype=torch.int64)

        # Test vectors along each axis
        h_vec = torch.stack(
            [test_range, torch.zeros_like(test_range), torch.zeros_like(test_range)],
            dim=-1,
        )
        k_vec = torch.stack(
            [torch.zeros_like(test_range), test_range, torch.zeros_like(test_range)],
            dim=-1,
        )
        l_vec = torch.stack(
            [torch.zeros_like(test_range), torch.zeros_like(test_range), test_range],
            dim=-1,
        )

        # Calculate d-spacings for axial reflections
        d_h = 1.0 / self.calc_length(h_vec, "r")
        d_k = 1.0 / self.calc_length(k_vec, "r")
        d_l = 1.0 / self.calc_length(l_vec, "r")

        # Find maximum index needed for each axis
        max_h = (
            torch.nonzero(d_h < dmin)[0].item()
            if len(torch.nonzero(d_h < dmin)) > 0
            else len(test_range)
        )
        max_k = (
            torch.nonzero(d_k < dmin)[0].item()
            if len(torch.nonzero(d_k < dmin)) > 0
            else len(test_range)
        )
        max_l = (
            torch.nonzero(d_l < dmin)[0].item()
            if len(torch.nonzero(d_l < dmin)) > 0
            else len(test_range)
        )

        if difference_table:  # differences of hkl (quadruple the size)
            h = torch.arange(
                -2 * max_h + 1, 2 * max_h, device=self.device, dtype=torch.int64
            )
            k = torch.arange(
                -2 * max_k + 1, 2 * max_k, device=self.device, dtype=torch.int64
            )
            l = torch.arange(
                -2 * max_l + 1, 2 * max_l, device=self.device, dtype=torch.int64
            )
            hh, kk, ll = torch.meshgrid(h, k, l, indexing="ij")
            # Stack into vectors
            hkl = torch.stack([hh, kk, ll], dim=-1).reshape(-1, 3)
            # remove (0,0,0)
            hkl = hkl[(hkl != 0).any(dim=-1)]
        else:  # normal reflection table
            # Create grid of h,k,l indices
            h = torch.arange(-max_h + 1, max_h, device=self.device, dtype=torch.int64)
            k = torch.arange(-max_k + 1, max_k, device=self.device, dtype=torch.int64)
            l = torch.arange(-max_l + 1, max_l, device=self.device, dtype=torch.int64)
            hh, kk, ll = torch.meshgrid(h, k, l, indexing="ij")
            # Stack into vectors
            hkl = torch.stack([hh, kk, ll], dim=-1).reshape(-1, 3)[1:]  # remove (0,0,0)

        if filter_centering:
            # Check if reflections are allowed by lattice centering
            allowed = self.is_g_allowed(hkl)

            hkl = hkl[allowed]

        if filter_dmin:
            # Calculate d-spacings (1/length in reciprocal space)
            d_hkl = 1.0 / self.calc_length(hkl.reshape(-1, 3), "r")

            # Find maximum needed indices where d-spacing ≥ dmin (all in nm)
            valid = d_hkl > dmin

            # return the (N,3) tensor of hkl that adhere to the dmin
            hkl = hkl[valid]

        return hkl


# # Nickel
# cell = Cell(
#     sg_num=225,
#     atom_data=[
#         (28, 0.0, 0.0, 0.0, 1.0, 0.00328),
#     ],
#     abc=(0.3524, 0.3524, 0.3524),
#     abc_units="nm",
#     angles=(90.0, 90.0, 90.0),
#     angles_units="deg",
# )

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

# # check the reflection range
# ref = cell.get_reflections().cpu().numpy()
# print(ref)
# print(len(ref))

# # print(cell)

# # NiS2 - this fails right now
# cell = Cell(
#     sg_num=205,
#     sg_setting=1,
#     atom_data=[
#         (28, 0.0, 0.0, 0.5, 1.0, 0.005),  # Ni at 4a Wyckoff position
#         (16, 0.5680, 0.5690, 0.5700, 1.0, 0.005),
#         # S at 8c Wyckoff position - ??? none of the plotted S positions in unit cell
#         # at Materials Project are in a special position - very confused but I will leave it here:
#         # https://next-gen.materialsproject.org/materials/mp-2282
#     ],
#     abc=(0.5680, 0.5690, 0.5700),
#     abc_units="nm",
#     angles=(90.0, 90.0, 90.0),
#     angles_units="deg",
# )
# print(cell)  # Density should be 4.43 g/cm^3

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
# print(cell)  # Density should be 5.07 g/cm^3

# # check the reflection range
# ref = cell.get_reflections().cpu().numpy()
# print(ref)
# print(len(ref))


# # FeSO4
# cell = Cell(
#     sg_num=62,
#     atom_data=[
#         (26, 0.5, 0.5, 0.5, 1.0, 0.005),  # Fe at 4a Wyckoff position
#         (16, 0.524136, 0.75, 0.822351, 1.0, 0.005),  # S at 4c Wyckoff position
#         (8, 0.232441, 0.75, 0.878885, 1.0, 0.005),  # O at 4c Wyckoff position
#         (8, 0.532202, 0.75, 0.650776, 1.0, 0.005),  # O at 4c Wyckoff position
#         (8, 0.835619, 0.930553, 0.374675, 1.0, 0.005),  # O at 8d Wyckoff position
#     ],
#     abc=(0.478, 0.675, 0.870),
#     abc_units="nm",
#     angles=(90.0, 90.0, 90.0),
#     angles_units="deg",
# )
# print(cell)  # Density should be 10.55 g/cm^3

# # Fe2P2O7
# cell = Cell(
#     sg_num=1,
#     atom_data=[
#         (26, 0.505455, 0.255512, 0.708082, 1.0, 0.005),
#         (26, 0.544846, 0.606076, 0.302714, 1.0, 0.005),
#         (15, 0.938616, 0.1433, 0.209556, 1.0, 0.005),
#         (15, 0.120705, 0.718928, 0.787498, 1.0, 0.005),
#         (8, 0.239998, 0.308523, 0.371617, 1.0, 0.005),
#         (8, 0.75986, 0.982369, 0.351896, 1.0, 0.005),
#         (8, 0.749722, 0.309053, 0.084447, 1.0, 0.005),
#         (8, 0.034505, 0.945726, 0.988598, 1.0, 0.005),
#         (8, 0.304393, 0.558993, 0.924587, 1.0, 0.005),
#         (8, 0.306299, 0.865284, 0.636169, 1.0, 0.005),
#         (8, 0.817001, 0.549336, 0.634235, 1.0, 0.005),
#     ],
#     abc=(0.458, 0.530, 0.563),
#     abc_units="nm",
#     angles=(103.46, 98.57, 99.32),
#     angles_units="deg",
# )
# print(cell)  # Density should be 3.69 g/cm^3
