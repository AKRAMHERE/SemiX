#!/usr/bin/env python3
"""
Transistor Models Module

This module contains simplified models for BJT, MOSFET, and JFET devices.
It is intended for educational purposes to illustrate basic device physics and
characteristics. The models and parameter values are illustrative and not meant
for production-level circuit simulation.

Author: Akram Syed (github.com/akramhere)
Date: 2025-02-02
"""

import numpy as np
import matplotlib.pyplot as plt

# Define semiconductor material properties.
# Note: The property values are for demonstration only.
MATERIALS = {
    "Si": {"bandgap": 1.12, "mobility": 1400, "dielectric": 11.7},
    "Ge": {"bandgap": 0.66, "mobility": 3900, "dielectric": 16},
    "GaAs": {"bandgap": 1.43, "mobility": 8500, "dielectric": 12.9},
    "SiC": {"bandgap": 3.26, "mobility": 700, "dielectric": 9.7},
}


class Transistor:
    """
    A collection of transistor device models: BJT, MOSFET, and JFET.

    The models use simplified equations for the device characteristics and
    include example plotting routines.
    """

    class Bjt:
        """
        Bipolar Junction Transistor (BJT) model using a simplified Ebers-Moll
        approach.
        """

        def __init__(self, material: str, beta: float = 100, V_A: float = 50,
                     I_S: float = 1e-12, temperature: float = 300):
            """
            Initialize the BJT model with basic parameters.

            Parameters
            ----------
            material : str
                Semiconductor material (e.g., 'Si', 'Ge').
            beta : float, optional
                Current gain (common-emitter). Default is 100.
            V_A : float, optional
                Early voltage (V). Default is 50 V.
            I_S : float, optional
                Saturation current (A). Default is 1e-12 A.
            temperature : float, optional
                Operating temperature (K). Default is 300 K.
            """
            if material not in MATERIALS:
                raise ValueError(
                    f"Material '{material}' not supported. "
                    f"Available: {list(MATERIALS.keys())}"
                )
            self.material = material
            self.beta = beta         # Current gain (h_FE)
            self.V_A = V_A           # Early voltage (V)
            self.I_S = I_S           # Saturation current (A)
            self.temperature = temperature

            # Thermal voltage: V_T ~ kT/q, approximately 0.0259 V at 300 K.
            self.V_T = 0.0259 * (temperature / 300)

            # Store material properties for reference.
            self.properties = MATERIALS[material]

        def collector_current(self, VBE: float, VBC: float) -> float:
            """
            Calculate the collector current using a simplified Ebers-Moll model.

            I_C = I_S * (exp(V_BE / V_T) - exp(V_BC / V_T))

            Parameters
            ----------
            VBE : float
                Base-emitter voltage (V).
            VBC : float
                Base-collector voltage (V).

            Returns
            -------
            float
                Collector current (A).
            """
            return self.I_S * (np.exp(VBE / self.V_T) -
                               np.exp(VBC / self.V_T))

        def base_current(self, VBE: float, VBC: float) -> float:
            """
            Calculate the base current assuming a current gain 'beta'.

            I_B = I_C / beta

            Parameters
            ----------
            VBE : float
                Base-emitter voltage (V).
            VBC : float
                Base-collector voltage (V).

            Returns
            -------
            float
                Base current (A).
            """
            return self.collector_current(VBE, VBC) / self.beta

        def emitter_current(self, VBE: float, VBC: float) -> float:
            """
            Calculate the emitter current.

            I_E = I_C + I_B

            Parameters
            ----------
            VBE : float
                Base-emitter voltage (V).
            VBC : float
                Base-collector voltage (V).

            Returns
            -------
            float
                Emitter current (A).
            """
            I_C = self.collector_current(VBE, VBC)
            I_B = self.base_current(VBE, VBC)
            return I_C + I_B

        def plot_i_v(self, VBE_range=(0, 1), VCE_range=(0, 10), steps=100):
            """
            Plot the collector current vs. collector-emitter voltage (output
            characteristics) for a fixed VBE value.

            The Early voltage effect is included using a simplified scaling:
            I_C = I_C0 * (1 + V_CE / V_A).

            Parameters
            ----------
            VBE_range : tuple, optional
                Range of base-emitter voltage (start, end) in volts.
                Default is (0, 1).
            VCE_range : tuple, optional
                Range of collector-emitter voltage (start, end) in volts.
                Default is (0, 10).
            steps : int, optional
                Number of points in the voltage sweep. Default is 100.
            """
            VBE = np.linspace(*VBE_range, steps)
            VCE = np.linspace(*VCE_range, steps)

            # Use the midpoint of VBE range for a typical curve.
            VBE_mid = (VBE_range[0] + VBE_range[1]) / 2
            I_C0 = self.collector_current(VBE_mid, 0)

            # Scale current with the Early effect.
            I_C = I_C0 * (1 + VCE / self.V_A)

            # Convert to milliamperes.
            I_C_mA = I_C * 1e3

            plt.figure(figsize=(6, 4))
            plt.plot(VCE, I_C_mA, label=f'BJT {self.material}')
            plt.xlabel("V_CE (V)")
            plt.ylabel("I_C (mA)")
            plt.title(f"BJT Output Characteristics ({self.material})")
            plt.legend()
            plt.grid(True)
            plt.show()

        def hybrid_pi_model(self, I_C: float) -> dict:
            """
            Calculate the hybrid-pi small-signal parameters for the BJT.

            g_m = I_C / V_T
            r_pi = beta / g_m
            r_o ~ (V_A + V_CE) / I_C (assume V_CE = 10 V for demonstration)
            C_pi and C_mu are example small-signal capacitances.

            Parameters
            ----------
            I_C : float
                Collector current (A).

            Returns
            -------
            dict
                Dictionary with keys: 'g_m', 'r_pi', 'r_o', 'C_pi', and 'C_mu'.
            """
            g_m = I_C / self.V_T      # Transconductance (S)
            r_pi = self.beta / g_m    # Base resistance (Ω)
            r_o = (self.V_A + 10) / I_C  # Output resistance (Ω), assume V_CE=10 V
            C_pi = g_m * self.V_T * 1e-12  # Example diffusion capacitance (F)
            C_mu = 2e-12  # Example junction capacitance (F)
            return {"g_m": g_m, "r_pi": r_pi, "r_o": r_o,
                    "C_pi": C_pi, "C_mu": C_mu}

        def diffusion_capacitance(self, I_C: float) -> float:
            """
            Estimate the diffusion capacitance.

            C_diff = tau_F * I_C / V_T, where tau_F is the base transit time.

            Parameters
            ----------
            I_C : float
                Collector current (A).

            Returns
            -------
            float
                Diffusion capacitance (F).
            """
            tau_F = 1e-9  # 1 ns transit time (example)
            return tau_F * I_C / self.V_T

        def temperature_effects(self, VBE_range=(0, 1), VCE=5,
                                temp_range=(250, 450), steps=100):
            """
            Plot the impact of temperature on the collector current vs. V_BE.

            Parameters
            ----------
            VBE_range : tuple, optional
                Range of base-emitter voltage (V). Default is (0, 1).
            VCE : float, optional
                Fixed collector-emitter voltage (V). Default is 5 V.
            temp_range : tuple, optional
                Temperature range (K). Default is (250, 450).
            steps : int, optional
                Number of points for the V_BE sweep. Default is 100.
            """
            VBE_values = np.linspace(*VBE_range, steps)

            plt.figure(figsize=(6, 4))
            for temp in np.linspace(*temp_range, 5):
                # Recalculate thermal voltage for each temperature.
                VT = 0.0259 * (temp / 300)
                I_C = self.I_S * (np.exp(VBE_values / VT) - 1)
                # Include Early voltage effect.
                I_C *= (1 + VCE / self.V_A)
                I_C_mA = I_C * 1e3  # Convert to mA
                plt.plot(VBE_values, I_C_mA, label=f'T = {temp:.1f} K')

            plt.xlabel("V_BE (V)")
            plt.ylabel("I_C (mA)")
            plt.title(f"BJT Temperature Effects ({self.material})")
            plt.legend()
            plt.grid(True)
            plt.show()

    class Mosfet:
        """
        MOSFET model using a simplified long-channel quadratic I-V equation.
        """

        def __init__(self, material: str, Vth: float = 1.0,
                     mu_n: float = None, Cox: float = 10e-3):
            """
            Initialize the MOSFET model.

            Parameters
            ----------
            material : str
                Semiconductor material (e.g., 'Si', 'Ge').
            Vth : float, optional
                Threshold voltage (V). Default is 1.0 V.
            mu_n : float or None, optional
                Electron mobility in cm^2/(V*s). If None, the material's default
                is used.
            Cox : float, optional
                Oxide capacitance per unit area (F/cm^2). Default is 10e-3 F/cm^2.
            """
            if material not in MATERIALS:
                raise ValueError(
                    f"Material '{material}' not supported. "
                    f"Available: {list(MATERIALS.keys())}"
                )
            self.material = material
            self.Vth = Vth
            self.Cox = Cox
            self.mu_n = mu_n if mu_n is not None else MATERIALS[material]["mobility"]

        def id_vs_vds(self, VGS_values, VDS_range):
            """
            Plot the drain current (I_D) vs. drain-source voltage (V_DS) for
            multiple gate-source voltages (V_GS).

            Regions:
              - Triode (linear): V_DS < (V_GS - Vth)
                I_D = μ_n * Cox * [(V_GS - Vth) * V_DS - 0.5 * V_DS^2]
              - Saturation: V_DS >= (V_GS - Vth)
                I_D = 0.5 * μ_n * Cox * (V_GS - Vth)^2

            Parameters
            ----------
            VGS_values : list or array-like
                List of V_GS values (V) to plot.
            VDS_range : tuple
                (start, end) values of V_DS (V).
            """
            VDS = np.linspace(*VDS_range, 100)
            plt.figure(figsize=(6, 4))
            for VGS in VGS_values:
                # Calculate current in the triode region.
                ID_triode = (self.mu_n * self.Cox) * (
                    (VGS - self.Vth) * VDS - 0.5 * VDS ** 2)
                # Calculate current in the saturation region.
                ID_sat = 0.5 * self.mu_n * self.Cox * (VGS - self.Vth) ** 2

                # Use triode equation if in the triode region, else saturation.
                ID = np.where(VDS < (VGS - self.Vth), ID_triode, ID_sat)

                # Ensure zero current below threshold.
                if VGS <= self.Vth:
                    ID[:] = 0.0

                plt.plot(VDS, ID * 1e3, label=f"V_GS = {VGS} V")

            plt.xlabel("V_DS (V)")
            plt.ylabel("I_D (mA)")
            plt.title(f"{self.material} MOSFET Output Characteristics")
            plt.legend()
            plt.grid(True)
            plt.show()

        def plot_transfer_characteristic(self, VGS_range, VDS: float = 5,
                                           steps: int = 100):
            """
            Plot the transfer characteristic: drain current (I_D) vs. gate-
            source voltage (V_GS) for a fixed V_DS.

            In saturation (V_DS > V_GS - Vth), I_D = 0.5 * μ_n * Cox *
            (V_GS - Vth)^2. For V_GS below Vth, I_D is assumed to be zero.

            Parameters
            ----------
            VGS_range : tuple
                (start, end) values for V_GS (V).
            VDS : float, optional
                Fixed drain-source voltage (V). Default is 5 V.
            steps : int, optional
                Number of points in V_GS sweep. Default is 100.
            """
            VGS = np.linspace(*VGS_range, steps)
            I_D = np.zeros_like(VGS)
            for idx, vgs in enumerate(VGS):
                if vgs > self.Vth:
                    I_D[idx] = 0.5 * self.mu_n * self.Cox * (vgs - self.Vth) ** 2
                else:
                    I_D[idx] = 0.0

            plt.figure(figsize=(6, 4))
            plt.plot(VGS, I_D * 1e3, color="purple")
            plt.xlabel("V_GS (V)")
            plt.ylabel("I_D (mA)")
            plt.title(f"{self.material} MOSFET Transfer Characteristic\n"
                      f"(V_DS = {VDS} V)")
            plt.grid(True)
            plt.show()

        def calculate_transconductance(self, VGS: float) -> float:
            """
            Calculate the transconductance (g_m) for the MOSFET in saturation.

            In saturation, g_m = μ_n * Cox * (V_GS - Vth).

            Parameters
            ----------
            VGS : float
                Gate-source voltage (V).

            Returns
            -------
            float
                Transconductance (S). Returns 0 if VGS is below threshold.
            """
            if VGS > self.Vth:
                return self.mu_n * self.Cox * (VGS - self.Vth)
            return 0.0

    class Jfet:
        """
        JFET model using a simplified quadratic equation in saturation.
        """

        def __init__(self, material: str, Vp: float = -4,
                     Idss: float = 10e-3):
            """
            Initialize the JFET model.

            Parameters
            ----------
            material : str
                Semiconductor material (e.g., 'Si', 'GaAs').
            Vp : float, optional
                Pinch-off voltage (V); typically negative for an n-channel JFET.
                Default is -4 V.
            Idss : float, optional
                Saturation drain current (A) at V_GS = 0. Default is 10e-3 A.
            """
            if material not in MATERIALS:
                raise ValueError(
                    f"Material '{material}' not supported. "
                    f"Available: {list(MATERIALS.keys())}"
                )
            self.material = material
            self.Vp = Vp     # Pinch-off voltage (V)
            self.Idss = Idss  # Saturation current (A) at V_GS = 0

        def id_vs_vds(self, VGS_values, VDS_range):
            """
            Plot the drain current (I_D) vs. drain-source voltage (V_DS) for
            various gate-source voltages (V_GS).

            For a JFET in saturation:
            I_D = Idss * (1 - (V_GS / Vp))^2, ignoring channel-length modulation.

            Parameters
            ----------
            VGS_values : list or array-like
                Gate-source voltage values (V) for plotting.
            VDS_range : tuple
                (start, end) values of V_DS (V).
            """
            VDS = np.linspace(*VDS_range, 100)
            plt.figure(figsize=(6, 4))
            for VGS in VGS_values:
                # Calculate saturation current for a given V_GS.
                ID_sat = self.Idss * (1 - (VGS / self.Vp)) ** 2
                ID_plot = np.full_like(VDS, ID_sat)
                plt.plot(VDS, ID_plot * 1e3, label=f"V_GS = {VGS} V")

            plt.xlabel("V_DS (V)")
            plt.ylabel("I_D (mA)")
            plt.title(f"{self.material} JFET Output Characteristics")
            plt.legend()
            plt.grid(True)
            plt.show()

        def calculate_id(self, VGS: float) -> float:
            """
            Calculate the drain current (I_D) for a given gate-source voltage.

            Uses the equation:
                I_D = Idss * (1 - (V_GS / Vp))^2,
            valid for V_GS between 0 and Vp (note: Vp is negative for an n-channel
            JFET).

            Parameters
            ----------
            VGS : float
                Gate-source voltage (V). Should be negative for proper biasing.

            Returns
            -------
            float
                Drain current (A). Returns 0 if V_GS is above 0.
            """
            # For an n-channel JFET, conduction occurs for VGS <= 0.
            if VGS > 0:
                return 0.0
            return self.Idss * (1 - (VGS / self.Vp)) ** 2

        def plot_transfer_characteristics(self, VGS_range, steps: int = 100):
            """
            Plot the transfer characteristic: drain current (I_D) vs. gate-
            source voltage (V_GS).

            Parameters
            ----------
            VGS_range : tuple
                (start, end) values for V_GS (V). Typically, these values are
                negative for an n-channel JFET.
            steps : int, optional
                Number of points in the V_GS sweep. Default is 100.
            """
            VGS = np.linspace(*VGS_range, steps)
            I_D = np.array([self.calculate_id(vgs) for vgs in VGS])

            plt.figure(figsize=(6, 4))
            plt.plot(VGS, I_D * 1e3, color="green")
            plt.xlabel("V_GS (V)")
            plt.ylabel("I_D (mA)")
            plt.title(f"{self.material} JFET Transfer Characteristic")
            plt.grid(True)
            plt.show()

