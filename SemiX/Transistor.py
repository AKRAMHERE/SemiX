#!/usr/bin/env python3
"""
Transistor Models Module (Expanded Version)

+ This module contains *simplified* models for BJT, MOSFET, and JFET devices.
+ It is intended for educational/demonstration purposes to illustrate basic
+ device physics and characteristics. The models and parameter values
+ are illustrative and not meant for production-level circuit simulation.

Key expansions beyond the "improved version":
  - Gummel plot and a toy beta(Ic) variation for the BJT.
  - Subthreshold conduction, gm/Id plotting for the MOSFET.
  - Triode region for the JFET (complete piecewise I-V).
  
Original Author:     Akram Syed (github.com/akramhere)

"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Define semiconductor material properties.
# Note: The property values are purely for demonstration.
# Mobility is given in cm^2/(V*s). We'll convert to m^2/(V*s) where needed.
# -----------------------------------------------------------------------------
MATERIALS = {
    "Si":   {"bandgap": 1.12, "mobility": 1400, "dielectric": 11.7},
    "Ge":   {"bandgap": 0.66, "mobility": 3900, "dielectric": 16},
    "GaAs": {"bandgap": 1.43, "mobility": 8500, "dielectric": 12.9},
    "SiC":  {"bandgap": 3.26, "mobility": 700,  "dielectric": 9.7},
}


def thermal_voltage(temperature: float) -> float:
    """
    Compute the thermal voltage at a given temperature.

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Thermal voltage in volts (approx).
    """
    # kT/q ~ 0.02585 V at 300 K, scale linearly
    return 0.02585 * (temperature / 300.0)


class Transistor:
    """
    A collection of transistor device models: BJT, MOSFET, and JFET,
    with extended methods for additional plots and characteristics.
    """

    class Bjt:
        """
        Bipolar Junction Transistor (BJT) model using a simplified Ebers-Moll
        approach and some small-signal approximations.
        """

        def __init__(
            self,
            material: str,
            beta: float = 100,
            V_A: float = 50.0,
            I_S: float = 1e-12,
            temperature: float = 300.0,
        ):
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

            # Thermal voltage: V_T ~ kT/q
            self.V_T = thermal_voltage(temperature)

            # Store material properties for reference (not heavily used here).
            self.properties = MATERIALS[material]

        def collector_current(self, VBE: float, VBC: float = 0.0) -> float:
            """
            Calculate the collector current using a simplified Ebers-Moll model.

            I_C = I_S * [exp(V_BE / V_T) - exp(V_BC / V_T)]

            Parameters
            ----------
            VBE : float
                Base-emitter voltage (V).
            VBC : float, optional
                Base-collector voltage (V). Default is 0 V.

            Returns
            -------
            float
                Collector current (A).
            """
            return self.I_S * (np.exp(VBE / self.V_T) - np.exp(VBC / self.V_T))

        def base_current(self, VBE: float, VBC: float = 0.0) -> float:
            """
            Calculate the base current assuming a current gain 'beta'.

            I_B = I_C / beta

            Parameters
            ----------
            VBE : float
                Base-emitter voltage (V).
            VBC : float, optional
                Base-collector voltage (V). Default is 0 V.

            Returns
            -------
            float
                Base current (A).
            """
            return self.collector_current(VBE, VBC) / self.beta

        def emitter_current(self, VBE: float, VBC: float = 0.0) -> float:
            """
            Calculate the emitter current.

            I_E = I_C + I_B

            Parameters
            ----------
            VBE : float
                Base-emitter voltage (V).
            VBC : float, optional
                Base-collector voltage (V). Default is 0 V.

            Returns
            -------
            float
                Emitter current (A).
            """
            I_C = self.collector_current(VBE, VBC)
            I_B = self.base_current(VBE, VBC)
            return I_C + I_B

        def plot_i_v(self, VBE_range=(0.0, 1.0), VCE_range=(0.0, 10.0), steps=100):
            """
            Plot the collector current vs. collector-emitter voltage (output
            characteristics) for a fixed V_BE value (the midpoint of VBE_range).
            Uses a simplified Early-voltage scaling:
                I_C = I_C0 * (1 + V_CE / V_A)

            Parameters
            ----------
            VBE_range : tuple, optional
                Range of base-emitter voltage (start, end) in volts.
            VCE_range : tuple, optional
                Range of collector-emitter voltage (start, end) in volts.
            steps : int, optional
                Number of points in the voltage sweep. Default is 100.
            """
            VBE_mid = (VBE_range[0] + VBE_range[1]) / 2
            VCE = np.linspace(*VCE_range, steps)

            # Assume VBC ~ 0 => the collector current at that VBE.
            I_C0 = self.collector_current(VBE_mid, 0.0)
            # Include Early effect.
            I_C = I_C0 * (1.0 + VCE / self.V_A)

            I_C_mA = I_C * 1e3

            plt.figure(figsize=(6, 4))
            plt.plot(VCE, I_C_mA, label=f"{self.material} BJT")
            plt.xlabel("V_CE (V)")
            plt.ylabel("I_C (mA)")
            plt.title(f"BJT Output Characteristics - Fixed V_BE={VBE_mid:.2f}V")
            plt.legend()
            plt.grid(True)
            plt.show()

        def hybrid_pi_model(self, I_C: float) -> dict:
            """
            Calculate the hybrid-pi small-signal parameters for the BJT.

            g_m = I_C / V_T
            r_pi = beta / g_m
            r_o ~ (V_A + V_CE) / I_C  (assume V_CE = 10 V for demonstration)
            C_pi and C_mu are example capacitances.

            Parameters
            ----------
            I_C : float
                Collector current (A).

            Returns
            -------
            dict
                Dictionary with keys: 'g_m', 'r_pi', 'r_o', 'C_pi', and 'C_mu'.
            """
            if I_C <= 0:
                raise ValueError("Collector current must be > 0 for small-signal model.")
            g_m = I_C / self.V_T      # Transconductance (S)
            r_pi = self.beta / g_m    # Base resistance (Ω)
            # For demonstration, assume V_CE = 10 V
            r_o = (self.V_A + 10.0) / I_C if I_C > 0 else np.inf
            # Example values for junction/diffusion capacitances
            C_pi = g_m * self.V_T * 1e-12  # simplified guess
            C_mu = 2e-12                  # constant guess
            return {
                "g_m": g_m,
                "r_pi": r_pi,
                "r_o": r_o,
                "C_pi": C_pi,
                "C_mu": C_mu,
            }

        def diffusion_capacitance(self, I_C: float) -> float:
            """
            Estimate the diffusion capacitance using:
                C_diff = tau_F * I_C / V_T
            where tau_F is the forward transit time.

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

        def temperature_effects(
            self, VBE_range=(0.0, 1.0), VCE=5.0, temp_range=(250, 450), steps=100
        ):
            """
            Plot the impact of temperature on collector current vs. V_BE.
            Uses a simple diode-like equation plus an Early effect factor.

            Parameters
            ----------
            VBE_range : tuple, optional
                Range of base-emitter voltage (V). Default is (0, 1).
            VCE : float, optional
                Fixed collector-emitter voltage (V). Default is 5 V.
            temp_range : tuple, optional
                Temperature range (K). Default is (250, 450).
            steps : int, optional
                Number of points in the V_BE sweep. Default is 100.
            """
            VBE_values = np.linspace(*VBE_range, steps)

            plt.figure(figsize=(6, 4))
            temp_points = np.linspace(temp_range[0], temp_range[1], 5)
            for temp in temp_points:
                VT = thermal_voltage(temp)
                # Simple diode eqn: I_C = I_S * (exp(VBE/VT) - 1)
                I_C = self.I_S * (np.exp(VBE_values / VT) - 1.0)
                # Include Early effect: I_C * (1 + VCE / VA)
                I_C *= (1.0 + VCE / self.V_A)
                I_C_mA = I_C * 1e3  # convert to mA
                plt.plot(VBE_values, I_C_mA, label=f"T = {temp:.0f} K")

            plt.xlabel("V_BE (V)")
            plt.ylabel("I_C (mA)")
            plt.title(f"BJT Temperature Effects ({self.material})")
            plt.legend()
            plt.grid(True)
            plt.show()

        # ---------------------------------------------------------------------
        # Additional / Missing Formulae for BJT
        # ---------------------------------------------------------------------

        def gummel_plot(self, VBE_range=(0.5, 0.8), steps=100):
            """
            Plot a Gummel plot: log(I_C) and log(I_B) vs. V_BE (assuming VBC=0).
            """
            VBE_vals = np.linspace(*VBE_range, steps)
            Ic_vals = []
            for vbe in VBE_vals:
                Ic_vals.append(self.collector_current(VBE=vbe, VBC=0.0))

            Ic_vals = np.array(Ic_vals)
            Ib_vals = Ic_vals / self.beta

            plt.figure(figsize=(6, 4))
            # Plot I_C on a semi-log scale
            plt.semilogy(VBE_vals, Ic_vals, label='I_C')
            # Plot I_B on a semi-log scale
            plt.semilogy(VBE_vals, Ib_vals, label='I_B')

            plt.xlabel("V_BE (V)")
            plt.ylabel("Current (A) [log scale]")
            plt.title(f"Gummel Plot ({self.material} BJT)")
            plt.legend()
            plt.grid(True)
            plt.show()

        def beta_vs_ic_plot(self, Ic_range=(1e-6, 1e-2), steps=100):
            """
            Plot a toy model of how beta might vary with collector current.
            In real devices, beta can vary with Ic (beta roll-off at high Ic).

            Here we use a fictional formula:
                beta(Ic) = beta_0 * [1 + alpha * ln(Ic / Iref)]
            just to demonstrate a typical shape. 
            """
            Ic_vals = np.logspace(np.log10(Ic_range[0]), np.log10(Ic_range[1]), steps)
            # Example toy shape: 
            # reference current = 1e-6; alpha = 0.05 for demonstration
            Iref = 1e-6
            alpha = 0.05
            beta_vals = self.beta * (1 + alpha * np.log(Ic_vals / Iref))

            plt.figure(figsize=(6,4))
            plt.semilogx(Ic_vals, beta_vals)
            plt.xlabel("I_C (A)")
            plt.ylabel("beta (h_FE)")
            plt.title(f"Toy Beta Variation with I_C ({self.material} BJT)")
            plt.grid(True)
            plt.show()

    class Mosfet:
        """
        MOSFET model using a simplified long-channel quadratic I-V equation.
        Ignores subthreshold conduction and other second-order effects by default,
        but includes optional subthreshold and gm/Id plots in extended methods.
        """

        def __init__(
            self,
            material: str,
            Vth: float = 1.0,
            mu_n: float = None,  # in cm^2/(V*s)
            Cox: float = 1e-2,   # F/m^2
            W: float = 1e-4,     # m
            L: float = 1e-6,     # m
            lambda_mod: float = 0.02,
        ):
            """
            Initialize the MOSFET model.

            Parameters
            ----------
            material : str
                Semiconductor material (e.g., 'Si', 'Ge', 'GaAs').
            Vth : float, optional
                Threshold voltage (V). Default is 1.0 V.
            mu_n : float or None, optional
                Electron mobility in cm^2/(V*s). If None, uses material's default.
            Cox : float, optional
                Gate oxide capacitance per unit area (F/m^2). Default is 1e-2 F/m^2.
            W : float, optional
                Channel width (m). Default is 1e-4 m.
            L : float, optional
                Channel length (m). Default is 1e-6 m.
            lambda_mod : float, optional
                Channel-length modulation parameter (1/V). Default is 0.02 V^-1.
            """
            if material not in MATERIALS:
                raise ValueError(
                    f"Material '{material}' not supported. "
                    f"Available: {list(MATERIALS.keys())}"
                )
            self.material = material
            self.Vth = Vth

            # Mobility in cm^2/(V*s) -> convert to m^2/(V*s)
            if mu_n is None:
                mu_n = MATERIALS[material]["mobility"]
            self.mu_n = mu_n / 1e4

            self.Cox = Cox
            self.W = W
            self.L = L
            self.lambda_mod = lambda_mod

        def _triode_current(self, VGS, VDS):
            """
            Triode-region (linear) current:
            I_D = mu_n * Cox * (W/L) * [ (VGS - Vth)*VDS - 0.5 * VDS^2 ], 
            valid when 0 < VDS < (VGS - Vth).
            """
            return (
                self.mu_n
                * self.Cox
                * (self.W / self.L)
                * ((VGS - self.Vth) * VDS - 0.5 * VDS**2)
            )

        def _saturation_current(self, VGS, VDS):
            """
            Saturation-region current (with channel-length modulation):
            I_D = 0.5 * mu_n * Cox * (W/L) * (VGS - Vth)^2 * (1 + lambda_mod * VDS).
            valid when VDS >= (VGS - Vth).
            """
            return (
                0.5
                * self.mu_n
                * self.Cox
                * (self.W / self.L)
                * (VGS - self.Vth)**2
                * (1.0 + self.lambda_mod * VDS)
            )

        def id_vs_vds(self, VGS_values, VDS_range):
            """
            Plot the drain current (I_D) vs. drain-source voltage (V_DS) for
            multiple gate-source voltages (V_GS).

            Piecewise model:
                - Triode (linear): V_DS < (V_GS - Vth)
                - Saturation: V_DS >= (V_GS - Vth)
            
            Ignores subthreshold conduction by default. If V_GS <= Vth, I_D = 0.

            Parameters
            ----------
            VGS_values : list or array-like
                List of V_GS values (V) to plot.
            VDS_range : tuple
                (start, end) values of V_DS (V).
            """
            VDS = np.linspace(*VDS_range, 200)

            plt.figure(figsize=(6, 4))
            for VGS in VGS_values:
                ID = np.zeros_like(VDS)
                if VGS > self.Vth:
                    # Triode region condition
                    triode_mask = VDS < (VGS - self.Vth)
                    ID[triode_mask] = self._triode_current(VGS, VDS[triode_mask])

                    # Saturation region condition
                    sat_mask = ~triode_mask
                    ID[sat_mask] = self._saturation_current(VGS, VDS[sat_mask])

                # Convert to mA
                ID_mA = ID * 1e3
                plt.plot(VDS, ID_mA, label=f"V_GS = {VGS} V")

            plt.xlabel("V_DS (V)")
            plt.ylabel("I_D (mA)")
            plt.title(f"{self.material} MOSFET Output Characteristics")
            plt.legend()
            plt.grid(True)
            plt.show()

        def plot_transfer_characteristic(
            self, VGS_range, VDS: float = 5.0, steps: int = 100
        ):
            """
            Plot the transfer characteristic: drain current (I_D) vs. gate-source
            voltage (V_GS) for a fixed V_DS.

            In saturation (V_DS > V_GS - Vth), 
            I_D = 0.5 * mu_n * Cox * (W/L) * (V_GS - Vth)^2 * (1 + lambda_mod * V_DS).

            Parameters
            ----------
            VGS_range : tuple
                (start, end) values for V_GS (V).
            VDS : float, optional
                Fixed drain-source voltage (V). Default is 5 V.
            steps : int, optional
                Number of points in V_GS sweep. Default is 100.
            """
            VGS_vals = np.linspace(*VGS_range, steps)
            I_D = np.zeros_like(VGS_vals)
            for i, VGS in enumerate(VGS_vals):
                if VGS > self.Vth:
                    I_D[i] = self._saturation_current(VGS, VDS)

            plt.figure(figsize=(6, 4))
            plt.plot(VGS_vals, I_D * 1e3, color="purple")
            plt.xlabel("V_GS (V)")
            plt.ylabel("I_D (mA)")
            plt.title(
                f"{self.material} MOSFET Transfer Characteristic (V_DS = {VDS} V)"
            )
            plt.grid(True)
            plt.show()

        def calculate_transconductance(self, VGS: float) -> float:
            """
            Calculate the transconductance (g_m) for the MOSFET in saturation.

            In saturation: g_m = mu_n * Cox * (W/L) * (V_GS - Vth).

            Parameters
            ----------
            VGS : float
                Gate-source voltage (V).

            Returns
            -------
            float
                Transconductance (S). Returns 0 if VGS <= Vth.
            """
            if VGS <= self.Vth:
                return 0.0
            return self.mu_n * self.Cox * (self.W / self.L) * (VGS - self.Vth)

        def small_signal_params(self, VGS: float, VDS: float) -> dict:
            """
            Calculate small-signal parameters for the MOSFET in saturation.

            Returns a dictionary containing:
                I_D (A): DC drain current
                g_m (S): Transconductance
                r_o (ohms): Output resistance, ~ 1 / (lambda_mod * I_D)

            Parameters
            ----------
            VGS : float
                Gate-source voltage (V). Must be > Vth.
            VDS : float
                Drain-source voltage (V). Should be >= (VGS - Vth) for saturation.

            Returns
            -------
            dict
                Dictionary with keys: 'I_D', 'g_m', and 'r_o'.
            """
            if VGS <= self.Vth:
                raise ValueError("VGS must be greater than Vth for conduction.")
            # Saturation-region current
            I_D = self._saturation_current(VGS, VDS)
            g_m = self.calculate_transconductance(VGS)
            if I_D > 0 and self.lambda_mod > 0:
                r_o = 1.0 / (self.lambda_mod * I_D)
            else:
                r_o = np.inf

            return {
                "I_D": I_D,
                "g_m": g_m,
                "r_o": r_o,
            }

        # ---------------------------------------------------------------------
        # Additional / Missing Formulae for MOSFET
        # ---------------------------------------------------------------------

        def subthreshold_id(self, VGS, n=1.2, I0=1e-14, temperature=300.0):
            """
            Simple subthreshold conduction current:
                I_D = I0 * exp( (VGS - Vth)/(n*V_T) )
            for VGS < Vth. n is the ideality factor (~1.2).
            """
            VT = thermal_voltage(temperature)
            return I0 * np.exp((VGS - self.Vth) / (n * VT))

        def plot_transfer_characteristic_with_subthreshold(
            self, 
            VGS_range, 
            VDS=5.0, 
            steps=100, 
            n=1.2, 
            I0=1e-14, 
            temperature=300.0
        ):
            """
            Plot the transfer characteristic including a subthreshold region:
            If VGS < Vth, use the exponential subthreshold model.
            If VGS >= Vth, use saturation-region model.
            Log-scale on I_D for clarity.
            """
            VGS_vals = np.linspace(*VGS_range, steps)
            ID = np.zeros_like(VGS_vals)
            for i, VGS in enumerate(VGS_vals):
                if VGS < self.Vth:
                    ID[i] = self.subthreshold_id(VGS, n=n, I0=I0, temperature=temperature)
                else:
                    ID[i] = self._saturation_current(VGS, VDS)

            plt.figure(figsize=(6,4))
            plt.semilogy(VGS_vals, ID, label="I_D")
            plt.xlabel("V_GS (V)")
            plt.ylabel("I_D (A) [log scale]")
            plt.title(f"MOSFET Transfer w/ Subthreshold (VDS={VDS}V)")
            plt.grid(True)
            plt.legend()
            plt.show()

        def plot_gm_over_id(self, VGS_range=(1, 5), steps=100):
            """
            Plots gm/Id vs. V_GS in the saturation region of a MOSFET.
            In the simple long-channel model:
                gm = mu_n * Cox * (W/L)*(VGS - Vth)
                Id = 0.5 * mu_n * Cox*(W/L)*(VGS - Vth)^2
                => gm/Id = 2 / (VGS - Vth)
            """
            VGS_vals = np.linspace(*VGS_range, steps)
            gm_id_vals = []
            for VGS in VGS_vals:
                if VGS > self.Vth:
                    gm = self.calculate_transconductance(VGS)
                    # Use large VDS to ensure saturation
                    Id = 0.5 * self.mu_n * self.Cox * (self.W / self.L) * (VGS - self.Vth)**2
                    gm_id_vals.append(gm / Id if Id != 0 else 0)
                else:
                    gm_id_vals.append(0)

            plt.figure(figsize=(6,4))
            plt.plot(VGS_vals, gm_id_vals)
            plt.xlabel("V_GS (V)")
            plt.ylabel("g_m / I_D (1/V)")
            plt.title("MOSFET g_m / I_D vs. V_GS")
            plt.grid(True)
            plt.show()

    class Jfet:
        """
        JFET model using a simplified quadratic equation in saturation.
        Assumes an n-channel device with pinch-off voltage Vp < 0.
        """

        def __init__(
            self,
            material: str,
            Vp: float = -4.0,   # negative pinch-off (n-channel typical)
            Idss: float = 10e-3 # saturation current at V_GS=0
        ):
            """
            Initialize the JFET model.

            Parameters
            ----------
            material : str
                Semiconductor material (e.g., 'Si', 'GaAs', 'Ge').
            Vp : float, optional
                Pinch-off voltage (V); negative for an n-channel JFET. Default is -4 V.
            Idss : float, optional
                Saturation drain current (A) at V_GS = 0. Default is 10 mA.
            """
            if material not in MATERIALS:
                raise ValueError(
                    f"Material '{material}' not supported. "
                    f"Available: {list(MATERIALS.keys())}"
                )
            self.material = material
            self.Vp = Vp   # typically negative for n-channel
            self.Idss = Idss

        def calculate_id(self, VGS: float) -> float:
            """
            Calculate the drain current (I_D) for a given gate-source voltage.

            Simplified JFET saturation equation:
                I_D = Idss * [1 - (V_GS / Vp)]^2

            Valid for V_GS <= 0 (for an n-channel). If V_GS >= 0, set I_D = 0
            (since forward-biasing the gate is out of scope for this model).
            
            If V_GS <= Vp (even more negative), channel is pinched off => I_D ~ 0.

            Parameters
            ----------
            VGS : float
                Gate-source voltage (V).

            Returns
            -------
            float
                Drain current (A).
            """
            if VGS >= 0.0:
                return 0.0
            if VGS <= self.Vp:
                return 0.0
            return self.Idss * (1.0 - (VGS / self.Vp))**2

        def id_vs_vds(self, VGS_values, VDS_range):
            """
            Plot the drain current (I_D) vs. drain-source voltage (V_DS) for
            various gate-source voltages (V_GS).

            In a simplified JFET saturation model, once in saturation,
            I_D ~ Idss * [1 - (V_GS / Vp)]^2  (independent of V_DS).

            Parameters
            ----------
            VGS_values : list or array-like
                Gate-source voltage values (V).
            VDS_range : tuple
                (start, end) values of V_DS (V).
            """
            VDS = np.linspace(*VDS_range, 100)
            plt.figure(figsize=(6, 4))

            for VGS in VGS_values:
                ID_sat = self.calculate_id(VGS)
                # Plot a flat line for the saturation region
                ID_plot = np.full_like(VDS, ID_sat)
                plt.plot(VDS, ID_plot * 1e3, label=f"V_GS = {VGS} V")

            plt.xlabel("V_DS (V)")
            plt.ylabel("I_D (mA)")
            plt.title(f"{self.material} JFET Output Characteristics (Sat only)")
            plt.legend()
            plt.grid(True)
            plt.show()

        def plot_transfer_characteristics(self, VGS_range, steps: int = 100):
            """
            Plot the transfer characteristic: drain current (I_D) vs. gate-
            source voltage (V_GS).

            Parameters
            ----------
            VGS_range : tuple
                (start, end) values for V_GS (V). Typically negative for n-JFET.
            steps : int, optional
                Number of points in the V_GS sweep. Default is 100.
            """
            VGS_vals = np.linspace(*VGS_range, steps)
            I_D = np.array([self.calculate_id(vgs) for vgs in VGS_vals])

            plt.figure(figsize=(6, 4))
            plt.plot(VGS_vals, I_D * 1e3, color="green")
            plt.xlabel("V_GS (V)")
            plt.ylabel("I_D (mA)")
            plt.title(f"{self.material} JFET Transfer Characteristic")
            plt.grid(True)
            plt.show()

        def small_signal_transconductance(self, VGS: float) -> float:
            """
            Calculate the small-signal transconductance (g_m) for the JFET.

            From the quadratic model:
                I_D = Idss * (1 - V_GS/Vp)^2
            => g_m = d(I_D)/d(V_GS) 
                   = 2 * Idss/|Vp| * (1 - V_GS/Vp)

            Valid for V_GS between 0 and Vp (negative).
            """
            if VGS >= 0.0 or VGS <= self.Vp:
                return 0.0
            return 2.0 * self.Idss / abs(self.Vp) * (1.0 - (VGS / self.Vp))

        # ---------------------------------------------------------------------
        # Additional / Missing Formulae for JFET
        # ---------------------------------------------------------------------

        def _triode_region_current(self, VGS, VDS):
            """
            Triode (linear) region for JFET:
            I_D ~ (Idss / Vp^2)* [2 (VGS - Vp)*VDS - VDS^2 ],
            valid for 0 <= VDS < |Vp - VGS|.
            (Sign conventions assume n-channel, Vp < 0.)
            """
            return (
                (self.Idss / (self.Vp**2))
                * (2.0 * (VGS - self.Vp) * VDS - VDS**2)
            )

        def _saturation_current(self, VGS):
            """
            JFET saturation current (same as calculate_id, but used internally
            to keep piecewise logic clean).
            """
            return self.Idss * (1.0 - (VGS / self.Vp))**2

        def id_vs_vds_complete(self, VGS_values, VDS_range=(0, 10), steps=100):
            """
            Plot drain current (I_D) vs VDS for multiple VGS, 
            including triode + saturation regions (piecewise).
            """
            VDS_vals = np.linspace(*VDS_range, steps)
            plt.figure(figsize=(6,4))

            for VGS in VGS_values:
                # If VGS >= 0 or VGS <= Vp, expect zero in this simple model
                # but we handle piecewise for demonstration.
                ID_list = []
                for VDS in VDS_vals:
                    # Boundary between triode & saturation:
                    # saturation for VDS >= (|Vp| - |VGS|)
                    # For n-channel, Vp < 0, VGS < 0 => threshold = -(Vp - VGS)
                    threshold = -(self.Vp - VGS)  # e.g. if Vp=-4, VGS=-2 => threshold=2
                    if VGS >= 0 or VGS <= self.Vp:
                        # outside normal conduction range
                        ID_list.append(0.0)
                    else:
                        if VDS < threshold:
                            # Triode region
                            Id_triode = self._triode_region_current(VGS, VDS)
                            # clamp negative if formula overshoots
                            ID_list.append(max(Id_triode, 0.0))
                        else:
                            # Saturation
                            Id_sat = self._saturation_current(VGS)
                            ID_list.append(max(Id_sat, 0.0))

                ID_array = np.array(ID_list)
                plt.plot(VDS_vals, ID_array * 1e3, label=f"V_GS={VGS}V")

            plt.xlabel("V_DS (V)")
            plt.ylabel("I_D (mA)")
            plt.title(f"{self.material} JFET Complete Output (Triode + Saturation)")
            plt.legend()
            plt.grid(True)
            plt.show()
