import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
from math import pi
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

@dataclass
class TemperatureEffects:
    """
    Container for temperature-dependent parameters.
    
    Attributes:
        bandgap (float): Temperature-adjusted bandgap in eV.
        mobility (float): Temperature-adjusted mobility (in cm^2/V·s).
        carrier_concentration (float): Intrinsic carrier concentration (cm^-3).
        resistivity (float): Resistivity (Ω·cm).
        thermal_conductivity (float): Thermal conductivity (W/m·K).
        diffusion_coefficient (float): Diffusion coefficient (cm^2/s).
    """
    bandgap: float
    mobility: float
    carrier_concentration: float
    resistivity: float
    thermal_conductivity: float
    diffusion_coefficient: float


@dataclass
class DiodeConstants:
    """
    Physical constants used in semiconductor calculations.
    
    Attributes:
        BOLTZMANN (float): Boltzmann constant in J/K.
        ELECTRON_CHARGE (float): Elementary charge in Coulombs.
        ROOM_TEMP (float): Reference room temperature in Kelvin.
    """
    BOLTZMANN: float = 1.380649e-23
    ELECTRON_CHARGE: float = 1.602176634e-19
    ROOM_TEMP: float = 300.0


class Diode:
    """
    Represents a diode with configurable material properties, 
    calculation of V-I characteristics, plotting, and basic temperature support.
    """

    BOLTZMANN_CONSTANT = 1.380649e-23  # in J/K
    ELECTRON_CHARGE = 1.602176634e-19  # C

    # Predefined material properties for convenience
    MATERIAL_PROPERTIES = {
        "Silicon": {
            "vt": 0.026,
            "isat": 1e-8,  # Reverse saturation current (A) at 300K
            "ideality_factor": 2,
            "cut_in_voltage": 0.7,
            "breakdown_voltage": 1000,  # Reverse breakdown voltage (V)
            "bandgap_energy": 1.12,      # Bandgap energy (eV)
            "eps": 11.7 * 8.854e-12,     # Permittivity of Silicon (F/m)
            "mobility_300": 1400,        # Mobility at 300K (cm^2/V·s)
        },
        "Germanium": {
            "vt": 0.0258,
            "isat": 1e-6,
            "ideality_factor": 1,
            "cut_in_voltage": 0.3,
            "breakdown_voltage": 300,
            "bandgap_energy": 0.66,
            "eps": 16.0 * 8.854e-12,
            "mobility_300": 3900,
        },
        "Gallium Arsenide": {
            "vt": 0.027,
            "isat": 1e-10,
            "ideality_factor": 1.5,
            "cut_in_voltage": 1.2,
            "breakdown_voltage": 500,
            "bandgap_energy": 1.42,
            "eps": 12.9 * 8.854e-12,
            "mobility_300": 8500,
        },
    }

    def __init__(
        self,
        material: str,
        temperature: float = 300.0,
        custom_props: Dict[str, Any] = None
    ) -> None:
        """
        Initialize the Diode instance with either predefined or custom material properties.

        Args:
            material (str): Material name, e.g., "Silicon", "Germanium", or "Gallium Arsenide".
            temperature (float): Operating temperature in Kelvin (default=300).
            custom_props (dict, optional): User-defined dictionary for custom material properties.
        """
        self.k = Diode.BOLTZMANN_CONSTANT
        self.q = Diode.ELECTRON_CHARGE

        # Validate temperature
        if not (200 <= temperature <= 600):
            raise ValueError(
                f"Temperature must be in the range [200K, 600K]. Got: {temperature}"
            )
        self.temperature = temperature

        # Load material properties
        if custom_props:
            # Custom material
            self.material = "Custom"
            self.eps = custom_props.get("eps", 11.7 * 8.854e-12)
            self.ideality_factor = custom_props.get("ideality_factor", 1)
            self.vt = custom_props.get("vt", 0.026)
            self.isat = custom_props.get("isat", 1e-8)
            self.cut_in_voltage = custom_props.get("cut_in_voltage", 0.7)
            self.breakdown_voltage = custom_props.get("breakdown_voltage", 1000)
            self.bandgap_energy = custom_props.get("bandgap_energy", 1.12)
            self.mobility_300 = custom_props.get("mobility_300", 1400)
        else:
            # Predefined material
            mat = material.strip().title()
            if mat not in self.MATERIAL_PROPERTIES:
                raise ValueError(
                    f"Material '{mat}' not supported. "
                    f"Choose from {list(self.MATERIAL_PROPERTIES.keys())}."
                )
            self.material = mat
            props = self.MATERIAL_PROPERTIES[mat]
            self.eps = props["eps"]
            self.ideality_factor = props["ideality_factor"]
            self.vt = props["vt"]
            self.isat = props["isat"]
            self.cut_in_voltage = props["cut_in_voltage"]
            self.breakdown_voltage = props["breakdown_voltage"]
            self.bandgap_energy = props["bandgap_energy"]
            self.mobility_300 = props["mobility_300"]

    def calculate_saturation_current(self, temperature: float) -> float:
        """
        Calculate temperature-dependent saturation current using an exponential model.

        Args:
            temperature (float): Operating temperature in Kelvin.

        Returns:
            float: The temperature-adjusted saturation current in Amperes.
        """
        # Boltzmann constant in eV/K:
        k_ev = 8.617333262145e-5
        isat_300 = self.isat  # Baseline at 300K
        eg = self.bandgap_energy

        t_ratio = temperature / 300.0
        # Temperature-dependent saturation current:
        isat_t = isat_300 * (
            (t_ratio ** 3) *
            np.exp((-eg / k_ev) * ((1 / temperature) - (1 / 300.0)))
        )
        return isat_t

    def calculate_vi(
        self,
        voltage_range: Tuple[float, float] = (-2, 2),
        steps: int = 1000
    ) -> Dict[str, List[float]]:
        """
        Compute the diode's V-I characteristics over a specified voltage range.

        Args:
            voltage_range (Tuple[float, float]): Start and end voltages (V).
            steps (int): Number of points to compute.

        Returns:
            Dict[str, List[float]]: Dictionary containing 'voltages' and 'currents'.
        """
        # Thermal voltage adjusted for current temperature in eV:
        vt_ev = 8.617333262145*10e-5 * self.temperature
        # Adjusted saturation current
        isat_f = self.calculate_saturation_current(self.temperature)
        isat_r = self.isat  # Reverse saturation current baseline

        voltages = np.linspace(voltage_range[0], voltage_range[1], steps)
        currents = []

        for v in voltages:
            if v >= self.cut_in_voltage:
                # Forward bias above cut-in voltage
                current = isat_f * (
                    np.exp((v - self.cut_in_voltage) /
                           (self.ideality_factor * vt_ev)) - 1
                )
            elif 0 <= v < self.cut_in_voltage:
                # Forward bias below cut-in voltage (minor conduction)
                current = isat_f * (
                    np.exp(v / (self.ideality_factor * vt_ev)) - 1
                ) * 0.01
            elif abs(v) < self.breakdown_voltage:
                # Reverse bias region, no breakdown
                current = -isat_r
            else:
                # Breakdown region
                current = -isat_r * (
                    1 + (abs(v) - self.breakdown_voltage) / 10
                )
            currents.append(current)

        return {"voltages": voltages.tolist(), "currents": currents}

    def plot_vi(
        self,
        voltage_range: Tuple[float, float] = (-2, 2),
        steps: int = 1000,
        log_scale: bool = False
    ) -> None:
        """
        Plot the diode's V-I characteristics as individual points.

        Args:
            voltage_range (Tuple[float, float]): Range of voltages to evaluate.
            steps (int): Number of steps for the voltage range.
            log_scale (bool): If True, plot current on a log scale.
        """
        data = self.calculate_vi(voltage_range, steps)
        voltages = np.array(data["voltages"])
        currents = np.array(data["currents"])

        plt.figure(figsize=(10, 6))
        plt.scatter(voltages, currents, color="blue", s=10,
                    label=f"{self.material} - V-I Points")
        plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
        plt.axvline(0, color="black", linestyle="--", linewidth=0.8)

        # Annotate cut-in voltage
        plt.axvline(self.cut_in_voltage, color="red", linestyle="--",
                    linewidth=1, label=f"Cut-in: {self.cut_in_voltage} V")

        plt.xlabel("Voltage (V)", fontsize=14)
        plt.ylabel("Current (A)", fontsize=14)
        title_str = (f"V-I Characteristics of {self.material} (Dots)"
                     if not log_scale else
                     f"Log-Scale V-I Characteristics of {self.material} (Dots)")
        plt.title(title_str, fontsize=16, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=12)

        if log_scale:
            plt.yscale("log")

        plt.tight_layout()
        plt.show()

    def plot_temperature_effects(
        self,
        voltage_range: Tuple[float, float] = (-1000, 2),
        steps: int = 1000,
        temperature_range: Tuple[float, float] = (250, 400)
    ) -> None:
        """
        Plot the diode's V-I characteristics for multiple temperatures.

        Args:
            voltage_range (Tuple[float, float]): Range of voltages.
            steps (int): Number of voltage points.
            temperature_range (Tuple[float, float]): Range of temperatures in Kelvin.
        """
        temperatures = np.linspace(temperature_range[0], temperature_range[1], 5)
        plt.figure(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(temperatures)))

        for temp, color in zip(temperatures, colors):
            self.temperature = temp
            data = self.calculate_vi(voltage_range, steps)
            plt.plot(data["voltages"], data["currents"],
                     label=f"{temp:.1f} K", linewidth=2, color=color)

        plt.xlabel("Voltage (V)", fontsize=14)
        plt.ylabel("Current (A)", fontsize=14)
        plt.title(f"Temperature Effects on V-I Characteristics ({self.material})",
                  fontsize=16, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(title="Temp (K)", fontsize=12)
        plt.tight_layout()
        plt.show()

    def export_to_csv(
        self,
        data: Dict[str, List[float]],
        filename: str = "diode_vi_data.csv"
    ) -> None:
        """
        Export computed V-I data to a CSV file.

        Args:
            data (dict): Dictionary containing 'voltages' and 'currents'.
            filename (str): Target CSV filename.
        """
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Voltage (V)", "Current (A)"])
            for v, i in zip(data["voltages"], data["currents"]):
                writer.writerow([v, i])
        print(f"Data exported to {filename}")

    def log_result(self, message: str) -> None:
        """
        Log results with timestamps to a local file.

        Args:
            message (str): Log message.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("diode_results.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"[{timestamp}] {message}\n")

    def validate_material_properties(self) -> None:
        """
        Validate material properties against typical ranges and print warnings if needed.
        """
        warnings = []
        if not (0.01e-9 <= self.isat <= 1e-6):
            warnings.append(f"Warning: Unusual reverse saturation current: {self.isat}")
        if not (0.02 <= self.vt <= 0.03):
            warnings.append(f"Warning: Unusual thermal voltage: {self.vt}")
        if not (0.6 <= self.bandgap_energy <= 1.5):
            warnings.append(f"Warning: Unusual bandgap energy: {self.bandgap_energy}")
        if not (100 <= self.breakdown_voltage <= 2000):
            warnings.append(
                f"Warning: Unusual breakdown voltage: {self.breakdown_voltage}"
            )

        if warnings:
            for warn in warnings:
                print(warn)
        else:
            print("All material properties are within typical ranges.")

    def plot_material_comparison(self) -> None:
        """
        Compare predefined material properties in a spider chart.
        """
        materials = list(self.MATERIAL_PROPERTIES.keys())
        categories = [
            "V_t", "I_s", "E_g", "Breakdown Voltage", "Cut-in Voltage", "Ideality Factor"
        ]

        data = []
        for mat in materials:
            props = self.MATERIAL_PROPERTIES[mat]
            data.append([
                props["vt"],
                props["isat"],
                props["bandgap_energy"],
                props["breakdown_voltage"],
                props["cut_in_voltage"],
                props["ideality_factor"],
            ])

        # Normalize data for plotting on spider chart
        max_vals = [max([d[i] for d in data]) for i in range(len(categories))]
        data_norm = [[val / mx for val, mx in zip(d, max_vals)] for d in data]

        # Angles for each category
        num_vars = len(categories)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]  # close the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        for i, mat_data in enumerate(data_norm):
            values = mat_data + mat_data[:1]
            ax.plot(angles, values, linewidth=2, label=materials[i])
            ax.fill(angles, values, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.title("Extended Material Property Comparison", fontsize=16)
        plt.show()

    def diffusion_current(self, dn_dx: float, diffusion_coeff: float = 0.01) -> float:
        """
        Calculate diffusion current given a carrier gradient.

        Args:
            dn_dx (float): Carrier concentration gradient.
            diffusion_coeff (float): Diffusion coefficient (m^2/s) default=0.01.

        Returns:
            float: Diffusion current in Amperes.
        """
        return self.q * diffusion_coeff * dn_dx

    def drift_current(self, electric_field: float, mobility: float = 1400) -> float:
        """
        Calculate drift current given an electric field and mobility.

        Args:
            electric_field (float): Electric field strength (V/m).
            mobility (float): Carrier mobility (cm^2/V·s) or (m^2/V·s, depending on units).

        Returns:
            float: Drift current in Amperes.
        """
        return self.q * mobility * electric_field

    def junction_capacitance(self, area: float = 1e-6, width: float = 1e-6) -> float:
        """
        Calculate junction capacitance using a parallel-plate approximation.

        Args:
            area (float): Junction area in m^2.
            width (float): Depletion region width in m.

        Returns:
            float: Junction capacitance in Farads.
        """
        return self.eps * area / width

    def thermal_noise(self, resistance: float, bandwidth: float = 1e6) -> float:
        """
        Calculate thermal noise (Johnson-Nyquist).

        Args:
            resistance (float): Resistance in Ohms.
            bandwidth (float): Bandwidth in Hz.

        Returns:
            float: RMS noise voltage in Volts.
        """
        return np.sqrt(4 * self.k * self.temperature * resistance * bandwidth)

    def shot_noise(self, dc_current: float, bandwidth: float = 1e6) -> float:
        """
        Calculate shot noise for a given DC current and bandwidth.

        Args:
            dc_current (float): DC current in Amperes.
            bandwidth (float): Bandwidth in Hz.

        Returns:
            float: RMS noise current in Amperes.
        """
        return np.sqrt(2 * self.q * dc_current * bandwidth)

    def plot_noise_vs_temperature(self) -> None:
        """
        Plot thermal noise voltage vs. temperature for a fixed resistance.
        """
        temperatures = np.linspace(200, 600, 100)
        noise_levels = []

        # Example: fix R = 1000 Ohms, BW = 1e6
        resistance = 1000
        bandwidth = 1e6

        for temp in temperatures:
            # Temporarily store the original temperature
            original_temp = self.temperature
            self.temperature = temp
            noise_levels.append(self.thermal_noise(resistance, bandwidth))
            # Restore original temperature
            self.temperature = original_temp

        plt.figure(figsize=(8, 5))
        plt.plot(temperatures, noise_levels, label="Thermal Noise")
        plt.xlabel("Temperature (K)")
        plt.ylabel("Noise Voltage (V)")
        plt.title("Thermal Noise vs Temperature")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_v_with_power(
        self,
        voltage_range: Tuple[float, float] = (-1000, 2),
        steps: int = 1000
    ) -> None:
        """
        Plot V-I characteristics along with power dissipation (P=V*I).

        Args:
            voltage_range (Tuple[float, float]): Range of voltages for evaluation.
            steps (int): Number of points in the sweep.
        """
        data = self.calculate_vi(voltage_range, steps)
        voltages = np.array(data["voltages"])
        currents = np.array(data["currents"])
        power = voltages * currents

        plt.figure(figsize=(12, 6))
        plt.plot(voltages, power, label="Power (W)", linestyle="--",
                 color="green", linewidth=2)
        plt.xlabel("Voltage (V)", fontsize=14)
        plt.ylabel("Current (A) & Power (W)", fontsize=14)
        plt.title(f"V-I and Power Dissipation of {self.material}",
                  fontsize=16, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
        plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_conductance_vs_voltage(
        diode,
        voltage_range: Tuple[float, float] = (-2, 2),
        steps: int = 1000
    ) -> None:
        """
        Plot differential conductance (dI/dV) vs. voltage for the given diode.

        Args:
            diode (Diode): Diode instance.
            voltage_range (Tuple[float, float]): Voltage sweep range.
            steps (int): Number of points in the sweep.
        """
        data = diode.calculate_vi(voltage_range, steps)
        voltages = np.array(data["voltages"])
        currents = np.array(data["currents"])
        conductance = np.gradient(currents, voltages)  # numerical derivative

        plt.figure(figsize=(10, 5))
        plt.plot(voltages, conductance, label="Conductance (S)", color="red")
        plt.xlabel("Voltage (V)")
        plt.ylabel("Conductance (Siemens)")
        plt.title(f"Differential Conductance vs. Voltage ({diode.material})")
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_bandgap_vs_temperature(
        diode,
        temperature_range: Tuple[float, float] = (200, 600)
    ) -> None:
        """
        Plot a simple bandgap vs. temperature trend (linear approximation).

        Args:
            diode (Diode): Diode instance for reference bandgap.
            temperature_range (Tuple[float, float]): Range of temperatures in Kelvin.
        """
        temperatures = np.linspace(temperature_range[0], temperature_range[1], 100)
        # Simple linear approximation for demonstration
        bandgaps = [
            diode.bandgap_energy - (0.0007 * (temp - 300))
            for temp in temperatures
        ]

        plt.figure(figsize=(8, 5))
        plt.plot(temperatures, bandgaps, label="Bandgap Energy (eV)", color="purple")
        plt.xlabel("Temperature (K)")
        plt.ylabel("Bandgap Energy (eV)")
        plt.title(f"Bandgap Energy vs Temperature ({diode.material})")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_reverse_breakdown(
        self,
        voltage_range: Tuple[float, float] = None,
        steps: int = 500
    ) -> None:
        """
        Plot the diode's reverse breakdown region.

        Args:
            voltage_range (Tuple[float, float], optional): Voltage range for reverse bias.
            steps (int): Number of points.
        """
        if voltage_range is None:
            voltage_range = (-self.breakdown_voltage - 50, -self.breakdown_voltage + 10)

        data = self.calculate_vi(voltage_range, steps)
        plt.figure(figsize=(8, 5))
        plt.plot(data["voltages"], data["currents"],
                 label="Reverse Breakdown", color="brown")
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        plt.title(f"Reverse Breakdown Region ({self.material})")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_junction_capacitance_vs_voltage(
        self,
        voltage_range: Tuple[float, float] = None,
        steps: int = 500
    ) -> None:
        """
        Plot junction capacitance vs. reverse voltage using a simple approximation.

        Args:
            voltage_range (Tuple[float, float], optional): Voltage range in reverse bias.
            steps (int): Number of voltage steps.
        """
        if voltage_range is None:
            voltage_range = (-self.breakdown_voltage, 0)

        voltages = np.linspace(voltage_range[0], voltage_range[1], steps)
        # Simple 1/sqrt(1 - V/Vbr) approximation for demonstration
        c0 = self.junction_capacitance(area=1e-6, width=1e-6)
        capacitances = c0 / np.sqrt(1 - (voltages / self.breakdown_voltage))

        plt.figure(figsize=(8, 5))
        plt.plot(voltages, capacitances, label="Junction Capacitance", color="cyan")
        plt.xlabel("Reverse Voltage (V)")
        plt.ylabel("Capacitance (F)")
        plt.title(f"Junction Capacitance vs Reverse Voltage ({self.material})")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_drift_diffusion_currents(self) -> None:
        """
        Plot exemplary drift and diffusion current curves vs. a range of field/gradient.
        """
        electric_fields = np.linspace(0, 1e5, 100)
        diffusion_gradients = np.linspace(0, 1e6, 100)

        drift_currents = [self.drift_current(ef) for ef in electric_fields]
        diffusion_currents = [self.diffusion_current(dn)
                              for dn in diffusion_gradients]

        plt.figure(figsize=(10, 5))
        plt.plot(electric_fields, drift_currents, label="Drift Current", color="blue")
        plt.plot(diffusion_gradients, diffusion_currents,
                 label="Diffusion Current", color="green")
        plt.xlabel("Field Strength / Carrier Gradient")
        plt.ylabel("Current (A)")
        plt.title(f"Drift & Diffusion Currents ({self.material})")
        plt.grid(True)
        plt.legend()
        plt.show()

    def __repr__(self) -> str:
        """
        String representation of the Diode object.
        """
        return (f"Diode(material={self.material}, "
                f"ideality_factor={self.ideality_factor}, "
                f"temperature={self.temperature} K)")


class DiodeTemperature(Diode):
    """
    Advanced diode model extending Diode with more comprehensive 
    temperature-dependent semiconductor physics.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize DiodeTemperature using the base Diode class constructor 
        and additional temperature effect calculations.
        """
        super().__init__(*args, **kwargs)

        # Diode must have mobility_300 from base class
        if not hasattr(self, "mobility_300"):
            raise ValueError(f"'mobility_300' is missing for material '{self.material}'.")

        # Constants and temperature effects container
        self.constants = DiodeConstants()
        self.temp_effects = self.calculate_temperature_effects()

    def get_varshni_parameters(self) -> Dict[str, float]:
        """
        Return Varshni equation parameters for the material's bandgap variation.
        """
        varshni_params = {
            "Silicon": {"alpha": 4.73e-4, "beta": 636},   # eV/K, K
            "Germanium": {"alpha": 4.77e-4, "beta": 235},
            "Gallium Arsenide": {"alpha": 5.41e-4, "beta": 204},
        }
        if self.material not in varshni_params:
            raise ValueError(f"Varshni params not defined for '{self.material}'.")
        return varshni_params[self.material]

    def get_effective_mass_ratio(self) -> float:
        """
        Approximate effective mass ratio (m*/m0) for the given semiconductor.
        """
        effective_mass_ratios = {
            "Silicon": 1.08,
            "Germanium": 0.56,
            "Gallium Arsenide": 0.067,
        }
        if self.material not in effective_mass_ratios:
            raise ValueError(f"Effective mass ratio not defined for '{self.material}'.")
        return effective_mass_ratios[self.material]

    def get_thermal_conductivity_300k(self) -> float:
        """
        Return thermal conductivity at 300K for the given material.
        """
        thermal_conductivities = {
            "Silicon": 148,
            "Germanium": 60,
            "Gallium Arsenide": 55,
        }
        if self.material not in thermal_conductivities:
            raise ValueError(f"Thermal conductivity not defined for '{self.material}'.")
        return thermal_conductivities[self.material]

    def calculate_temperature_effects(self) -> TemperatureEffects:
        """
        Calculate all temperature-dependent parameters and return a TemperatureEffects dataclass.

        Returns:
            TemperatureEffects: Dataclass with temperature-adjusted properties.
        """
        bandgap = self.calculate_temperature_bandgap()
        mobility = self.calculate_temperature_mobility()
        carrier_conc = self.calculate_carrier_concentration(bandgap)
        resistivity = self.calculate_resistivity(mobility, carrier_conc)
        thermal_cond = self.calculate_thermal_conductivity()
        diffusion_coeff = self.calculate_diffusion_coefficient(mobility)

        return TemperatureEffects(
            bandgap=bandgap,
            mobility=mobility,
            carrier_concentration=carrier_conc,
            resistivity=resistivity,
            thermal_conductivity=thermal_cond,
            diffusion_coefficient=diffusion_coeff,
        )

    def calculate_temperature_bandgap(self) -> float:
        """
        Calculate the bandgap at the current temperature using the Varshni equation.

        Returns:
            float: Temperature-adjusted bandgap in eV.
        """
        params = self.get_varshni_parameters()
        alpha, beta = params['alpha'], params['beta']

        # Varshni equation with optional strain factor
        strain_factor = 1.0
        return (self.bandgap_energy * strain_factor
                - (alpha * self.temperature**2) / (self.temperature + beta))

    def calculate_temperature_mobility(self) -> float:
        """
        Calculate mobility at the current temperature (advanced model).

        Returns:
            float: Temperature-adjusted mobility in cm^2/V·s.
        """
        # Example scattering exponents
        phonon_scattering = (self.temperature / 300) ** (-2.42)
        ionized_impurity = (self.temperature / 300) ** (1.5)

        # Combine scattering mechanisms (example: Matthiessen's rule–like approach)
        combined_factor = 1.0 / (1.0/phonon_scattering + 1.0/ionized_impurity)
        return self.mobility_300 * combined_factor

    def calculate_carrier_concentration(self, bandgap: float) -> float:
        """
        Calculate intrinsic carrier concentration using bandgap and effective mass ratio.

        Args:
            bandgap (float): Temperature-adjusted bandgap in eV.

        Returns:
            float: Intrinsic carrier concentration in cm^-3.
        """
        k_ev = 8.617333262145e-5  # Boltzmann constant in eV/K
        effective_mass_ratio = self.get_effective_mass_ratio()
        # Prefactor for Nc * Nv (rough estimate)
        nc_nv_300 = 2.5e19  # typical value at 300 K
        nc_nv = nc_nv_300 * (self.temperature / 300) ** 3 * effective_mass_ratio

        return nc_nv * np.exp(-bandgap / (2 * k_ev * self.temperature))

    def calculate_resistivity(self, mobility: float, carrier_conc: float) -> float:
        """
        Calculate resistivity based on mobility and carrier concentration.

        Args:
            mobility (float): Mobility in cm^2/V·s.
            carrier_conc (float): Carrier concentration in cm^-3.

        Returns:
            float: Resistivity in Ω·cm.
        """
        # Convert mobility to cm^2/(V·s), carrier_conc is in cm^-3
        q_cgs = 1.602176634e-19  # coulomb
        # σ = q * n * μ (in 1/Ω·cm)
        conductivity = q_cgs * carrier_conc * mobility
        return 1.0 / conductivity if conductivity != 0 else 1e99

    def calculate_thermal_conductivity(self) -> float:
        """
        Approximate temperature-dependent thermal conductivity.

        Returns:
            float: Thermal conductivity (W/m·K).
        """
        k300 = self.get_thermal_conductivity_300k()
        return k300 * (300 / self.temperature) ** 1.5

    def calculate_diffusion_coefficient(self, mobility: float) -> float:
        """
        Calculate diffusion coefficient using Einstein's relation.

        Args:
            mobility (float): Mobility in cm^2/V·s.

        Returns:
            float: Diffusion coefficient (cm^2/s).
        """
        # Convert mobility to cm^2/(V·s), constants in cgs
        k = Diode.BOLTZMANN_CONSTANT
        T = self.temperature
        q = Diode.ELECTRON_CHARGE

        # Einstein relation: D = (kB * T / q) * (μ)
        # We keep units consistent with cm^2/s
        d_cgs = (k * T / q) * mobility  # J/K * K / C * cm^2/(V·s) => ??? 
        # Because 1 J = 1 (C·V), you get cm^2/s if mobility is in cm^2/(V·s)
        return d_cgs

    def get_thermal_resistance(self, area: float, thickness: float) -> float:
        """
        Calculate device thermal resistance (K/W).

        Args:
            area (float): Junction area in m^2.
            thickness (float): Device thickness in m.

        Returns:
            float: Thermal resistance in K/W.
        """
        thermal_cond = self.get_thermal_conductivity_300k()  # baseline
        return thickness / (thermal_cond * area)

    def calculate_junction_temperature(
        self,
        ambient_temp: float,
        power_dissipation: float,
        thermal_resistance: float
    ) -> float:
        """
        Calculate junction temperature given power and thermal resistance.

        Args:
            ambient_temp (float): Ambient temperature in K.
            power_dissipation (float): Power in W.
            thermal_resistance (float): Thermal resistance in K/W.

        Returns:
            float: Junction temperature in Kelvin.
        """
        return ambient_temp + (power_dissipation * thermal_resistance)

    def calculate_leakage_current(self) -> float:
        """
        Approximate leakage current based on reverse saturation current.

        Returns:
            float: Leakage current in Amperes.
        """
        return self.calculate_saturation_current(self.temperature)

    def calculate_breakdown_voltage_temp(self) -> float:
        """
        Simple approximation for temperature-dependent breakdown voltage.

        Returns:
            float: Adjusted breakdown voltage in Volts.
        """
        # Negative temperature coefficient example: -1 mV/K
        temp_coeff = -0.001
        return self.breakdown_voltage + (self.temperature - 300) * temp_coeff

    def calculate_lifetime_factor(self) -> float:
        """
        Estimate lifetime factor based on Arrhenius-type model.

        Returns:
            float: Relative lifetime factor (dimensionless).
        """
        activation_energy = 0.7  # eV, example
        k_ev = 8.617333262145e-5
        return np.exp(-activation_energy / (k_ev * self.temperature))

    def plot_comprehensive_temperature_effects(
        self,
        temp_range: Tuple[float, float] = (250, 450)
    ) -> None:
        """
        Plot multiple temperature-dependent properties across a given temperature range.

        Args:
            temp_range (Tuple[float, float]): Start and end temperatures in Kelvin.
        """
        temperatures = np.linspace(temp_range[0], temp_range[1], 100)

        # Prepare data containers
        params = {
            "Bandgap (eV)": [],
            "Mobility (cm^2/V·s)": [],
            "Carrier Conc. (cm^-3)": [],
            "Resistivity (Ω·cm)": [],
            "Thermal Cond. (W/m·K)": [],
            "Diffusion Coeff. (cm^2/s)": []
        }

        for temp in temperatures:
            # Temporarily override diode temperature
            self.temperature = temp
            effects = self.calculate_temperature_effects()

            params["Bandgap (eV)"].append(effects.bandgap)
            params["Mobility (cm^2/V·s)"].append(effects.mobility)
            params["Carrier Conc. (cm^-3)"].append(effects.carrier_concentration)
            params["Resistivity (Ω·cm)"].append(effects.resistivity)
            params["Thermal Cond. (W/m·K)"].append(effects.thermal_conductivity)
            params["Diffusion Coeff. (cm^2/s)"].append(effects.diffusion_coefficient)

        # Restore diode temperature to original or last used
        # (not strictly necessary, but good practice)

        # Create subplots (2x3)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Temperature Effects in {self.material} Diode", fontsize=16)

        for (param_label, values), ax in zip(params.items(), axes.flat):
            ax.plot(temperatures, values, "b-")
            ax.set_xlabel("Temperature (K)")
            ax.set_ylabel(param_label)
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_vi_temperature_family(
        self,
        temp_range: List[float],
        voltage_range: Tuple[float, float] = (-2, 2),
        steps: int = 1000
    ) -> None:
        """
        Plot a family of V-I curves at different temperatures.

        Args:
            temp_range (List[float]): List of temperatures to plot in Kelvin.
            voltage_range (Tuple[float, float]): Voltage sweep range.
            steps (int): Number of points for the sweep.
        """
        plt.figure(figsize=(10, 6))

        for temp in temp_range:
            self.temperature = temp
            data = self.calculate_vi(voltage_range, steps)
            plt.semilogy(data["voltages"], np.abs(data["currents"]),
                         label=f"T = {temp} K")

        plt.grid(True)
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        plt.title(f"Temperature Dependence of {self.material} Diode V-I")
        plt.legend()
        plt.show()

    def plot_power_dissipation_effects(
        self,
        voltage_range: Tuple[float, float] = (-2, 2),
        ambient_temp: float = 300,
        steps: int = 1000
    ) -> None:
        """
        Visualize how power dissipation affects junction temperature.

        Args:
            voltage_range (Tuple[float, float]): Range of voltages.
            ambient_temp (float): Ambient temperature in Kelvin.
            steps (int): Number of points.
        """
        voltages = np.linspace(voltage_range[0], voltage_range[1], steps)
        vi_data = self.calculate_vi(voltage_range, steps)
        currents = np.array(vi_data["currents"])
        power = voltages * currents

        # Estimate thermal resistance (example area=1e-6, thickness=1e-4)
        th_res = self.get_thermal_resistance(1e-6, 1e-4)
        junction_temps = [
            self.calculate_junction_temperature(ambient_temp, p, th_res)
            for p in power
        ]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(voltages, power * 1e3, "r-")  # mW
        ax1.set_xlabel("Voltage (V)")
        ax1.set_ylabel("Power Dissipation (mW)")
        ax1.grid(True)

        ax2.plot(voltages, junction_temps, "b-")
        ax2.set_xlabel("Voltage (V)")
        ax2.set_ylabel("Junction Temperature (K)")
        ax2.grid(True)

        plt.suptitle("Power Dissipation and Junction Temperature Effects")
        plt.tight_layout()
        plt.show()

    def plot_temperature_reliability_indicators(
        self,
        temp_range: Tuple[float, float] = (250, 450)
    ) -> None:
        """
        Plot reliability indicators (leakage current, breakdown voltage, lifetime factor)
        as functions of temperature.

        Args:
            temp_range (Tuple[float, float]): Start and end temperatures in Kelvin.
        """
        temperatures = np.linspace(temp_range[0], temp_range[1], 100)

        leakage_current = []
        breakdown_voltage = []
        lifetime_factor = []

        for temp in temperatures:
            self.temperature = temp
            leakage_current.append(self.calculate_leakage_current())
            breakdown_voltage.append(self.calculate_breakdown_voltage_temp())
            lifetime_factor.append(self.calculate_lifetime_factor())

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.semilogy(temperatures, leakage_current)
        ax1.set_xlabel("Temperature (K)")
        ax1.set_ylabel("Leakage Current (A)")
        ax1.grid(True)

        ax2.plot(temperatures, breakdown_voltage)
        ax2.set_xlabel("Temperature (K)")
        ax2.set_ylabel("Breakdown Voltage (V)")
        ax2.grid(True)

        ax3.plot(temperatures, lifetime_factor)
        ax3.set_xlabel("Temperature (K)")
        ax3.set_ylabel("Relative Lifetime")
        ax3.grid(True)

        plt.suptitle("Temperature-Dependent Reliability Indicators")
        plt.tight_layout()
        plt.show()
