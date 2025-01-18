import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
from math import pi
import csv
from dataclasses import dataclass
from typing import Dict, Any



class Diode:
    BOLTZMANN_CONSTANT = 1.380649e-23  # in J/K
    ELECTRON_CHARGE = 1.602176634e-19  # C
    MATERIAL_PROPERTIES = {
        "Silicon": {
            "vt": 0.026,
            "isat": 10e-9,  # Reverse saturation current (A) at 300K
            "ideality_factor": 2,
            "cut_in_voltage": 0.7,
            "breakdown_voltage": 1000,  # Reverse breakdown voltage (V)
            "bandgap_energy": 1.12,  # Bandgap energy (eV)
            "eps": 11.7 * 8.854e-12,  # Permittivity of Silicon (F/m)
            "mobility_300": 1400,  # cm²/V·s
        },
        "Germanium": {
            "vt": 0.0258,
            "isat": 1e-6,  # Reverse saturation current (A) at 300K
            "ideality_factor": 1,
            "cut_in_voltage": 0.3,
            "breakdown_voltage": 300,  # Reverse breakdown voltage (V)
            "bandgap_energy": 0.66,  # Bandgap energy (eV)
            "eps": 16.0 * 8.854e-12,  # Permittivity of Germanium (F/m)
            "mobility_300": 3900,  # cm²/V·s

        },
        "Gallium Arsenide": {
            "vt": 0.027,  # Thermal voltage at 300 K
            "isat": 1e-10,  # Very small saturation current due to high bandgap
            "ideality_factor": 1.5,
            "cut_in_voltage": 1.2,
            "breakdown_voltage": 500,  # Reverse breakdown voltage (V)
            "bandgap_energy": 1.42,  # Bandgap energy (eV)
            "eps": 12.9 * 8.854e-12,  # Permittivity of GaAs (F/m)
            "mobility_300": 8500,  # cm²/V·s
        },
    }

    def __init__(self, material, temperature=300, custom_props=None):
        """Initialize the Diode class with material properties or user-defined properties."""
        self.k = 1.380649e-23  # Boltzmann Constant (J/K)
        self.q = 1.602176634e-19  # Elementary charge (C)
        
        if custom_props:
            # Custom material properties
            self.material = "Custom"
            self.eps = custom_props.get("eps", 11.7 * 8.854e-12)  # Default to Silicon if not provided
        else:
            # Predefined material properties
            self.material = material.strip().title()
            if self.material not in self.MATERIAL_PROPERTIES:
                raise ValueError(f"Material '{self.material}' not supported. Choose from {list(self.MATERIAL_PROPERTIES.keys())}.")
            props = self.MATERIAL_PROPERTIES[self.material]
            
            self.eps = props["eps"]  # ✅ Assign permittivity correctly
            
        # Validate temperature to ensure it is in Kelvin
        if not (200 <= temperature <= 600):
            raise ValueError(f"Temperature must be in Kelvin (200 K to 600 K). Given: {temperature} K")

        if custom_props:
            # Use user-defined material properties
            self.material = "Custom"
            self.ideality_factor = custom_props.get("ideality_factor", 1)
            self.vt = custom_props.get("vt", 0.026)
            self.isat = custom_props.get("isat", 10e-9)
            self.cut_in_voltage = custom_props.get("cut_in_voltage", 0.7)
            self.breakdown_voltage = custom_props.get("breakdown_voltage", 1000)
            self.bandgap_energy = custom_props.get("bandgap_energy", 1.12)
        else:
            # Use predefined material properties
            self.material = material.strip().title()
            if self.material not in self.MATERIAL_PROPERTIES:
                raise ValueError(f"Material '{self.material}' not supported. Choose from {list(self.MATERIAL_PROPERTIES.keys())}.")
            props = self.MATERIAL_PROPERTIES[self.material]
            self.ideality_factor = props["ideality_factor"]
            self.vt = props["vt"]
            self.isat = props["isat"]
            print(self.isat)
            self.cut_in_voltage = props["cut_in_voltage"]
            self.breakdown_voltage = props["breakdown_voltage"]
            self.bandgap_energy = props["bandgap_energy"]

        self.temperature = temperature

    def calculate_saturation_current(self, temperature):
        """Calculate temperature-dependent saturation current."""
        k = 8.617333262145e-5   # Boltzmann constant in eV/K
        isat_300 = self.isat  # Reverse saturation current at 300 K
        eg = self.bandgap_energy
        t_ratio = self.temperature / 300
        isat_t = isat_300 * ((t_ratio**3) * np.exp((-eg / k) * ((1 / temperature) - (1 / 300))))
        print(isat_t)
        return isat_t

    def calculate_vi(self, voltage_range=(-2, 2), steps=1000):
        """Calculate realistic V-I characteristics with accurate knee voltage behavior."""
        vt = 8.617333262145*10e-5 * (self.temperature)  # Adjusted thermal voltage
        isat_f = self.calculate_saturation_current(self.temperature)  # Temperature-dependent saturation current
        isat_r = self.isat  # Reverse saturation current remains constant

        voltages = np.linspace(voltage_range[0], voltage_range[1], steps)
        currents = []

        for v in voltages:
            if v >= self.cut_in_voltage:  # Forward bias: Above cut-in voltage
                # Apply exponential model beyond the cut-in voltage threshold
                current = isat_f * ((np.exp((v - self.cut_in_voltage) / (self.ideality_factor * vt)) - 1))
            elif 0 <= v < self.cut_in_voltage:  # Forward bias: Below cut-in voltage
                current = isat_f * (np.exp(v / (self.ideality_factor * vt)) - 1) * 0.01  # Small leakage
            elif abs(v) < self.breakdown_voltage:  # Reverse bias: No breakdown
                current = -isat_r
            else:  # Reverse bias: Breakdown region
                current = -isat_r * (1 + (abs(v) - self.breakdown_voltage) / 10)  # Gradual breakdown rise
            currents.append(current)
            print(currents)

        return {"voltages": voltages, "currents": currents}


    def log_result(self, message):
        """Log results with timestamps to a file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("diode_results.log", "a") as log_file:
            log_file.write(f"[{timestamp}] {message}\n")

    def plot_vi(self, voltage_range=(-2, 2), steps=1000, log_scale=False):
        """Plot V-I characteristics as individual points for each voltage-current pair."""
        data = self.calculate_vi(voltage_range, steps)
        voltages = data["voltages"]
        currents = data["currents"]

        plt.figure(figsize=(10, 6))
        plt.scatter(voltages, currents, color="blue", s=10, label=f"{self.material} - V-I Points")  # Plot points as dots
        plt.axhline(0, color="black", linestyle="--", linewidth=0.8)  # X-axis
        plt.axvline(0, color="black", linestyle="--", linewidth=0.8)  # Y-axis

        # Annotate Cut-in Voltage
        plt.axvline(self.cut_in_voltage, color="red", linestyle="--", linewidth=1, label=f"Cut-in Voltage: {self.cut_in_voltage} V")

        # Plot configuration
        plt.xlabel("Voltage (V)", fontsize=14)
        plt.ylabel("Current (A)", fontsize=14)
        plt.title(f"V-I Characteristics of {self.material} (Dots)", fontsize=16, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()

        # Optional: Log-scale for current
        if log_scale:
            plt.yscale("log")
            plt.title(f"Log-Scale V-I Characteristics of {self.material} (Dots)", fontsize=16, fontweight="bold")

        plt.show()

    def plot_temperature_effects(self, voltage_range=(-1000, 2), steps=1000, temperature_range=(250, 400)):
        """Plot V-I characteristics for multiple temperatures."""
        temperatures = np.linspace(temperature_range[0], temperature_range[1], 5)
        plt.figure(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(temperatures)))

        for temp, color in zip(temperatures, colors):
            self.temperature = temp
            data = self.calculate_vi(voltage_range, steps)
            plt.plot(data["voltages"], data["currents"], label=f"{temp} K", linewidth=2, color=color)

        plt.xlabel("Voltage (V)", fontsize=14)
        plt.ylabel("Current (A)", fontsize=14)
        plt.title(f"Temperature Effects on V-I Characteristics ({self.material})", fontsize=16, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(title="Temperature (K)", fontsize=12)
        plt.tight_layout()
        plt.show()
        self.log_result(f"Plotted temperature effects: voltage_range={voltage_range}, steps={steps}, temperature_range={temperature_range}")

    def export_to_csv(self, data, filename="diode_vi_data.csv"):
        """Export V-I data to a CSV file."""
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Voltage (V)", "Current (A)"])
            for v, i in zip(data["voltages"], data["currents"]):
                writer.writerow([v, i])
        self.log_result(f"Exported V-I data to {filename}")
        print(f"Data exported to {filename}")

    def validate_material_properties(self):
        """Validate material properties against typical values."""
        warnings = []
        if not (0.01e-9 <= self.isat <= 1e-6):
            warnings.append(f"Warning: Unusual reverse saturation current (I_s): {self.isat}")
        if not (0.02 <= self.vt <= 0.03):
            warnings.append(f"Warning: Unusual thermal voltage (V_t): {self.vt}")
        if not (0.6 <= self.bandgap_energy <= 1.5):
            warnings.append(f"Warning: Unusual bandgap energy (E_g): {self.bandgap_energy}")
        if not (100 <= self.breakdown_voltage <= 2000):
            warnings.append(f"Warning: Unusual breakdown voltage: {self.breakdown_voltage}")
        
        if warnings:
            for warning in warnings:
                print(warning)
        else:
            print("All material properties are within typical ranges.")
    def animate_vi(self, voltage_range=(-1000, 2), steps=1000, temperature_range=(250, 400), interval=500):
        """Animate V-I characteristics across a temperature range."""
        fig, ax = plt.subplots(figsize=(12, 6))
        line, = ax.plot([], [], lw=2)
        ax.set_xlim(voltage_range[0], voltage_range[1])
        ax.set_ylim(-1e-12, 1e-3)  # Adjusted for reverse and forward current scales
        ax.set_xlabel("Voltage (V)", fontsize=14)
        ax.set_ylabel("Current (A)", fontsize=14)
        ax.set_title("Temperature-dependent V-I Characteristics", fontsize=16)

        temperatures = np.linspace(temperature_range[0], temperature_range[1], 50)

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            self.temperature = temperatures[frame]
            data = self.calculate_vi(voltage_range, steps)
            line.set_data(data["voltages"], data["currents"])
            ax.set_title(f"V-I Characteristics at {self.temperature:.1f} K", fontsize=14)
            return line,

        ani = FuncAnimation(fig, update, frames=len(temperatures), init_func=init, blit=True, interval=interval)
        self.log_result(f"Generated V-I animation for temperature range {temperature_range}")
        plt.show()

    def plot_material_comparison(self):
        """Compare material properties using a spider chart."""
        materials = list(self.MATERIAL_PROPERTIES.keys())
        categories = ["V_t", "I_s", "E_g", "Breakdown Voltage", "Cut-in Voltage", "Ideality Factor"]
        data = []

        for material in materials:
            props = self.MATERIAL_PROPERTIES[material]
            data.append([
                props["vt"],
                props["isat"],
                props["bandgap_energy"],
                props["breakdown_voltage"],
                props["cut_in_voltage"],
                props["ideality_factor"],
            ])

        # Normalize data for plotting
        max_values = [max([d[i] for d in data]) for i in range(len(categories))]
        data_normalized = [[d[i] / max_values[i] for i in range(len(categories))] for d in data]

        # Create spider chart
        num_vars = len(categories)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        for i, material_data in enumerate(data_normalized):
            values = material_data + material_data[:1]
            ax.plot(angles, values, label=materials[i], linewidth=2)
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.title("Extended Material Property Comparison", fontsize=16)
        plt.show()
        
    def diffusion_current(self, dn_dx, diffusion_coeff=0.01):
        return self.q * diffusion_coeff * dn_dx

    def drift_current(self, electric_field, mobility=1400):
        return self.q * mobility * electric_field

    def junction_capacitance(self, area=1e-6, width=1e-6):
        return self.eps * area / width

    def breakdown_voltage(self, doping_concentration):
        return 1 / (self.q * doping_concentration) * (2 * self.eps * self.q * doping_concentration ** 2)
    
    def thermal_noise(self, resistance, bandwidth=1e6):
        """Calculate thermal noise voltage (Johnson-Nyquist noise)."""
        return np.sqrt(4 * self.k * self.temperature * resistance * bandwidth)

    def shot_noise(self, dc_current, bandwidth=1e6):
        return np.sqrt(2 * self.q * dc_current * bandwidth)

    def plot_noise_vs_temperature(self):
        temperatures = np.linspace(200, 600, 100)
        noise_levels = [self.thermal_noise(1000, 1e6) for temp in temperatures]

        plt.figure(figsize=(8, 5))
        plt.plot(temperatures, noise_levels, label="Thermal Noise")
        plt.xlabel("Temperature (K)")
        plt.ylabel("Noise Voltage (V)")
        plt.title("Thermal Noise vs Temperature")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_vi_with_power(self, voltage_range=(-1000, 2), steps=1000):
        """Plot V-I characteristics with power dissipation."""
        data = self.calculate_vi(voltage_range, steps)
        voltages = data["voltages"]
        currents = data["currents"]
        power = np.array(voltages) * np.array(currents)

        plt.figure(figsize=(12, 6))
        plt.plot(voltages, currents, label="Current (A)", color="blue", linewidth=2)
        plt.plot(voltages, power, label="Power (W)", linestyle="--", color="green", linewidth=2)
        plt.xlabel("Voltage (V)", fontsize=14)
        plt.ylabel("Current (A) / Power (W)", fontsize=14)
        plt.title(f"V-I and Power Dissipation of {self.material}", fontsize=16, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
        plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_conductance_vs_voltage(diode, voltage_range=(-2, 2), steps=1000):
        data = diode.calculate_vi(voltage_range, steps)
        voltages = np.array(data["voltages"])
        currents = np.array(data["currents"])
        conductance = np.gradient(currents, voltages)  # Numerical derivative

        plt.figure(figsize=(10, 5))
        plt.plot(voltages, conductance, label="Conductance (dI/dV)", color="red")
        plt.xlabel("Voltage (V)")
        plt.ylabel("Conductance (Siemens)")
        plt.title(f"Differential Conductance vs. Voltage ({diode.material})")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_bandgap_vs_temperature(diode, temperature_range=(200, 600)):
        temperatures = np.linspace(temperature_range[0], temperature_range[1], 100)
        k = 8.617333262145e-5  # Boltzmann constant in eV/K
        bandgaps = diode.bandgap_energy - (0.0007 * (temperatures - 300))  # Linear approximation

        plt.figure(figsize=(8, 5))
        plt.plot(temperatures, bandgaps, label="Bandgap Energy (eV)", color="purple")
        plt.xlabel("Temperature (K)")
        plt.ylabel("Bandgap Energy (eV)")
        plt.title(f"Bandgap Energy vs Temperature ({diode.material})")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_reverse_breakdown(self, voltage_range=None, steps=500):
        """Plot the reverse breakdown region of the diode."""
        if voltage_range is None:
            voltage_range = (-self.breakdown_voltage - 50, -self.breakdown_voltage + 10)

        data = self.calculate_vi(voltage_range, steps)
        
        plt.figure(figsize=(8, 5))
        plt.plot(data["voltages"], data["currents"], label="Reverse Breakdown", color="brown")
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        plt.title(f"Reverse Breakdown Region ({self.material})")
        plt.grid(True)
        plt.legend()
        plt.show()


    def plot_junction_capacitance_vs_voltage(self, voltage_range=None, steps=500):
        """Plot Junction Capacitance vs Reverse Voltage."""
        if voltage_range is None:
            voltage_range = (-self.breakdown_voltage, 0)  # Correctly access self.breakdown_voltage

        voltages = np.linspace(voltage_range[0], voltage_range[1], steps)
        capacitances = self.junction_capacitance(area=1e-6, width=1e-6) / np.sqrt(1 - (voltages / self.breakdown_voltage))

        plt.figure(figsize=(8, 5))
        plt.plot(voltages, capacitances, label="Junction Capacitance", color="cyan")
        plt.xlabel("Reverse Voltage (V)")
        plt.ylabel("Capacitance (F)")
        plt.title(f"Junction Capacitance vs Reverse Voltage ({self.material})")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_drift_diffusion_currents(diode):
        electric_fields = np.linspace(0, 1e5, 100)
        diffusion_gradients = np.linspace(0, 1e6, 100)

        drift_currents = diode.drift_current(electric_fields)
        diffusion_currents = diode.diffusion_current(diffusion_gradients)

        plt.figure(figsize=(10, 5))
        plt.plot(electric_fields, drift_currents, label="Drift Current", color="blue")
        plt.plot(diffusion_gradients, diffusion_currents, label="Diffusion Current", color="green")
        plt.xlabel("Field Strength / Charge Gradient")
        plt.ylabel("Current (A)")
        plt.title(f"Drift & Diffusion Currents ({diode.material})")
        plt.grid(True)
        plt.legend()
        plt.show()


    def __repr__(self):
        return (
            f"Diode(material={self.material}, ideality_factor={self.ideality_factor}, "
            f"temperature={self.temperature} K)"
        )
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from dataclasses import dataclass

@dataclass
class TemperatureEffects:
    """Container for temperature-dependent parameters."""
    bandgap: float
    mobility: float
    carrier_concentration: float
    resistivity: float
    thermal_conductivity: float
    diffusion_coefficient: float

@dataclass
class DiodeConstants:
    """Physical constants used in semiconductor calculations."""
    BOLTZMANN = 1.380649e-23  # Boltzmann constant [J/K]
    ELECTRON_CHARGE = 1.602176634e-19  # Elementary charge [C]
    ROOM_TEMP = 300.0  # Room temperature [K]

class DiodeforTemp:
    def __init__(self, material, temperature=300, custom_props=None):
        """Initialize diode with corrected temperature handling."""
        self.constants = DiodeConstants()
        self._validate_temperature(temperature)
        self.temperature = temperature
        
        # Initialize material properties (existing code remains the same)
        if custom_props:
            self._init_custom_properties(custom_props)
        else:
            self._init_material_properties(material)
            
        # Calculate temperature-dependent parameters
        self.update_temperature_parameters()
    
    def _validate_temperature(self, temperature: float) -> None:
        """Validate temperature is within physical bounds."""
        if not (0 < temperature < 1000):
            raise ValueError(
                f"Temperature must be between 0K and 1000K. Got: {temperature}K"
            )
    
    def update_temperature_parameters(self) -> None:
        """Update all temperature-dependent parameters."""
        # Thermal voltage calculation (kT/q)
        self.vt = (self.constants.BOLTZMANN * self.temperature) / self.constants.ELECTRON_CHARGE
        
        # Update temperature-dependent saturation current
        self.isat_t = self.calculate_saturation_current(self.temperature)
        
        # Update other temperature-dependent parameters
        self.update_bandgap()
        self.update_mobility()
    
    def calculate_saturation_current(self, temperature: float) -> float:
        """
        Calculate temperature-dependent saturation current.
        
        Args:
            temperature: Operating temperature [K]
            
        Returns:
            float: Temperature-corrected saturation current [A]
        """
        k = self.constants.BOLTZMANN
        q = self.constants.ELECTRON_CHARGE
        t_ratio = temperature / self.constants.ROOM_TEMP
        
        # Temperature dependence of bandgap
        eg_t = self.bandgap_energy * (1 - 0.0002677 * (temperature - self.constants.ROOM_TEMP))
        
        # Calculate temperature-dependent saturation current
        isat_t = self.isat * (t_ratio ** 3) * np.exp(
            (q * eg_t / (2 * k)) * (1/self.constants.ROOM_TEMP - 1/temperature)
        )
        
        return isat_t
    
    def update_bandgap(self) -> None:
        """Update temperature-dependent bandgap."""
        # Varshni equation parameters (material dependent)
        alpha = self.get_varshni_parameters()['alpha']
        beta = self.get_varshni_parameters()['beta']
        
        # Varshni equation for temperature-dependent bandgap
        self.bandgap_t = self.bandgap_energy - (
            (alpha * self.temperature ** 2) / (self.temperature + beta)
        )
    
    def update_mobility(self) -> None:
        """Update temperature-dependent carrier mobility."""
        # Power law approximation for mobility
        self.mobility_t = self.mobility_300 * (self.temperature / 300) ** (-2.42)
    
    def calculate_vi(self, voltage_range=(-2, 2), steps=1000):
        """Calculate V-I characteristics with corrected temperature dependence."""
        # Use instance thermal voltage calculated from temperature
        vt = self.vt
        isat_t = self.isat_t  # Use temperature-corrected saturation current
        
        voltages = np.linspace(voltage_range[0], voltage_range[1], steps)
        currents = np.zeros_like(voltages)
        
        # Vectorized calculations for different regions
        forward_mask = voltages >= 0
        reverse_mask = ~forward_mask
        
        # Forward bias region
        currents[forward_mask] = isat_t * (
            np.exp(voltages[forward_mask] / (self.ideality_factor * vt)) - 1
        )
        
        # Reverse bias region with temperature-dependent breakdown
        breakdown_voltage_t = self.breakdown_voltage * (1 + 0.0006 * (self.temperature - 300))
        reverse_voltages = voltages[reverse_mask]
        
        currents[reverse_mask] = -isat_t * np.ones_like(reverse_voltages)
        breakdown_mask = reverse_voltages < -breakdown_voltage_t
        if np.any(breakdown_mask):
            currents[reverse_mask][breakdown_mask] = -isat_t * (
                1 + np.abs(reverse_voltages[breakdown_mask] + breakdown_voltage_t)
            )
        
        return {"voltages": voltages, "currents": currents}
    
    def get_varshni_parameters(self) -> Dict[str, float]:
        """Get Varshni parameters for bandgap temperature dependence."""
        # Material-specific Varshni parameters
        params = {
            "Silicon": {"alpha": 4.73e-4, "beta": 636},
            "Germanium": {"alpha": 4.77e-4, "beta": 235},
            "Gallium Arsenide": {"alpha": 5.41e-4, "beta": 204},
        }
        return params.get(self.material, {"alpha": 4.73e-4, "beta": 636})  # Default to Silicon
    
    def get_thermal_resistance(self, area: float, thickness: float) -> float:
        """
        Calculate thermal resistance of the device.
        
        Args:
            area: Junction area [m²]
            thickness: Device thickness [m]
            
        Returns:
            float: Thermal resistance [K/W]
        """
        # Material-specific thermal conductivity [W/(m·K)]
        thermal_conductivity = {
            "Silicon": 148,
            "Germanium": 60,
            "Gallium Arsenide": 55
        }.get(self.material, 148)  # Default to Silicon
        
        return thickness / (thermal_conductivity * area)
    
    def calculate_junction_temperature(
        self, ambient_temp: float, power_dissipation: float, 
        thermal_resistance: float
    ) -> float:
        """
        Calculate junction temperature based on power dissipation.
        
        Args:
            ambient_temp: Ambient temperature [K]
            power_dissipation: Power dissipated in device [W]
            thermal_resistance: Thermal resistance [K/W]
            
        Returns:
            float: Junction temperature [K]
        """
        return ambient_temp + (power_dissipation * thermal_resistance)
    
class DiodeTemperature(Diode):
    """Extension of Diode class with advanced temperature effects."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Retrieve mobility_300 from material properties
        if self.material in self.MATERIAL_PROPERTIES:
            self.mobility_300 = self.MATERIAL_PROPERTIES[self.material].get("mobility_300")
            if self.mobility_300 is None:
                raise ValueError(f"'mobility_300' is missing for material '{self.material}'.")
        else:
            raise ValueError(f"Material '{self.material}' not found in MATERIAL_PROPERTIES.")

        # Compute temperature effects
        self.temp_effects = self.calculate_temperature_effects()

    def get_varshni_parameters(self):
        """Return Varshni parameters (alpha and beta) for the material."""
        varshni_params = {
            "Silicon": {"alpha": 4.73e-4, "beta": 636},  # Values in eV/K and K
            "Germanium": {"alpha": 4.77e-4, "beta": 235},
            "Gallium Arsenide": {"alpha": 5.41e-4, "beta": 204},
        }
        if self.material not in varshni_params:
            raise ValueError(f"Varshni parameters not defined for material '{self.material}'.")
        return varshni_params[self.material]

    def get_effective_mass_ratio(self):
        """Return the effective mass ratio (m*/m0) for the material."""
        effective_mass_ratios = {
            "Silicon": 1.08,  # Electron effective mass in m*/m0
            "Germanium": 0.56,
            "Gallium Arsenide": 0.067,
        }
        if self.material not in effective_mass_ratios:
            raise ValueError(f"Effective mass ratio not defined for material '{self.material}'.")
        return effective_mass_ratios[self.material]
    
    def get_thermal_conductivity_300K(self):
        """Return the thermal conductivity at 300 K for the material."""
        thermal_conductivities = {
            "Silicon": 148,  # W/m·K
            "Germanium": 60,
            "Gallium Arsenide": 55,
        }
        if self.material not in thermal_conductivities:
            raise ValueError(f"Thermal conductivity not defined for material '{self.material}'.")
        return thermal_conductivities[self.material]

    def get_thermal_resistance(self, area, thickness):
        """Calculate thermal resistance (K/W) based on area and thickness."""
        thermal_conductivity = self.get_thermal_conductivity_300K()
        return thickness / (thermal_conductivity * area)

    def calculate_junction_temperature(self, ambient_temp, power, thermal_resistance):
        """Calculate junction temperature."""
        return ambient_temp + (power * thermal_resistance)

    def calculate_leakage_current(self):
        """Calculate leakage current based on reverse saturation current and temperature."""
        isat_t = self.calculate_saturation_current(self.temperature)
        return isat_t  # Leakage current is approximately equal to saturation current in reverse bias

    def calculate_breakdown_voltage_temp(self):
        """Calculate temperature-dependent breakdown voltage."""
        # Approximation: Linear decrease with temperature
        temp_coeff = -0.001  # Example coefficient (V/K)
        return self.breakdown_voltage + (self.temperature - 300) * temp_coeff
    
    def calculate_lifetime_factor(self):
        """Calculate relative lifetime factor based on temperature."""
        activation_energy = 0.7  # Example value in eV
        k = 8.617333262145e-5  # Boltzmann constant in eV/K
        return np.exp(-activation_energy / (k * self.temperature))

    def calculate_temperature_effects(self) -> TemperatureEffects:
        """Calculate all temperature-dependent parameters."""
        bandgap = self.calculate_temperature_bandgap()
        mobility = self.calculate_temperature_mobility()
        carrier_conc = self.calculate_carrier_concentration(bandgap=bandgap)  # Pass bandgap explicitly
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
        """Calculate temperature-dependent bandgap using advanced model."""
        # Varshni parameters
        params = self.get_varshni_parameters()
        alpha, beta = params['alpha'], params['beta']
        
        # Advanced bandgap model including strain effects
        strain_factor = 1.0  # Can be modified for strained semiconductors
        return (self.bandgap_energy * strain_factor - 
                (alpha * self.temperature**2) / (self.temperature + beta))
    
    def calculate_temperature_mobility(self) -> float:
        """Calculate temperature-dependent carrier mobility."""
        # Advanced mobility model including various scattering mechanisms
        phonon_scattering = (self.temperature / 300) ** (-2.42)
        ionized_impurity = (self.temperature / 300) ** (3/2)
        
        # Combined mobility effect
        return self.mobility_300 * (1 / (1/phonon_scattering + 1/ionized_impurity))
    
    def calculate_carrier_concentration(self, bandgap=None):
        """Calculate intrinsic carrier concentration."""
        if bandgap is None:
            bandgap = self.calculate_temperature_bandgap()  # Use bandgap if not provided

        effective_mass_ratio = self.get_effective_mass_ratio()
        k = 8.617333262145e-5  # Boltzmann constant in eV/K
        nc_nv = 2.5e19 * (self.temperature / 300) ** 3 * effective_mass_ratio

        return nc_nv * np.exp(-bandgap / (2 * k * self.temperature))

    
    def calculate_resistivity(self, mobility: float, carrier_conc: float) -> float:
        """Calculate temperature-dependent resistivity."""
        return 1 / (self.ELECTRON_CHARGE * mobility * carrier_conc)
    
    def calculate_thermal_conductivity(self) -> float:
        """Calculate temperature-dependent thermal conductivity."""
        # Material-specific parameters
        k300 = self.get_thermal_conductivity_300K()
        return k300 * (300 / self.temperature) ** 1.5
    
    def calculate_diffusion_coefficient(self, mobility: float) -> float:
        """Calculate temperature-dependent diffusion coefficient."""
        # Einstein relation
        return mobility * self.BOLTZMANN_CONSTANT * self.temperature / self.ELECTRON_CHARGE
    
    # Visualization Methods
    
    def plot_comprehensive_temperature_effects(self, temp_range: Tuple[float, float] = (250, 450)):
        """Create a comprehensive visualization of temperature effects."""
        temperatures = np.linspace(temp_range[0], temp_range[1], 100)
        
        # Calculate parameters across temperature range
        params = {
            'Bandgap (eV)': [],
            'Mobility (cm²/V·s)': [],
            'Carrier Conc. (cm⁻³)': [],
            'Resistivity (Ω·cm)': [],
            'Thermal Cond. (W/m·K)': [],
            'Diffusion Coeff. (cm²/s)': []
        }
        
        for temp in temperatures:
            self.temperature = temp
            effects = self.calculate_temperature_effects()
            params['Bandgap (eV)'].append(effects.bandgap)
            params['Mobility (cm²/V·s)'].append(effects.mobility)
            params['Carrier Conc. (cm⁻³)'].append(effects.carrier_concentration)
            params['Resistivity (Ω·cm)'].append(effects.resistivity)
            params['Thermal Cond. (W/m·K)'].append(effects.thermal_conductivity)
            params['Diffusion Coeff. (cm²/s)'].append(effects.diffusion_coefficient)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Temperature Effects in {self.material} Diode', fontsize=16)
        
        for (param, values), ax in zip(params.items(), axes.flat):
            ax.plot(temperatures, values, 'b-')
            ax.set_xlabel('Temperature (K)')
            ax.set_ylabel(param)
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_vi_temperature_family(self, temp_range: List[float], voltage_range: Tuple[float, float]):
        """Plot family of V-I curves at different temperatures."""
        plt.figure(figsize=(10, 6))
        
        for temp in temp_range:
            self.temperature = temp
            data = self.calculate_vi(voltage_range)
            plt.semilogy(data['voltages'], np.abs(data['currents']), 
                        label=f'T = {temp}K')
        
        plt.grid(True)
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current (A)')
        plt.title(f'Temperature Dependence of {self.material} Diode V-I Characteristics')
        plt.legend()
        plt.show()
    
    def plot_power_dissipation_effects(self, voltage_range: Tuple[float, float], ambient_temp: float = 300, steps: int = 1000):
        """Visualize power dissipation and temperature effects."""
        # Generate voltage and current data with the same number of steps
        voltages = np.linspace(voltage_range[0], voltage_range[1], steps)
        vi_data = self.calculate_vi(voltage_range, steps)
        currents = vi_data["currents"]

        # Calculate power dissipation
        power = np.array(voltages) * np.array(currents)

        # Calculate junction temperature
        thermal_resistance = self.get_thermal_resistance(1e-6, 1e-4)  # Example area and thickness
        junction_temps = [self.calculate_junction_temperature(ambient_temp, p, thermal_resistance) for p in power]

        # Plot power dissipation and junction temperature
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Power dissipation plot
        ax1.plot(voltages, power * 1000, 'r-')  # Convert to mW
        ax1.set_xlabel('Voltage (V)')
        ax1.set_ylabel('Power Dissipation (mW)')
        ax1.grid(True)

        # Junction temperature plot
        ax2.plot(voltages, junction_temps, 'b-')
        ax2.set_xlabel('Voltage (V)')
        ax2.set_ylabel('Junction Temperature (K)')
        ax2.grid(True)

        plt.suptitle('Power Dissipation and Junction Temperature Effects')
        plt.tight_layout()
        plt.show()

    def plot_temperature_reliability_indicators(self, temp_range: Tuple[float, float]):
        """Plot reliability indicators vs temperature."""
        temperatures = np.linspace(temp_range[0], temp_range[1], 100)
        
        # Calculate reliability indicators
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
        ax1.set_xlabel('Temperature (K)')
        ax1.set_ylabel('Leakage Current (A)')
        ax1.grid(True)
        
        ax2.plot(temperatures, breakdown_voltage)
        ax2.set_xlabel('Temperature (K)')
        ax2.set_ylabel('Breakdown Voltage (V)')
        ax2.grid(True)
        
        ax3.plot(temperatures, lifetime_factor)
        ax3.set_xlabel('Temperature (K)')
        ax3.set_ylabel('Relative Lifetime')
        ax3.grid(True)
        
        plt.suptitle('Temperature-Dependent Reliability Indicators')
        plt.tight_layout()
        plt.show()