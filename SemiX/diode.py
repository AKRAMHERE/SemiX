import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
from math import pi
import csv


class Diode:
    BOLTZMANN_CONSTANT = 1.380649e-23  # in J/K
    MATERIAL_PROPERTIES = {
        "Silicon": {
            "vt": 0.026,
            "isat": 10e-9,  # Reverse saturation current (A) at 300K
            "ideality_factor": 2,
            "cut_in_voltage": 0.7,
            "breakdown_voltage": 1000,  # Reverse breakdown voltage (V)
            "bandgap_energy": 1.12,  # Bandgap energy (eV)
            "eps": 11.7 * 8.854e-12,  # Permittivity of Silicon (F/m)
        },
        "Germanium": {
            "vt": 0.0258,
            "isat": 1e-6,  # Reverse saturation current (A) at 300K
            "ideality_factor": 1,
            "cut_in_voltage": 0.3,
            "breakdown_voltage": 300,  # Reverse breakdown voltage (V)
            "bandgap_energy": 0.66,  # Bandgap energy (eV)
            "eps": 16.0 * 8.854e-12,  # Permittivity of Germanium (F/m)
        },
        "Gallium Arsenide": {
            "vt": 0.027,  # Thermal voltage at 300 K
            "isat": 1e-10,  # Very small saturation current due to high bandgap
            "ideality_factor": 1.5,
            "cut_in_voltage": 1.2,
            "breakdown_voltage": 500,  # Reverse breakdown voltage (V)
            "bandgap_energy": 1.42,  # Bandgap energy (eV)
            "eps": 12.9 * 8.854e-12,  # Permittivity of GaAs (F/m)
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
