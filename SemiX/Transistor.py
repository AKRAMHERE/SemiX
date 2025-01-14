import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define Semiconductor Material Properties
MATERIALS = {
    "Si": {"bandgap": 1.12, "mobility": 1400, "dielectric": 11.7},
    "Ge": {"bandgap": 0.66, "mobility": 3900, "dielectric": 16},
    "GaAs": {"bandgap": 1.43, "mobility": 8500, "dielectric": 12.9},
    "SiC": {"bandgap": 3.26, "mobility": 700, "dielectric": 9.7},
}

class Transistor:
    class Bjt:
        def __init__(self, material: str, beta: float = 100, V_A: float = 50, I_S: float = 1e-12, temperature: float = 300):
            self.material = material
            self.beta = beta  # Current gain
            self.V_A = V_A  # Early voltage
            self.I_S = I_S  # Saturation current
            self.V_T = 0.0259 * (temperature / 300)  # Thermal voltage
            self.temperature = temperature
            
            if material not in MATERIALS:
                raise ValueError("Material not supported")
            self.properties = MATERIALS[material]

        def collector_current(self, VBE, VBC):
            return self.I_S * (np.exp(VBE / self.V_T) - np.exp(VBC / self.V_T))

        def base_current(self, VBE, VBC):
            return self.collector_current(VBE, VBC) / self.beta

        def plot_i_v(self, VBE_range=(0, 1), VCE_range=(0, 10), steps=100):
            VBE = np.linspace(*VBE_range, steps)
            VCE = np.linspace(*VCE_range, steps)
            IC = self.collector_current(VBE, 0) * (1 + VCE / self.V_A)
            
            plt.figure()
            plt.plot(VCE, IC, label=f'BJT {self.material}')
            plt.xlabel("VCE (V)")
            plt.ylabel("IC (mA)")
            plt.title(f"BJT Output Characteristics ({self.material}) - Ebers-Moll Model")
            plt.legend()
            plt.grid()
            plt.show()

        def hybrid_pi_model(self, IC):
            g_m = IC / self.V_T  # Transconductance
            r_pi = self.beta / g_m  # Base resistance
            r_o = (self.V_A + 10) / IC  # Output resistance
            C_pi = g_m * self.V_T * 1e-12  # Base capacitance
            C_mu = 2e-12  # Assumed junction capacitance
            
            return {"g_m": g_m, "r_pi": r_pi, "r_o": r_o, "C_pi": C_pi, "C_mu": C_mu}

        def diffusion_capacitance(self, IC):
            tau_F = 1e-9  # Assumed base transit time
            C_diff = tau_F * IC / self.V_T
            return C_diff

        def temperature_effects(self, VBE_range=(0, 1), VCE=5, temp_range=(250, 450), steps=100):
            VBE = np.linspace(*VBE_range, steps)
            plt.figure()
            
            for temp in np.linspace(*temp_range, 5):
                VT = 0.0259 * (temp / 300)  # Thermal voltage adjustment
                IC = self.I_S * (np.exp(VBE / VT) - 1) * (1 + VCE / self.V_A)
                plt.plot(VBE, IC, label=f'T = {temp}K')
            
            plt.xlabel("VBE (V)")
            plt.ylabel("IC (mA)")
            plt.title(f"BJT Temperature Effects ({self.material}) - Gummel-Poon Model")
            plt.legend()
            plt.grid()
            plt.show()

    class Mosfet:
        def __init__(self, material: str, Vth: float = 1.0, mu_n: float = None, Cox: float = 10e-3):
            self.material = material
            self.Vth = Vth  # Threshold voltage
            self.Cox = Cox  # Oxide capacitance
            
            if material not in MATERIALS:
                raise ValueError("Material not supported")
            self.properties = MATERIALS[material]
            self.mu_n = mu_n if mu_n else self.properties['mobility']

        def id_vs_vds(self, VGS_values, VDS_range):
            VDS = np.linspace(*VDS_range, 100)
            plt.figure()
            
            for VGS in VGS_values:
                ID = np.where(VDS < VGS - self.Vth,
                              self.mu_n * self.Cox * ((VGS - self.Vth) * VDS - 0.5 * VDS**2),
                              0.5 * self.mu_n * self.Cox * (VGS - self.Vth) ** 2)
                plt.plot(VDS, ID, label=f'VGS = {VGS} V')
            
            plt.xlabel("VDS (V)")
            plt.ylabel("ID (mA)")
            plt.title(f"{self.material} MOSFET Output Characteristics")
            plt.legend()
            plt.grid()
            plt.show()

    class Jfet:
        def __init__(self, material: str, Vp: float = -4, Idss: float = 10e-3):
            self.material = material
            self.Vp = Vp  # Pinch-off voltage
            self.Idss = Idss  # Drain-source saturation current

        def id_vs_vds(self, VGS_values, VDS_range):
            VDS = np.linspace(*VDS_range, 100)  # Generate an array of VDS values
            plt.figure()
            
            for VGS in VGS_values:
                # Ensure ID is an array of the same length as VDS
                ID = self.Idss * (1 - VGS / self.Vp) ** 2 * np.ones_like(VDS)
                plt.plot(VDS, ID, label=f'VGS = {VGS} V')  # Now both arrays have matching dimensions
            
            plt.xlabel("VDS (V)")
            plt.ylabel("ID (mA)")
            plt.title(f"{self.material} JFET Output Characteristics")
            plt.legend()
            plt.grid()
            plt.show()

# Example Usage

