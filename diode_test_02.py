from SemiX import Diode

# Test Silicon
print("Testing Silicon...")
silicon_diode = Diode(material="Silicon", temperature=300)
silicon_diode.plot_vi(voltage_range=(-5, 2), steps=50)
silicon_diode.plot_vi(voltage_range=(-5, 2), steps=50, log_scale=True)
silicon_diode.plot_temperature_effects(voltage_range=(-5, 5), steps=50, temperature_range=(250, 450))
silicon_diode.plot_v_with_power(voltage_range=(-5, 5), steps=100)
silicon_diode.plot_noise_vs_temperature()
silicon_diode.animate_vi(voltage_range=(-5, 2), steps=50, temperature_range=(250, 450), interval=500)
silicon_diode.plot_material_comparison()
silicon_diode.plot_reverse_breakdown()
silicon_diode.plot_bandgap_vs_temperature(silicon_diode)
silicon_diode.plot_junction_capacitance_vs_voltage()
silicon_diode.plot_conductance_vs_voltage(silicon_diode)
silicon_diode.plot_drift_diffusion_currents()

# Test Germanium
print("Testing Germanium...")
germanium_diode = Diode(material="Germanium", temperature=300)
germanium_diode.plot_vi(voltage_range=(-2, 2), steps=50)
germanium_diode.plot_vi_with_power(voltage_range=(-2, 2), steps=50)
germanium_diode.plot_noise_vs_temperature()
germanium_diode.plot_bandgap_vs_temperature(germanium_diode)
germanium_diode.plot_junction_capacitance_vs_voltage()
germanium_diode.plot_conductance_vs_voltage(germanium_diode)
germanium_diode.plot_drift_diffusion_currents()

# Test Gallium Arsenide
print("Testing Gallium Arsenide...")
gaas_diode = Diode(material="Gallium Arsenide", temperature=300)
gaas_diode.plot_vi(voltage_range=(-5, 2), steps=50)
gaas_diode.plot_vi_with_power(voltage_range=(-5, 2), steps=50)
gaas_diode.plot_noise_vs_temperature()
gaas_diode.plot_bandgap_vs_temperature()
gaas_diode.plot_junction_capacitance_vs_voltage(gaas_diode)
gaas_diode.plot_conductance_vs_voltage()
gaas_diode.plot_drift_diffusion_currents(gaas_diode)
