from SemiX import Diode

# Test Silicon
print("Testing Silicon...")
silicon_diode = Diode(material="Silicon", temperature=300)
silicon_diode.plot_vi(voltage_range=(-5, 2), steps=50)
silicon_diode.plot_vi(voltage_range=(-5, 2), steps=50, log_scale=True)
silicon_diode.plot_temperature_effects(voltage_range=(-5, 5), steps=50, temperature_range=(250, 450))

# Test Germanium
print("Testing Germanium...")
germanium_diode = Diode(material="Germanium", temperature=300)
germanium_diode.plot_vi(voltage_range=(-2, 2), steps=50)

# Test Gallium Arsenide
print("Testing Gallium Arsenide...")
gaas_diode = Diode(material="Gallium Arsenide", temperature=300)
gaas_diode.plot_vi(voltage_range=(-5, 2), steps=50)
