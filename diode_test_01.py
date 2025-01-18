from SemiX import Diode
from SemiX import DiodeTemperature
# Test Silicon
print("Testing Silicon...")
silicon_diode = Diode(material="Silicon", temperature=300)
silicon_diode.plot_vi(voltage_range=(-5, 2), steps=50)
silicon_diode.plot_vi(voltage_range=(-5, 2), steps=50, log_scale=True)
silicon_diode.plot_temperature_effects(voltage_range=(-5, 5), steps=50, temperature_range=(250, 450))

silicon_diode = DiodeTemperature("Silicon", temperature=300)
silicon_diode.plot_comprehensive_temperature_effects(temp_range=(250, 450))
silicon_diode.plot_vi_temperature_family([250, 300, 350, 400], voltage_range=(-2, 2))
silicon_diode.plot_power_dissipation_effects(voltage_range=(0.1, 1), ambient_temp=300)
silicon_diode.plot_temperature_reliability_indicators((250, 400))

# Test Germanium
print("Testing Germanium...")
germanium_diode = Diode(material="Germanium", temperature=300)
germanium_diode.plot_vi(voltage_range=(-2, 2), steps=50)
germanium_diode = DiodeTemperature("Germanium", temperature=300)
germanium_diode.plot_comprehensive_temperature_effects(temp_range=(250, 450))
germanium_diode.plot_vi_temperature_family([250, 300, 350, 400], voltage_range=(-2, 2))
germanium_diode.plot_power_dissipation_effects(voltage_range=(0.1, 1), ambient_temp=300)
germanium_diode.plot_temperature_reliability_indicators((250, 400))

# Test Gallium Arsenide
print("Testing Gallium Arsenide...")
gaas_diode = Diode(material="Gallium Arsenide", temperature=300)
gaas_diode.plot_vi(voltage_range=(-5, 2), steps=50)
gaas_diode = DiodeTemperature("Gallium Arsenide", temperature=300)
gaas_diode.plot_comprehensive_temperature_effects(temp_range=(250, 450))
gaas_diode.plot_vi_temperature_family([250, 300, 350, 400], voltage_range=(-2, 2))
gaas_diode.plot_power_dissipation_effects(voltage_range=(0.1, 1), ambient_temp=300)
gaas_diode.plot_temperature_reliability_indicators((250, 400))