from SemiX import DiodeTemperature

silicon_diode = DiodeTemperature("Silicon", temperature=300)
silicon_diode.plot_comprehensive_temperature_effects(temp_range=(250, 450))
silicon_diode.plot_vi_temperature_family([250, 300, 350, 400], voltage_range=(-2, 2))
silicon_diode.plot_power_dissipation_effects(voltage_range=(0.1, 1), ambient_temp=300)
silicon_diode.plot_temperature_reliability_indicators((250, 400))

germanium_diode = DiodeTemperature("Germanium", temperature=300)
germanium_diode.plot_comprehensive_temperature_effects(temp_range=(250, 450))
germanium_diode.plot_vi_temperature_family([250, 300, 350, 400], voltage_range=(-2, 2))
germanium_diode.plot_power_dissipation_effects(voltage_range=(0.1, 1), ambient_temp=300)
germanium_diode.plot_temperature_reliability_indicators((250, 400))

gaas_diode = DiodeTemperature("Gallium Arsenide", temperature=300)
gaas_diode.plot_comprehensive_temperature_effects(temp_range=(250, 450))
gaas_diode.plot_vi_temperature_family([250, 300, 350, 400], voltage_range=(-2, 2))
gaas_diode.plot_power_dissipation_effects(voltage_range=(0.1, 1), ambient_temp=300)
gaas_diode.plot_temperature_reliability_indicators((250, 400))