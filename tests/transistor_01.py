from SemiX import Transistor


if __name__ == "__main__":
    # =============================================================================
    # Example Usage and Demonstration
    # =============================================================================

    # ---------------------------
    # BJT Demonstration
    # ---------------------------
    print("Demonstrating BJT Model:")
    bjt = Transistor.Bjt(material="Si", beta=150, V_A=100, I_S=1e-15, temperature=300)
    VBE = 0.7
    I_C = bjt.collector_current(VBE)
    print(f"BJT Collector Current at V_BE={VBE} V: {I_C:.2e} A")
    bjt.plot_i_v(VBE_range=(0.6, 0.8), VCE_range=(0, 15), steps=100)
    bjt.temperature_effects(VBE_range=(0.6, 0.8), VCE=10, temp_range=(250, 350), steps=100)

    # ---------------------------
    # MOSFET Demonstration
    # ---------------------------
    print("\nDemonstrating MOSFET Model:")
    mosfet = Transistor.Mosfet(material="Si", Vth=1.2, Cox=10e-3,
                               W=1e-4, L=1e-6, lambda_mod=0.03)
    mosfet.id_vs_vds(VGS_values=[1.5, 2.0, 2.5, 3.0], VDS_range=(0, 10))
    mosfet.plot_transfer_characteristic(VGS_range=(0, 4), VDS=5, steps=100)
    VGS = 2.5
    gm = mosfet.calculate_transconductance(VGS)
    ss_params = mosfet.small_signal_params(VGS, VDS=5)
    print(f"MOSFET at V_GS={VGS} V: g_m={gm:.2e} S, I_D={ss_params['I_D']:.2e} A, r_o={ss_params['r_o']:.2e} Ω")

    # ---------------------------
    # JFET Demonstration
    # ---------------------------
    print("\nDemonstrating JFET Model:")
    jfet = Transistor.Jfet(material="Si", Vp=-4, Idss=15e-3)
    jfet.id_vs_vds(VGS_values=[0, -1, -2, -3], VDS_range=(0, 10))
    jfet.plot_transfer_characteristics(VGS_range=(-4, 0), steps=100)
    VGS_jfet = -2.0
    g_m_jfet = jfet.small_signal_transconductance(VGS_jfet)
    print(f"JFET at V_GS={VGS_jfet} V: g_m={g_m_jfet:.2e} S")
    #ID_jfet = jfet.calculate_id(VGS_jfet)