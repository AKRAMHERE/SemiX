from SemiX import Transistor
# Example Usage:

def main():
    """Run example simulations for each transistor type."""
    # --------------------- BJT Examples ---------------------
    bjt = Transistor.Bjt(material="Si", beta=120, V_A=60, I_S=1e-12,
                         temperature=300)
    # Plot collector current vs. V_CE.
    bjt.plot_i_v(VBE_range=(0.6, 0.8), VCE_range=(0, 10), steps=100)

    # Plot temperature effects.
    bjt.temperature_effects(VBE_range=(0.6, 0.8), VCE=5, temp_range=(250, 450))

    # Calculate and print emitter current for given V_BE and V_BC.
    VBE_example = 0.7
    VBC_example = 0.0
    I_E = bjt.emitter_current(VBE_example, VBC_example)
    print(f"BJT Emitter Current (V_BE={VBE_example} V, V_BC={VBC_example} V): "
          f"{I_E:.2e} A")

    # --------------------- MOSFET Examples ---------------------
    mosfet = Transistor.Mosfet(material="Si", Vth=1.0, Cox=10e-3)
    # Plot output characteristics.
    mosfet.id_vs_vds(VGS_values=[1.5, 2.0, 2.5, 3.0], VDS_range=(0, 5))
    # Plot transfer characteristic.
    mosfet.plot_transfer_characteristic(VGS_range=(0, 4), VDS=5)
    # Calculate and print transconductance.
    VGS_test = 2.5
    gm = mosfet.calculate_transconductance(VGS_test)
    print(f"MOSFET Transconductance (V_GS={VGS_test} V): {gm:.2e} S")

    # --------------------- JFET Examples ---------------------
    jfet = Transistor.Jfet(material="Si", Vp=-4, Idss=10e-3)
    # Plot output characteristics.
    jfet.id_vs_vds(VGS_values=[-1, -2, -3, -4], VDS_range=(0, 10))
    # Plot transfer characteristic.
    jfet.plot_transfer_characteristics(VGS_range=(-5, 0))
    # Calculate and print drain current for a given V_GS.
    VGS_jfet = -2.5
    ID_jfet = jfet.calculate_id(VGS_jfet)
    print(f"JFET Drain Current (V_GS={VGS_jfet} V): {ID_jfet:.2e} A")


if __name__ == "__main__":
    main()
