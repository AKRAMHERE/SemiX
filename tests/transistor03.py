#!/usr/bin/env python3
"""
Test script to exercise all plot functions in the extended transistor_models module.
"""

from SemiX import Transistor

def test_bjt_plots():
    print("=== Testing BJT Plots ===")
    bjt = Transistor.Bjt(material="Si", beta=100, V_A=50.0, I_S=1e-12, temperature=300.0)
    
    # 1) Output characteristic: collector current vs. collector-emitter voltage
    bjt.plot_i_v(VBE_range=(0.5, 0.8), VCE_range=(0, 10), steps=100)

    # 2) Temperature effect: I_C vs. V_BE at different temperatures
    bjt.temperature_effects(VBE_range=(0.4, 0.8), VCE=5.0, temp_range=(250, 450), steps=100)

    # 3) Gummel plot: log(I_C) vs. V_BE
    bjt.gummel_plot(VBE_range=(0.5, 0.8), steps=100)

    # 4) Toy beta vs. I_C variation
    bjt.beta_vs_ic_plot(Ic_range=(1e-6, 1e-2), steps=100)

def test_mosfet_plots():
    print("=== Testing MOSFET Plots ===")
    mos = Transistor.Mosfet(
        material="Si",
        Vth=1.0,       # threshold voltage
        mu_n=None,     # use default from MATERIALS
        Cox=1e-2,      # gate oxide capacitance per area
        W=10e-6,       # channel width
        L=1e-6,        # channel length
        lambda_mod=0.02
    )

    # 1) Output characteristic: I_D vs. V_DS for various V_GS
    mos.id_vs_vds(VGS_values=[1.0, 1.5, 2.0, 2.5, 3.0], VDS_range=(0, 5))

    # 2) Transfer characteristic: I_D vs. V_GS at a fixed V_DS
    mos.plot_transfer_characteristic(VGS_range=(0, 5), VDS=5, steps=100)

    # 3) Transfer characteristic with subthreshold conduction
    mos.plot_transfer_characteristic_with_subthreshold(VGS_range=(0, 2.0), VDS=5.0, steps=100)

    # 4) g_m / I_D plot
    mos.plot_gm_over_id(VGS_range=(1.0, 3.0), steps=50)

def test_jfet_plots():
    print("=== Testing JFET Plots ===")
    jfet = Transistor.Jfet(
        material="Si",
        Vp=-4.0,     # pinch-off voltage (negative for n-channel)
        Idss=10e-3   # Idss = 10 mA
    )

    # 1) Simple "flat" saturation output characteristic
    jfet.id_vs_vds(VGS_values=[-1.0, -2.0, -3.0], VDS_range=(0, 10))

    # 2) Transfer characteristic: I_D vs. V_GS
    jfet.plot_transfer_characteristics(VGS_range=(-4, 0), steps=100)

    # 3) Complete piecewise (triode + saturation) characteristic
    jfet.id_vs_vds_complete(VGS_values=[-1.0, -2.0, -3.0], VDS_range=(0, 10), steps=100)

def main():
    test_bjt_plots()
    test_mosfet_plots()
    test_jfet_plots()

if __name__ == "__main__":
    main()
