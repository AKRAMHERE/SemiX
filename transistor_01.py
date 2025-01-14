from SemiX import Transistor

# BJT Example
BJT = Transistor.Bjt("Si")
BJT.plot_i_v()
print("Hybrid Pi Model:", BJT.hybrid_pi_model(1e-3))
print("Diffusion Capacitance:", BJT.diffusion_capacitance(1e-3))
BJT.temperature_effects()

# MOSFET Example
NMOS = Transistor.Mosfet("Si", Vth=1.2)
NMOS.id_vs_vds(VGS_values=[2, 3, 4], VDS_range=(0, 5))

# JFET Example
JFET = Transistor.Jfet("Si", Vp=-4, Idss=10e-3)
JFET.id_vs_vds(VGS_values=[-2, -3], VDS_range=(0, 5))
