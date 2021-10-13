import bempp.api
import numpy as np
import fractalpp
prefractal_level = 0
#mesh parameter
h = 0.5
#inc wave params:
kwave = 5
root1over3 = np.sqrt(1/3)
inc_dir = np.array([root1over3,root1over3,-root1over3])
#impedance parameters:
lambda_plus = kwave*(1.5+1.5j)
lambda_minus = kwave*(1+1j)

koch_mesh = fractalpp.mesh.koch(prefractal_level,h_max = h)
# koch_mesh.plot()

# create incident wave
uinc = fractalpp.solve.u_i(kwave,inc_dir)
imp_sol = fractalpp.solve.impedance(koch_mesh,uinc,lambda_plus,lambda_minus)