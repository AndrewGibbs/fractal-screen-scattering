import numpy as np
import bempp.api

class dir_solve():
    def __init__(self,grid,uinc):
        self.uinc = uinc
        self.piecewise_const_space = bempp.api.function_space(grid, "DP", 0)
        slp = bempp.api.operators.boundary.helmholtz.single_layer(self.piecewise_const_space, self.piecewise_const_space, self.piecewise_const_space, self.uinc.k)
        RHS_data = self.getRHS_data()
        grid_fun = bempp.api.GridFunction(self.piecewise_const_space, fun=RHS_data)
        self.neumann_fun, _ = bempp.api.linalg.gmres(slp, grid_fun)

    def getRHS_data(self):
        k = self.uinc.k
        d = self.uinc.incDir
        @bempp.api.complex_callable
        def RHS_data(x, n, domain_index, result):
            result[0] = np.exp(1j * k * (x[0]*d[0]+x[1]*d[1]+x[2]*d[2]))
        return RHS_data

    def u_s(self,points):
        slp_pot = bempp.api.helmholtz_potential.single_layer(self.piecewise_const_space, points, self.uinc.k)
        return  - slp_pot.evaluate(self.neumann_fun)