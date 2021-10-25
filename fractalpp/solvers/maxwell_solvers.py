import numpy as np
import bempp.api
from bempp.api.linalg import lu
# import numba


class max_sol():
    def __init__(self,grid,uinc):
        self.uinc = uinc
        self.div_space = bempp.api.function_space(grid, "RWG", 0)
        self.curl_space = bempp.api.function_space(grid, "SNC", 0)
        elec = bempp.api.operators.boundary.maxwell.electric_field(self.div_space, self.div_space, self.curl_space, self.uinc.k)
        _, tangential_trace = self.getRHS_data()
        trace_fun = bempp.api.GridFunction(self.div_space, fun=tangential_trace, dual_space=self.curl_space)
        self.lambda_data = lu(elec, trace_fun)

    def getRHS_data(self):
        x = self.uinc.incDir # object created in solve.py
        k = self.uinc.k
        def incident_field(x):
            return np.array([np.exp(1j * k * x[2]), 0. * x[2], 0. * x[2]])

        @bempp.api.complex_callable
        def tangential_trace(x, n, domain_index, result):
            incident_field = np.array([np.exp(1j * k * x[2]), 0. * x[2], 0. * x[2]])
            result[:] = np.cross(incident_field, n)

        return incident_field, tangential_trace

    def u_s(self,points):
        slp_pot = bempp.api.operators.potential.maxwell.electric_field(self.div_space, points, self.uinc.k)
        return -slp_pot * self.lambda_data