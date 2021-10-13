import numpy as np
import bempp.api
import numba

def get_impedance_operators(grid,k):
    
    #construct mesh on screen
    L2_subspace = bempp.api.function_space(grid, "DP", 0, include_boundary_dofs=True)
    C1o_subspace = bempp.api.function_space(grid, "P", 1, include_boundary_dofs = False)

    #BIOs
    T_k = bempp.api.operators.boundary.helmholtz.hypersingular(C1o_subspace, L2_subspace, C1o_subspace, k)
    S_k = bempp.api.operators.boundary.helmholtz.single_layer(L2_subspace, L2_subspace, L2_subspace, k)
      
    # Restriction operators
    Id_dcs2dcs = bempp.api.operators.boundary.sparse.identity(L2_subspace, L2_subspace, L2_subspace)
    Id_dcs2cts = bempp.api.operators.boundary.sparse.identity(L2_subspace, L2_subspace, C1o_subspace)
    Id_cts2dcs = bempp.api.operators.boundary.sparse.identity(C1o_subspace, L2_subspace, L2_subspace)
    Id_cts2cts = bempp.api.operators.boundary.sparse.identity(C1o_subspace, L2_subspace, C1o_subspace)

    return L2_subspace,C1o_subspace,T_k,S_k,Id_dcs2dcs,Id_dcs2cts,Id_cts2dcs,Id_cts2cts

class impedance(object):
    def __init__(self,grid,uinc,lambda_plus,lambda_minus):
        self.k = uinc.k
        self.uinc = uinc
        self.lambda_plus = lambda_plus
        self.lambda_minus = lambda_minus

        # now get all (or most) of the operators needed
        self.L2_subspace,self.C1o_subspace,self.T_k,self.S_k,self.Id_dcs2dcs,self.Id_dcs2cts,self.Id_cts2dcs,self.Id_cts2cts = get_impedance_operators(grid,self.k)

    def get_g(self):
        # for compatibility with numba, g needs to be defined without any reference to any class
        lambda_minus = self.lambda_minus
        lambda_plus = self.lambda_plus
        incDir = self.uinc.incDir
        k = self.uinc.k

        def Dirichlet_trace(x):
            return np.exp(1j * k * incDir.dot(x))

        def Neumann_trace(x, n):
            return 1j * k * n.dot(incDir) * Dirichlet_trace(x)

        def g(pm_char,x,n):
            if pm_char == "+":
                pm = 1
                lambda_pm = lambda_plus
            else:
                pm = -1
                lambda_pm = lambda_minus
        
            return -(Neumann_trace(x, n) + pm*lambda_pm*Dirichlet_trace(x))
        return g

    def u_s(self,x):
        S_pot = bempp.api.operators.potential.helmholtz.single_layer(self.L2_subspace, x, self.k)
        D_pot = bempp.api.operators.potential.helmholtz.double_layer(self.C1o_subspace, x, self.k)
        return D_pot.evaluate(self.Neumann_solution)-S_pot.evaluate(self.Dirichlet_solution)


class imp_sol(impedance):
    def __init__(self, grid,uinc,lambda_plus,lambda_minus):
        
        super().__init__(grid,uinc,lambda_plus,lambda_minus)
        
        #impedance operator
        A_rect = bempp.api.BlockedOperator(2, 2)
        A_rect[0,0] = self.lambda_plus*self.lambda_minus*self.Id_cts2cts + (self.lambda_plus + self.lambda_minus)*self.T_k
        A_rect[0,1] = -.5*(self.lambda_plus-self.lambda_minus)*self.Id_dcs2cts
        A_rect[1,0] = .5*(self.lambda_plus-self.lambda_minus)*self.Id_cts2dcs
        A_rect[1,1] = self.Id_dcs2dcs - (self.lambda_plus+self.lambda_minus)*self.S_k

        # get impedance data as numba functions
        data = self.get_data()
        # make RHS
        rhs1 = bempp.api.GridFunction(self.L2_subspace, fun=data[0], dual_space=self.C1o_subspace)
        rhs2 = bempp.api.GridFunction(self.L2_subspace, fun=data[1], dual_space=self.L2_subspace)

        (self.Neumann_solution, self.Dirichlet_solution), _ = bempp.api.linalg.gmres(A_rect, [rhs1, rhs2])

    def get_data(self):
        g = self.get_g()
        lambda_minus = self.lambda_minus
        lambda_plus = self.lambda_plus
        @bempp.api.complex_callable
        def data1(x, n, domain_index, result):
            with numba.objmode():
                result[0] = lambda_minus*g('+',x,n) + lambda_plus*g('-',x,n)

        @bempp.api.complex_callable
        def data2(x, n, domain_index, result):
            with numba.objmode():
                result[0] = g('+',x,n) - g('-',x,n)
        
        return [data1,data2]

class new_imp_sol(impedance):
    def __init__(self, grid,uinc,lambda_plus,lambda_minus):
        super().__init__(grid,uinc,lambda_plus,lambda_minus)

        #create new discrete operator needed for this formulation:
        S_k_dcs2cts = bempp.api.operators.boundary.helmholtz.single_layer(self.L2_subspace, self.L2_subspace, self.C1o_subspace, self.k)

        #impedance operator, based on Dave's wwhatsapp
        A_rect = bempp.api.BlockedOperator(2, 2)
        A_rect[0,0] =  (self.lambda_plus+self.lambda_minus)/4*self.Id_cts2cts + self.T_k
        A_rect[0,1] = -(self.lambda_plus-self.lambda_minus)/2*S_k_dcs2cts
        A_rect[1,0] =  (self.lambda_plus-self.lambda_minus)/2*self.Id_cts2dcs
        A_rect[1,1] =  self.Id_dcs2dcs - (self.lambda_plus+self.lambda_minus)*self.S_k

        data = self.get_data()
        # make RHS
        rhs1 = bempp.api.GridFunction(self.L2_subspace, fun=data[0], dual_space=self.C1o_subspace)
        rhs2 = bempp.api.GridFunction(self.L2_subspace, fun=data[1], dual_space=self.L2_subspace)

        (self.Neumann_solution, self.Dirichlet_solution), _ = bempp.api.linalg.gmres(A_rect, [rhs1, rhs2])

    def get_data(self):
        g = self.get_g()

        @bempp.api.complex_callable
        def data1(x, n, domain_index, result):
            with numba.objmode():
                result[0] = (g('+',x,n) + g('-',x,n))/2

        @bempp.api.complex_callable
        def data2(x, n, domain_index, result):
            with numba.objmode():
                result[0] = g('+',x,n) - g('-',x,n)
        
        return [data1,data2]


#function which returns a cyclindrical set of points:
