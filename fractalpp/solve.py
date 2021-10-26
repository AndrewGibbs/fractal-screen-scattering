import fractalpp.solvers.impedance_solvers, fractalpp.solvers.dirichlet_solvers, numpy

class u_i:
    def __init__(self,k,incDir):
        self.k = float(k)
        self.incDir = numpy.array(incDir)

def impedance(grid,uinc,lambda_plus,lambda_minus,Hassen=False):
    if Hassen: #flag to use classical impedance solver
        return fractalpp.solvers.impedance_solvers.imp_sol(grid,uinc,lambda_plus,lambda_minus)
    else:
        return fractalpp.solvers.impedance_solvers.new_imp_sol(grid,uinc,lambda_plus,lambda_minus)

def maxwell(grid,uinc):
    return fractalpp.solvers.maxwell_solvers.max_sol(grid,uinc)

def dirichlet(grid,uinc):
    return fractalpp.solvers.dirichlet_solvers.dir_sol(grid,uinc)