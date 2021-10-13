import fractalpp.solvers.impedance_solvers, numpy

class u_i:
    def __init__(self,k,incDir):
        self.k = float(k)
        self.incDir = numpy.array(incDir)

def impedance(grid,uinc,lambda_plus,lambda_minus,classical=False):
    if classical: #flag to use classical impedance solver
        return fractalpp.solvers.impedance_solvers.imp_sol(grid,uinc,lambda_plus,lambda_minus)
    else:
        return fractalpp.solvers.impedance_solvers.new_imp_sol(grid,uinc,lambda_plus,lambda_minus)