[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AndrewGibbs/fractal-screen-scattering/HEAD?labpath=EG_Koch.ipynb)

## fractalpp
* A collection of codes for modelling acoustic and electro-magnetic scattering by fractal screens.
* These codes use [*bempp*](http://bempp.com), but do not require an understanding of the underlying mathematics/machinery of bempp. Click the binder badge above for an interactive example for impedance scattering by a Koch snowflake, which will run in a browser window.
* Contains a graded meshing strategy for the Koch snowflake, based on the ideas of [1]. These codes do not depend on bempp, and can be found in *fractalpp/meshes/koch.py*
* For numerical analysis, see [2] (Dirichlet) and [3] (Impedance).

## Bibliography

[1]: [**Bagnerini, P., Buffa, A. and Vacca, E., 2013. Mesh generation and numerical analysis of a Galerkin method for highly conductive prefractal layers. Applied Numerical Mathematics, 65, pp.63-78.**](https://www.sciencedirect.com/science/article/abs/pii/S0168927412001936)

[2]: [**Chandler-Wilde, S.N., Hewett, D.P., Moiola, A. and Besson, J., 2021. Boundary element methods for acoustic scattering by fractal screens. Numerische Mathematik, 147(4), pp.785-837.**](https://link.springer.com/article/10.1007/s00211-021-01182-y)

[3]: [**Bannister, J., Gibbs, A. and Hewett, D.P., 2020. Acoustic scattering by impedance screens/cracks with fractal boundary: well-posedness analysis and boundary element approximation. arXiv preprint arXiv:2009.13236.**](https://arxiv.org/abs/2009.13236)
