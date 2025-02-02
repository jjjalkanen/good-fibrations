"""
    Explore polynomials which are continuous or differentiable
    across polynomial boundary curves.
"""
from functools import wraps
import time

import sympy as sp
from sympy.abc import x, y
from sympy import MatrixSymbol, Matrix, MatMul

import ringkit as rk

sp.init_printing(use_unicode=True)

def time_it(func):
    """ Well-known decorator for timing function calls. """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f'{func.__name__} took {time.time() - start}')
        return result
    return wrapper

# Parametric equations for boundary curve segments.
# Let's use simple 3rd order boundaries
# which are expressible with just x, y.

## Having shape parameter a would require special handling
# right_side = (x, 1 - a * y * (y**2 - 1))
# top_shelf = (y, 1 + a * x * (x**2 - 1))
# left_side = (x, -1 - a * y * (y**2 - 1))
# bottom_shelf = (y, -1 + a * x * (x**2 - 1))

# We just set parameter a to 1/6.
right_side = (x, 1 - y * (y**2 - 1)/6, y)
top_shelf = (y, 1 + x * (x**2 - 1)/6, x)
left_side = (x, -1 - y * (y**2 - 1)/6, y)
bottom_shelf = (y, -1 + x * (x**2 - 1)/6, x)

boundary_curves = (right_side, top_shelf, left_side, bottom_shelf)

# 22 would yield precisely a 484 equations for 484 unknowns
# in the continuous and differentiable case
MAXDEG = 6

bc_options = {
    'continuous': True,
    'differentiable': False,
}

def coefficients_to_poly(coefficient_matrix, one_variable_max_degree):
    """
        Given an n times n matrix of coefficients C, return
        polynomial (x^(n-1), ..., x, 1)*C*(y^(n-1), ..., y, 1)
    """
    deg = one_variable_max_degree
    assert coefficient_matrix.shape == (deg, deg)

    px = Matrix([x**(deg - i - 1) for i in range(deg)]).T
    py = Matrix([y**(deg - i - 1) for i in range(deg)])
    result_1x1_matrix = MatMul(px, coefficient_matrix, py)

    return result_1x1_matrix.as_explicit()[0, 0]

# Create a polynomial with max multidegree pdim - 1
C = MatrixSymbol('C', MAXDEG, MAXDEG)
base_polynomial = coefficients_to_poly(C, MAXDEG)

def form_equations(base_poly, poly_boundaries, continuous=True, differentiable=False):
    """
        Substitutes boundary curves to a given polynomial and its derivatives,
        and returns expressions which must vanish if the polynomial
        is to be continuous or differentiable or both across the boundaries.
    """

    def subs_boundaries(what, boundaries):
        """
            On each boundary, after substitution of the parametric
            boundary curve representation, the polynomial
            depends only on the parameters.
        """
        expanded_what = sp.expand(what)
        def subs_powers(symb, with_what):
            terms = sp.collect(expanded_what, symb, evaluate=False)
            return sum(v*(with_what**sp.degree(k, gen=symb)) for k, v in terms.items())
        return [sp.collect(sp.expand(subs_powers(bc[0], bc[1])), bc[2]) for bc in boundaries]

    @time_it
    def timed_subs_boundaries(what, boundaries):
        return subs_boundaries(what, boundaries)

    def define_obstructions(substituted_poly):
        def collect_nonzero_terms(idx1, idx2, var):
            return (v for v in sp.collect(
                substituted_poly[idx1] - substituted_poly[idx2],
                var,
                evaluate=False).values() if not v.is_zero)
        return tuple((*collect_nonzero_terms(0, 2, y), *collect_nonzero_terms(1, 3, x)))

    @time_it
    def timed_define_obstructions(substituted_poly):
        return define_obstructions(substituted_poly)

    result = []

    ## Require that across matching boundaries, the polynomial is...
    if continuous:
        value_at_boundary = timed_subs_boundaries(base_poly, poly_boundaries)
        result += timed_define_obstructions(value_at_boundary)

    if differentiable:
        # Find derivatives of the polynomial wrt x and y
        dx = sp.diff(base_poly, x)
        dx_at_boundary = timed_subs_boundaries(dx, poly_boundaries)
        result += timed_define_obstructions(dx_at_boundary)

        dy = sp.diff(base_poly, y)
        dy_at_boundary = timed_subs_boundaries(dy, poly_boundaries)
        result += define_obstructions(dy_at_boundary)

    return result

@time_it
def timed_form_equations(*args, **kwargs):
    """ Timed version of form equations. """
    return form_equations(*args, **kwargs)

obstruction_eqs = timed_form_equations(base_polynomial, boundary_curves, **bc_options)

# Create a matrix where for each equation, there is a row
# whose columns are the polynomials multiplying the unknown polynomial coefficients.

nr, nc = C.shape
coeffs = tuple(C[i, j] for i in range(nr) for j in range(nc))

eqs_mat = Matrix([[val.coeff(ci, 1) for ci in coeffs] for val in obstruction_eqs])

print('degree', MAXDEG, 'equations shape', eqs_mat.shape)
assert eqs_mat.shape[1] >= eqs_mat.shape[0]

@time_it
def timed_ringkit_nullspace(matx):
    """ Timed version of ringkit's nullspace function. """
    return rk.nullspace(matx, rk.DType.BigInt)

def timed_sympy_nullspace(matx):
    """ Timed version of sympy's nullspace function. """
    return sp.Matrix.hstack(*matx.nullspace()).T

nullspace_rk = timed_ringkit_nullspace(eqs_mat)
print("Nullspace shape (ringkit):", nullspace_rk.shape)
assert nullspace_rk.cols == MAXDEG * MAXDEG
nullspace_dim = nullspace_rk.rows

## Compare with sympy
# nullspace_sp = timed_sympy_nullspace(eqs_mat)
# print("Nullspace shape (sympy):"nullspace_sp.shape)
# assert nullspace_rk.shape == nullspace_sp.shape

k = MatrixSymbol('k', nullspace_dim, 1)

# It should also be zero for all returned basis vectors individually
# k = sp.Matrix(sp.ZeroMatrix(nsp_base_count, 1))
# k[0, 0] = sp.sympify(1)

C_res = [sum(sp.simplify(nullspace_rk[i, j])*k[i, 0]
             for i in range(nullspace_dim)) for j in range(MAXDEG * MAXDEG)]
C_res_mat = Matrix([[C_res[i*nc + j] for j in range(nc)] for i in range(nr)])

base_polynomial_res = coefficients_to_poly(C_res_mat, MAXDEG)

# # After substitution, we should get nothing
eqs_res = timed_form_equations(base_polynomial_res, boundary_curves, **bc_options)

assert not eqs_res  # No nontrivial equations should remain
