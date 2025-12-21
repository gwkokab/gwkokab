# Copyright (c) 2021 Colm Talbot
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# Adapted from transformations derived in Iwaya et al. 2024


import jax
import numpy as np


# Relative tolerance for the series
TOL = 2.220446092504131e-16
PISQ_6 = 1.6449340668482264365
MAX_ITER = 500


def I1(chieff, chip, q):
    x1max = np.minimum(
        np.sqrt(q**2 - ((4 + 3 * q) / (3 + 4 * q)) ** 2 * chip**2),
        (1 + q) * chieff + np.sqrt(1 - chip**2),
    )
    x1min = np.maximum(
        -np.sqrt(q**2 - ((4 + 3 * q) / (3 + 4 * q)) ** 2 * chip**2),
        (1 + q) * chieff - np.sqrt(1 - chip**2),
    )
    cond1 = chip > 0
    cond2 = q >= (4 + 3 * q) * chip / (3 + 4 * q)
    cond3 = x1max >= x1min

    Fterm = F(x1max, (1 + q) * chieff, chip, (4 + 3 * q) / (3 + 4 * q) * chip, q) - F(
        x1min, (1 + q) * chieff, chip, (4 + 3 * q) / (3 + 4 * q) * chip, q
    )
    return np.where(cond1 & cond2 & cond3, (1 + q) / (8 * q) * Fterm, 0)


def I2(chieff, chip, q):
    x2max = np.minimum(q, (1 + q) * chieff + np.sqrt(1 - chip**2))
    x2min = np.maximum(-q, (1 + q) * chieff - np.sqrt(1 - chip**2))

    cond1 = chip > 0
    cond2 = chip < 1
    cond3 = x2max >= x2min

    Fterm = F(x2max, (1 + q) * chieff, chip, 0, q) - F(
        x2min, (1 + q) * chieff, chip, 0, q
    )
    return np.where(cond1 & cond2 & cond3, -(1 + q) / (8 * q) * Fterm, 0)


def I3(chieff, chip, q):
    x3max = np.minimum(
        np.sqrt(1 - chip**2),
        (1 + q) * chieff + np.sqrt(q**2 - ((4 + 3 * q) / (3 + 4 * q)) ** 2 * chip**2),
    )
    x3min = np.maximum(
        -np.sqrt(1 - chip**2),
        (1 + q) * chieff - np.sqrt(q**2 - ((4 + 3 * q) / (3 + 4 * q)) ** 2 * chip**2),
    )

    cond1 = chip > 0
    cond2 = q >= (4 + 3 * q) * chip / (3 + 4 * q)
    cond3 = x3max > x3min

    Fterm = F(x3max, (1 + q) * chieff, (4 + 3 * q) / (3 + 4 * q) * chip, chip, 1) - F(
        x3min, (1 + q) * chieff, (4 + 3 * q) / (3 + 4 * q) * chip, chip, 1
    )

    return np.where(
        cond1 & cond2 & cond3, (1 + q) / (8 * q) * (4 + 3 * q) / (3 + 4 * q) * Fterm, 0
    )


def I4(chieff, chip, q):
    x4max = np.minimum(
        1,
        (1 + q) * chieff + np.sqrt(q**2 - ((4 + 3 * q) / (3 + 4 * q)) ** 2 * chip**2),
    )
    x4min = np.maximum(
        -1,
        (1 + q) * chieff - np.sqrt(q**2 - ((4 + 3 * q) / (3 + 4 * q)) ** 2 * chip**2),
    )

    cond1 = chip > 0
    cond2 = q >= (4 + 3 * q) * chip / (3 + 4 * q)
    cond3 = x4max > x4min

    Fterm = F(x4max, (1 + q) * chieff, (4 + 3 * q) / (3 + 4 * q) * chip, 0, 1) - F(
        x4min, (1 + q) * chieff, (4 + 3 * q) / (3 + 4 * q) * chip, 0, 1
    )

    return np.where(
        cond1 & cond2 & cond3, -(1 + q) / (8 * q) * (4 + 3 * q) / (3 + 4 * q) * Fterm, 0
    )


def F(x, a, b, c, d):
    return G(x / b, a / b, c / b) + np.log(b**2 / d**2) * (
        np.arctan((x - a) / b) + np.arctan(a / b)
    )


def G(x, alpha, beta):
    pre = np.where(x >= 0, 1, -1)
    alpha = np.where(x >= 0, alpha, -alpha)
    x = np.where(x >= 0, x, -x)
    g1 = g(x, alpha, beta)
    g2 = g(x, alpha, -beta)
    g3 = -g(0, alpha, beta)
    g4 = -g(0, alpha, -beta)

    out = pre * np.imag(g1 + g2 + g3 + g4)
    return out


def g(x, alpha, beta):
    cond1 = np.abs(beta) < 1
    cond2 = (beta == 1) & (alpha <= 0)
    x_ = np.where((x == 0) & (beta == 0), 0.01, x)
    ret = np.nan_to_num(
        np.where(
            cond1,
            np.log(x_ - beta * 1j)
            * np.log((alpha - x_ + 1j) / (alpha + 1j - beta * 1j))
            + Li2((x_ - beta * 1j) / (alpha + 1j - beta * 1j)),
            np.where(
                cond2,
                0.5 * (np.log(x_ - alpha - 1j)) ** 2 + Li2(-alpha / (x_ - alpha - 1j)),
                np.log(alpha + 1j - beta * 1j) * np.log(alpha - x_ + 1j)
                - Li2((alpha - x_ + 1j) / (alpha + 1j - beta * 1j)),
            ),
        )
    )
    return np.where((x == 0) & (beta == 0), 0, ret)


def jaxspence(z):
    """From scipy.special.spence's implementation:

    Compute Spence's function for complex arguments. The strategy is:
    - If z is close to 0, use a series centered at 0.
    - If z is far away from 1, use the reflection formula

    spence(z) = -spence(z/(z - 1)) - pi**2/6 - ln(z - 1)**2/2

    to move close to 1.
    - If z is close to 1, use a series centered at 1.
    """
    return jax.lax.cond(
        np.abs(z) < 0.5,
        lambda z: cspence_series0(z),
        lambda z: jax.lax.cond(
            np.abs(1 - z) > 1,
            lambda z: -cspence_series1(z / (z - 1)) - PISQ_6 - 0.5 * np.log(z - 1) ** 2,
            lambda z: cspence_series1(z),
            z,
        ),
        z,
    )


def cspence_series0(z):
    """A series centered at z = 0; see.

    http://functions.wolfram.com/10.07.06.0005.02
    """
    z_ = np.where(z == 0, 0.01, z)

    def condition(args):
        n, zfac, term1, sum1, term2, sum2 = args
        return (
            (np.abs(term1) > TOL * np.abs(sum1)) | (np.abs(term2) > TOL * np.abs(sum2))
        ) & (n < MAX_ITER)

    def body(args):
        n, zfac, term1, sum1, term2, sum2 = args
        zfac *= z_
        term1 = zfac / n**2
        sum1 += term1
        term2 = zfac / n
        sum2 += term2
        return (n + 1, zfac, term1, sum1, term2, sum2)

    def body_fori(i, args):
        return body(args)

    n, zfac, term1, sum1, term2, sum2 = jax.lax.fori_loop(
        1, MAX_ITER, body_fori, (1, 1, 0, 0, 0, 0)
    )

    # n, zfac, term1, sum1, term2, sum2 = jax.lax.while_loop(
    # condition, body, (1, 1, np.inf, 0, np.inf, 0)
    # )
    return np.where(z == 0, PISQ_6, PISQ_6 - sum1 + np.log(z_) * sum2)


def cspence_series1(z):
    """A series centered at z = 1 which enjoys faster convergence than the Taylor
    series.

    The number of terms used comes from bounding the absolute tolerance at the edge of
    the radius of convergence where the sum is O(1).
    """

    z_ = np.where(z == 1, 1 - 1e-10, z)

    z_ = 1 - z_
    zz = z_**2

    def condition(args):
        n, zfac, res, term = args
        return (np.abs(term) > TOL * np.abs(res)) & (n < MAX_ITER)

    def body(args):
        n, zfac, res, _ = args
        zfac *= z_
        term = ((zfac / n**2) / (n + 1) ** 2) / (n + 2) ** 2
        res += term
        return (n + 1, zfac, res, term)

    def body_fori(i, args):
        return body(args)

    n, zfac, res, term = jax.lax.fori_loop(1, MAX_ITER, body_fori, (1, 1, 0, np.inf))
    # n, zfac, res, term = jax.lax.while_loop(condition, body, (1, 1, 0, np.inf))
    res *= 4 * zz
    res += 4 * z_ + 5.75 * zz + 3 * (1 - zz) * np.log(1 - z_)
    res /= 1 + 4 * z_ + zz
    return np.where(z == 1, 0.0, res)


def Li2(z):
    if "jax" in np.__name__:
        spence = jax.vmap(jaxspence)
    else:
        from scipy.special import spence
    z = np.atleast_1d(z)
    return spence(1 - z)


def prior_chieff_chip_isotropic(chieff, chip, q, amax=1):
    chieff = chieff / amax
    chip = chip / amax
    return (
        I1(chieff, chip, q)
        + I2(chieff, chip, q)
        + I3(chieff, chip, q)
        + I4(chieff, chip, q)
    ) / amax**2


def chi_effective_prior_from_isotropic_spins(chi_eff, q, amax=1):
    """Function defining the conditional priors p(chi_eff|q) corresponding to uniform,
    isotropic component spin priors.

    Taken from https://github.com/tcallister/effective-spin-priors/blob/cd5813890de043b2dc59bfaaf9f5eb7d57882641/priors.py
    with some fairly significant refactoring so the branching only depends
    on whether the two components can have opposite signs.

    Parameters
    ==========
    chi_eff: array-like
        Chi_effective value or values at which we wish to compute prior
    q: array-like
        Mass ratio value (according to the convention q<1)
    amax: array-like
        Maximum allowed dimensionless component spin magnitude

    Returns
    =======
    array-like
        The prior values
    """

    # Ensure that `xs` is an array and take absolute value
    chi_eff = np.abs(np.array(chi_eff))

    max_primary = amax / (1 + q)
    max_secondary = q * amax / (1 + q)
    max_difference = amax * (1 - q) / (1 + q)

    # Set up various piecewise cases, these are applied consecutively so lower bounds are implicit
    opposite_signs_allowed = chi_eff < max_difference
    same_sign_required = chi_eff < amax

    with np.errstate(divide="ignore", invalid="ignore"):
        secondary_ratio = max_secondary / chi_eff
        primary_ratio = max_primary / chi_eff

        lower = (
            (4 - np.log(q**2) - np.log(np.abs(1 / secondary_ratio**2 - 1)))
            + np.nan_to_num(
                np.log(np.abs(1 - secondary_ratio) / (1 + secondary_ratio))
                + (Li2(-secondary_ratio + 0j) - Li2(secondary_ratio + 0j)).real
            )
            / secondary_ratio
        ) / (4 * max_primary)

        # these terms diverge on boundaries and so we manually regularize
        primary_term = np.log(np.abs(1 / primary_ratio - 1) + (primary_ratio == 1))
        secondary_term = np.log(
            np.abs(1 / secondary_ratio - 1) + (secondary_ratio == 1)
        )

        upper = (
            2 * (amax - chi_eff)
            + max_difference * np.log(q)
            + (chi_eff - max_secondary) * secondary_term
            + (chi_eff - max_primary) * primary_term
            + chi_eff * np.log(primary_ratio) * (primary_term - np.log(q))
            + chi_eff * (Li2(1 - primary_ratio + 0j) - Li2(secondary_ratio + 0j)).real
        ) / (4 * max_primary * max_secondary)

    pdfs = np.select([opposite_signs_allowed, same_sign_required], [lower, upper], 0.0)

    return pdfs.squeeze()
