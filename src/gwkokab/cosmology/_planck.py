# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


# Copyright (c) 2023 Farr Out Lab
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from ..constants import H0_SI
from ._cosmology import Cosmology


PLANCK_2015_Ho = 67.74 * H0_SI  # Hubble constant in s^-1
PLANCK_2015_OmegaMatter = 0.3089
PLANCK_2015_OmegaLambda = 1.0 - PLANCK_2015_OmegaMatter
PLANCK_2015_OmegaRadiation = 0.0

PLANCK_2015_Cosmology = Cosmology(
    PLANCK_2015_Ho,
    PLANCK_2015_OmegaMatter,
    PLANCK_2015_OmegaRadiation,
    PLANCK_2015_OmegaLambda,
)
"Cosmology : See Table4 in arXiv:1502.01589, OmegaMatter from astropy Planck 2015"

PLANCK_2018_Ho = 67.32 * H0_SI  # Hubble constant in s^-1
PLANCK_2018_OmegaMatter = 0.3158
PLANCK_2018_OmegaLambda = 1.0 - PLANCK_2018_OmegaMatter
PLANCK_2018_OmegaRadiation = 0.0

PLANCK_2018_Cosmology = Cosmology(
    PLANCK_2018_Ho,
    PLANCK_2018_OmegaMatter,
    PLANCK_2018_OmegaRadiation,
    PLANCK_2018_OmegaLambda,
)
"Cosmology : See Table1 in arXiv:1807.06209"
