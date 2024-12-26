# Copyright 2023 The GWKokab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Copyright (c) 2023 Farr Out Lab
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from ._cosmology import Cosmology


PLANCK_2015_Ho = 67.74 / (1e-3)  # Mpc
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

PLANCK_2018_Ho = 67.32 / (1e-3)  # Mpc
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
