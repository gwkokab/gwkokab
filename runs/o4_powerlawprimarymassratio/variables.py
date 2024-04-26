#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

POSTERIOR_REGEX = (
    "/home/muhammad.zeeshan/o4a-analysis/data/o*_lower_mass/*.dat"  # set the path to the posterior files regex
)


N_CHAINS = 20

# MaskedCouplingRQSpline parameters

N_LAYERS = 5
HIDDEN_SIZE = [32, 32]
NUM_BINS = 5


# MALA Sampler parameters

STEP_SIZE = 1e-1

# Sampler parameters

N_LOOP_TRAINING = 500
N_LOOP_PRODUCTION = 500
N_LOCAL_STEPS = 150
N_GLOBAL_STEPS = 20
NUM_EPOCHS = 5

LEARNING_RATE = 0.001
MOMENTUM = 0.9
BATCH_SIZE = 5000
MAX_SAMPLES = 5000
