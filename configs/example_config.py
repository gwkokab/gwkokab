from __future__ import annotations

from jax import random as jrd
from numpyro import distributions as dist

import gwkokab.models as gwk_model
import gwkokab.population as gwk_pop
from gwkokab.errors import banana_error_m1_m2


models = [
    {
        gwk_pop.ModelMeta.NAME: gwk_model.PowerLawPrimaryMassRatio,
        gwk_pop.ModelMeta.OUTPUT: [
            gwk_pop.Parameter.PRIMARY_MASS,
            gwk_pop.Parameter.SECONDARY_MASS,
        ],
        gwk_pop.ModelMeta.PARAMETERS: {
            "alpha": 0.5,
            "beta": 1.0,
            "mmin": 10.0,
            "mmax": 50.0,
        },
        gwk_pop.ModelMeta.SAVE_AS: {"alpha": "alpha", "beta": "beta", "mmin": "mass_min", "mmax": "mass_max"},
    },
    {
        gwk_pop.ModelMeta.NAME: dist.Beta,
        gwk_pop.ModelMeta.OUTPUT: [gwk_pop.Parameter.PRIMARY_ALIGNED_SPIN],
        gwk_pop.ModelMeta.PARAMETERS: {"concentration1": 1.8, "concentration0": 0.9, "validate_args": True},
        gwk_pop.ModelMeta.SAVE_AS: {
            "concentration1": "alpha_1",
            "concentration0": "beta_1",
        },
    },
    {
        gwk_pop.ModelMeta.NAME: dist.Beta,
        gwk_pop.ModelMeta.OUTPUT: [gwk_pop.Parameter.SECONDARY_ALIGNED_SPIN],
        gwk_pop.ModelMeta.PARAMETERS: {"concentration1": 1.8, "concentration0": 0.9, "validate_args": True},
        gwk_pop.ModelMeta.SAVE_AS: {
            "concentration1": "alpha_2",
            "concentration0": "beta_2",
        },
    },
    {
        gwk_pop.ModelMeta.NAME: dist.HalfNormal,
        gwk_pop.ModelMeta.OUTPUT: [gwk_pop.Parameter.ECCENTRICITY],
        gwk_pop.ModelMeta.PARAMETERS: {"scale": 0.05, "validate_args": True},
        gwk_pop.ModelMeta.SAVE_AS: {"scale": "sigma_ecc"},
    },
]


popinfo = gwk_pop.PopInfo(
    ROOT_DIR=r"syn_data",
    EVENT_FILENAME="event_{}",
    CONFIG_FILENAME="configuration",
    RATE=1e6,
    NUM_REALIZATIONS=5,
    VT_FILE=r"/media/gradf/Academic/project/jaxtro/neural_vt_0.5_200_1day_SimNoisePSDaLIGO175MpcT1800545_IMRPhenomD_snr10.eqx",
    VT_PARAMS=[gwk_pop.Parameter.PRIMARY_MASS, gwk_pop.Parameter.SECONDARY_MASS],
    TIME=365.25,
)


header = []

for output in models:
    header.extend(output[gwk_pop.ModelMeta.OUTPUT])

popfactory = gwk_pop.PopulationFactory(
    models,
    popinfo,
    seperate_injections=True,
)


noisepopinfo = gwk_pop.NoisePopInfo(
    FILENAME_REGEX=r"/media/gradf/Academic/project/jaxtro/syn_data/realization_0/injections/event_*.dat",
    OUTPUT_DIR=r"/media/gradf/Academic/project/jaxtro/syn_data/realization_0/posteriors/event_{}.dat",
    HEADER=header,
    SIZE=4000,
    ERROR_FUNCS=[
        (
            (gwk_pop.Parameter.PRIMARY_MASS, gwk_pop.Parameter.SECONDARY_MASS),
            lambda x, size, key: banana_error_m1_m2(x, size, key, scale_Mc=1.0, scale_eta=1.0),
        ),
        (
            (gwk_pop.Parameter.PRIMARY_ALIGNED_SPIN,),
            lambda x, size, key: jrd.truncated_normal(key=key, lower=0.0, upper=1.0, shape=(size,)) * 0.5 + x,
        ),
        (
            (gwk_pop.Parameter.SECONDARY_ALIGNED_SPIN,),
            lambda x, size, key: jrd.truncated_normal(key=key, lower=0.0, upper=1.0, shape=(size,)) * 0.5 + x,
        ),
        (
            (gwk_pop.Parameter.ECCENTRICITY,),
            lambda x, size, key: jrd.truncated_normal(key=key, lower=0.0, upper=1.0, shape=(size,)) * 0.01 + x,
        ),
    ],
)


popfactory.generate_realizations()
gwk_pop.run_noise_factory(noisepopinfo)
