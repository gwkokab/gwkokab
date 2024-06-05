from __future__ import annotations

from numpyro import distributions as dist

import gwkokab.population as gwk_pop
from gwkokab.errors import banana_error_m1_m2


models = [
    {
        gwk_pop.ModelMeta.NAME: dist.Normal,
        gwk_pop.ModelMeta.OUTPUT: [gwk_pop.Parameter.PRIMARY_MASS],
        gwk_pop.ModelMeta.PARAMETERS: {
            "loc": 80,
            "scale": 2,
            "validate_args": True,
        },
        gwk_pop.ModelMeta.SAVE_AS: {
            "loc": "m1_loc",
            "scale": "m1_scale",
        },
    },
    {
        gwk_pop.ModelMeta.NAME: dist.Normal,
        gwk_pop.ModelMeta.OUTPUT: [gwk_pop.Parameter.SECONDARY_MASS],
        gwk_pop.ModelMeta.PARAMETERS: {
            "loc": 30,
            "scale": 2,
            "validate_args": True,
        },
        gwk_pop.ModelMeta.SAVE_AS: {
            "loc": "m2_loc",
            "scale": "m2_scale",
        },
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
        )
    ],
)


popfactory.generate_realizations()
gwk_pop.run_noise_factory(noisepopinfo)
