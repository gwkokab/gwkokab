import os


env_vars = {
    "JAX_PLATFORMS": "cpu",
    "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}

for name, value in env_vars.items():
    os.environ[name] = value


import gc

import tqdm

from gwkokab.analysis.n_pls_m_gs.common import NPowerlawMGaussianCore
from gwkokab.analysis.utils.marginals import generate_marginal_probs
from gwkokab.parameters import Parameters as P


if __name__ == "__main__":
    analyses_dirs = [
        "analytical_flowMC",
        "analytical_numpyro",
        "discrete_flowMC",
        "discrete_numpyro",
    ]

    for analyses_dir in tqdm.tqdm(analyses_dirs):
        generate_marginal_probs(
            model_meta_cls=NPowerlawMGaussianCore,
            input_file_path=f"{analyses_dir}/inference_data.hdf5",
            output_file_path=f"{analyses_dir}/marginal_probs.hdf5",
            domain_cfg={
                P.PRIMARY_MASS_SOURCE: (1.0, 60.0, 500),
                P.SECONDARY_MASS_SOURCE: (1.0, 60.0, 500),
                P.PRIMARY_SPIN_Z: (-1.0, 1.0, 200),
                P.SECONDARY_SPIN_Z: (-1.0, 1.0, 200),
                P.ECCENTRICITY: (0.0, 1.0, 200),
            },
            max_samples=10_000,
            batch_size=100,
        )
        gc.collect()
