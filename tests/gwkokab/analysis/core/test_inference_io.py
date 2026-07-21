import os
import tempfile

from gwkokab.analysis.core.inference_io import (
    FlowMCGlobalConfig,
    NumpyroGlobalConfig,
    SamplerConfig,
)


def test_numpyro_global_config_hdf5_roundtrip():
    config = NumpyroGlobalConfig()
    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = os.path.join(tmpdir, "numpyro_cfg.h5")
        config.write_to_hdf5(hdf5_path)

        # Read back using specific class method
        loaded_cfg = NumpyroGlobalConfig.read_from_hdf5(hdf5_path)
        assert loaded_cfg.sampler_name == "numpyro"
        assert loaded_cfg.kernel.max_tree_depth == config.kernel.max_tree_depth
        assert loaded_cfg.mcmc.num_samples == config.mcmc.num_samples

        # Read back using SamplerConfig factory
        factory_cfg = SamplerConfig.read_from_hdf5(hdf5_path)
        assert isinstance(factory_cfg, NumpyroGlobalConfig)
        assert factory_cfg.sampler_name == "numpyro"


def test_flowmc_global_config_hdf5_roundtrip():
    config = FlowMCGlobalConfig()
    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = os.path.join(tmpdir, "flowmc_cfg.h5")
        config.write_to_hdf5(hdf5_path, n_dims=5)

        # Read back using specific class method
        loaded_cfg = FlowMCGlobalConfig.read_from_hdf5(hdf5_path)
        assert loaded_cfg.sampler_name == "flowMC"
        assert loaded_cfg.n_chains == config.n_chains
        assert loaded_cfg.n_local_steps == config.n_local_steps

        # Read back using SamplerConfig factory
        factory_cfg = SamplerConfig.read_from_hdf5(hdf5_path)
        assert isinstance(factory_cfg, FlowMCGlobalConfig)
        assert factory_cfg.sampler_name == "flowMC"
