#!/bin/bash

XLA_PYTHON_CLIENT_ALLOCATOR=platform \
	XLA_PYTHON_CLIENT_PREALLOCATE=false \
	JAX_COMPILATION_CACHE_DIR="$HOME/jax_cache" \
	GWKOKAB_LOG_FILE="discrete.log" \
	discrete_n_pls_m_gs \
	--n-pl 1 \
	--n-g 0 \
	--seed $RANDOM \
	--n-buckets 4 \
	--data-loader-cfg "../data_loader_cfg.json" \
	--prior-cfg "../prior_cfg.json" \
	--pmean-cfg "../pmean_cfg.json" \
	--sampler-cfg "./sampler_cfg.json" \
	--add-truncated-normal-spin-z \
	--add-eccentricity-mixture

gwk_report -i inference_data.hdf5 -o discrete_flowMC_report.html
