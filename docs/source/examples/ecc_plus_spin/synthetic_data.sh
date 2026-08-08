#!/bin/bash

JAX_PLATFORMS=cpu \
	XLA_PYTHON_CLIENT_ALLOCATOR=platform \
	XLA_PYTHON_CLIENT_PREALLOCATE=false \
	GWKOKAB_LOG_FILE="synthetic_events.log" \
	synthetic_events_n_pls_m_gs \
	--seed $RANDOM \
	--n-pl 1 \
	--n-g 0 \
	--pmean-cfg pmean_cfg.json \
	--model-params model.json \
	--add-truncated-normal-spin-z \
	--add-eccentricity-mixture \
	--derive-parameters

JAX_PLATFORMS=cpu \
	XLA_PYTHON_CLIENT_ALLOCATOR=platform \
	XLA_PYTHON_CLIENT_PREALLOCATE=false \
	GWKOKAB_LOG_FILE="discrete.log" \
	synthetic_discrete_pe \
	--seed $RANDOM \
	--filename synthetic_events.hdf5 \
	--error-params error_params.json \
	--size 5000 \
	--coords=chirp_mass,symmetric_mass_ratio,spin_1z,spin_2z,eccentricity \
	--derive-parameters

for event_file in $(ls data/event_*.hdf5); do

	JAX_PLATFORMS=cpu \
		XLA_PYTHON_CLIENT_ALLOCATOR=platform \
		XLA_PYTHON_CLIENT_PREALLOCATE=false \
		GWKOKAB_LOG_FILE="$event_file.log" \
		synthetic_analytical_gwalk_pe "$event_file" \
		--coords=mass_1_source,mass_2_source,spin_1z,spin_2z,eccentricity \
		--seed $RANDOM
done
