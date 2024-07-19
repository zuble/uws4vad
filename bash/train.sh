#!/bin/bash

DATASETS=("ucf") #"xdv"
NETWORKS=("mir")
MODELS=("rank")

for dataset in "${DATASETS[@]}"; do
    # Dataset-specific configuration (if any)
    DATASET_CONFIG="${dataset}"

    # Loop through networks
    for network in "${NETWORKS[@]}"; do
        # Loop through models
        for model in "${MODELS[@]}"; do
            # Create the Hydra override string
            OVERRIDES="data=${dataset} net.id=${network} model.id=${model}"

            # Include the dataset-specific config if it exists
            if [[ -f "${DATASET_CONFIG}" ]]; then
            OVERRIDES="${OVERRIDES} +${DATASET_CONFIG}"
            fi

            # Launch the experiment using Hydra multirun
            python main.py -m ${OVERRIDES}
        done
    done
done