import h5py
import numpy as np

from constants import (
    N_VOXELS_L0,
    N_VOXELS_L1,
    N_VOXELS_L2,
    N_VOXELS_L3,
    N_VOXELS_L12,
    N_VOXELS,
    ETOT_EINC_SCALE,
)


def preprocess(file_name):
    """
    Build training arrays:
      - features: per-voxel ratios (concat), (Etot/Einc)/ETOT_EINC_SCALE, layer fractions
      - condition: normalized incident energy
    """
    # Read the HDF5 file
    h5_file = h5py.File(file_name)
    incident_energies = np.array(h5_file["incident_energies"])
    showers = np.array(h5_file["showers"])

    max_energy = np.max(incident_energies)
    training_data = []
    training_condition = []

    # Loop over the events and build the ratios
    for event in range(len(incident_energies)):
        current_event = []

        # Per-layer voxel energy vectors (photons use L0, L1, L2, L3, L12)
        energy_voxels_l0 = showers[event][:N_VOXELS_L0]
        energy_voxels_l1 = showers[event][N_VOXELS_L0 : N_VOXELS_L0 + N_VOXELS_L1]
        energy_voxels_l2 = showers[event][
            N_VOXELS_L0 + N_VOXELS_L1 : N_VOXELS_L0 + N_VOXELS_L1 + N_VOXELS_L2]
        energy_voxels_l3 = showers[event][
            N_VOXELS_L0 + N_VOXELS_L1 + N_VOXELS_L2 : N_VOXELS_L0 + N_VOXELS_L1 + N_VOXELS_L2 + N_VOXELS_L3]
        energy_voxels_l12 = showers[event][
            N_VOXELS_L0 + N_VOXELS_L1 + N_VOXELS_L2 + N_VOXELS_L3 : N_VOXELS_L0 + N_VOXELS_L1 + N_VOXELS_L2 + N_VOXELS_L3 + N_VOXELS_L12]

        # Per-layer energy sums
        energy_sum_l0 = np.sum(energy_voxels_l0)
        energy_sum_l1 = np.sum(energy_voxels_l1)
        energy_sum_l2 = np.sum(energy_voxels_l2)
        energy_sum_l3 = np.sum(energy_voxels_l3)
        energy_sum_l12 = np.sum(energy_voxels_l12)

        total_energy = (
            energy_sum_l0
            + energy_sum_l1
            + energy_sum_l2
            + energy_sum_l3
            + energy_sum_l12
        )

        # Per-voxel ratios per layer
        current_event.append(energy_voxels_l0 / energy_sum_l0)
        current_event.append(energy_voxels_l1 / energy_sum_l1)
        current_event.append(energy_voxels_l2 / energy_sum_l2)
        current_event.append(energy_voxels_l3 / energy_sum_l3)
        current_event.append(energy_voxels_l12 / energy_sum_l12)

        # Scaled Etot / Einc  (so target lies ~ in [0,1])
        current_event.append((total_energy / incident_energies[event]) / ETOT_EINC_SCALE)

        # Per-layer energy fractions (photons have 5 layers)
        current_event.append(energy_sum_l0 / total_energy)
        current_event.append(energy_sum_l1 / total_energy)
        current_event.append(energy_sum_l2 / total_energy)
        current_event.append(energy_sum_l3 / total_energy)
        current_event.append(energy_sum_l12 / total_energy)

        # Stack and sanitize NaNs/Infs if any zero-sum layers appear
        training_data.append(np.nan_to_num(np.hstack(current_event)))
        training_condition.append(np.log2(incident_energies[event]) / np.log2(max_energy)) #APPLIED log scaling so training energies are linear

    training_data = np.array(training_data)        # (n, 368 + 1 + 5) = (n, 374)
    training_condition = np.array(training_condition)

    # close file and clean up
    h5_file.close()
    del incident_energies
    del showers
    return training_data, training_condition


def postprocess(predicted_energies, incident_energies):
    """
    Inverse transform predicted vectors back to voxel energies (MeV) for photons.
    Input layout:
      [ voxel ratios (N_VOXELS),
        (Etot/Einc)/ETOT_EINC_SCALE (1),
        layer fractions (5: L0,L1,L2,L3,L12) ]
    """
    predicted_energies_rescaled = []
    for event in range(len(predicted_energies)):
        # Total deposited energy = (Etot/Einc)*scale * Einc
        total_energy = (
            predicted_energies[event][N_VOXELS : N_VOXELS + 1]
            * incident_energies[event]
            * ETOT_EINC_SCALE
        )

        # Per-layer energy sums from the stored fractions
        energy_sum_l0 = (
            predicted_energies[event][N_VOXELS + 1 : N_VOXELS + 2] * total_energy
        )
        energy_sum_l1 = (
            predicted_energies[event][N_VOXELS + 2 : N_VOXELS + 3] * total_energy
        )
        energy_sum_l2 = (
            predicted_energies[event][N_VOXELS + 3 : N_VOXELS + 4] * total_energy
        )
        energy_sum_l3 = (
            predicted_energies[event][N_VOXELS + 4 : N_VOXELS + 5] * total_energy
        )
        energy_sum_l12 = (
            predicted_energies[event][N_VOXELS + 5 : N_VOXELS + 6] * total_energy
        )

        # Reconstruct per-voxel energies using ratios * per-layer sums
        energy_voxels_l0 = predicted_energies[event][:N_VOXELS_L0] * energy_sum_l0
        energy_voxels_l1 = (
            predicted_energies[event][N_VOXELS_L0 : N_VOXELS_L0 + N_VOXELS_L1]
            * energy_sum_l1
        )
        energy_voxels_l2 = (
            predicted_energies[event][
                N_VOXELS_L0 + N_VOXELS_L1 : N_VOXELS_L0 + N_VOXELS_L1 + N_VOXELS_L2
            ]
            * energy_sum_l2
        )
        energy_voxels_l3 = (
            predicted_energies[event][
                N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2 : N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3
            ]
            * energy_sum_l3
        )
        energy_voxels_l12 = (
            predicted_energies[event][
                N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3 : N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3
                + N_VOXELS_L12
            ]
            * energy_sum_l12
        )

        event_i = []
        event_i.append(energy_voxels_l0)
        event_i.append(energy_voxels_l1)
        event_i.append(energy_voxels_l2)
        event_i.append(energy_voxels_l3)
        event_i.append(energy_voxels_l12)
        event_i = np.concatenate(event_i)
        predicted_energies_rescaled.append(event_i)

    predicted_energies_rescaled = np.array(predicted_energies_rescaled)
    return predicted_energies_rescaled


def load_incident_energies(file_name):
    """Returns (incident_energies, max_energy)."""
    h5_file = h5py.File(file_name)
    incident_energies = np.array(h5_file["incident_energies"])
    max_energy = np.max(incident_energies)
    h5_file.close()
    return incident_energies, max_energy
