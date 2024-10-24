"""Test MinDist functions of symbolic representations."""

import itertools
import os
from warnings import simplefilter

import numpy as np
import pandas as pd
from scipy.stats import zscore

simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=UserWarning)

from numba import njit, prange

from aeon.distances._dft_sfa_mindist import dft_sfa_mindist
from aeon.distances._paa_sax_mindist import paa_sax_mindist

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from aeon.transformations.collection.dictionary_based import SAX, SFAFast



def read(fp, dim, data_type=np.float32, count=100):
    if data_type != np.float32:
        a = np.fromfile(fp, dtype=data_type, count=dim*count)
        return a.reshape(-1, dim).copy().astype(np.float32, copy=False)
    else:
        return np.fromfile(fp, dtype=np.float32, count=dim*count).reshape(-1, dim)



@njit(cache=True, fastmath=True, parallel=True)
def compute_distances(
        queries,
        samples,
        PAA_queries,
        SAX_samples,
        SAX_breakpoints,
        # sfa_transforms,
        all_breakpoints,
        all_dfts,
        all_words,
        method_names
):
    pruning_power = np.zeros((len(method_names)), dtype=np.float64)

    for i in prange(queries.shape[0]):
        # ED first
        nn_dist = np.inf
        for j in range(data.shape[0]):
            ed = np.linalg.norm(queries[i] - data[j])
            nn_dist = min(nn_dist, ed)

        for j in range(data.shape[0]):
            # SAX-PAA Min-Distance
            min_dist = paa_sax_mindist(
                PAA_queries[i], SAX_samples[j], SAX_breakpoints, samples.shape[-1]
            )
            if min_dist > nn_dist:
                pruning_power[0] += 1

            a = 0
            for sfa_breakpoints, X_dfts, Y_sfa_words in zip(all_breakpoints, all_dfts, all_words):
                # DFT-SFA Min-Distance variants
                min_dist = dft_sfa_mindist(
                    X_dfts[i], Y_sfa_words[j], sfa_breakpoints)

                if min_dist > nn_dist:
                    pruning_power[a + 1] += 1

                a = a+1

        # for i, mind in enumerate(mindist):
        #    if mind > ed:
        #        print(f"mindist {method_names[i]} is:\t {mind} but ED is: \t {ed}")

    for i in range(len(pruning_power)):
        pruning_power[i] /= data.shape[0] * queries.shape[0]

    return pruning_power


NORMAL_PATH = "/vol/tmp/schaefpa/messi_datasets/"
SEISBENCH_PATH = "/vol/tmp/schaefpa/seismic/"

datasets = {
    # Other DS
    "ASTRO": ["astro.bin", "astro_queries.bin", 256, 0, np.float32],
    "BIGANN": ["bigANN.bin", "bigANN_queries.bin", 100, 0, np.int8],
    "SALD": ["SALD.bin", "SALD_queries.bin", 128, 0, np.float32],
    "SIFT1B": ["sift1b.bin", "sift1b_queries.bin", 128, 0, np.float32],
    "DEPP1B": ["deep1b.bin", "deep1b_queries.bin", 96, 0, np.float32],
    #"SCEDC": ["SCEDC.bin", "SCEDC_queries.bin", 256, 0, np.float32],

    # Seisbench
    "ETHZ": ["ETHZ.bin", "ETHZ_queries.bin", 256, 1, np.float32],
    "ISC_EHB_DepthPhases": ["ISC_EHB_DepthPhases.bin", "ISC_EHB_DepthPhases_queries.bin", 256, 1, np.float32],
    #"LenDB": ["LenDB.bin", "LenDB_queries.bin", 256, 1, np.float32],
    "Iquique": ["Iquique.bin", "Iquique_queries.bin", 256, 1, np.float32],
    "NEIC": ["NEIC.bin", "NEIC_queries.bin", 256, 1, np.float32],
    "OBS": ["OBS.bin", "OBS_queries.bin", 256, 1, np.float32],
    "OBST2024": ["OBST2024.bin", "OBST2024_queries.bin", 256, 1, np.float32],
    "PNW": ["PNW.bin", "PNW_queries.bin", 256, 1, np.float32],
    "Meier2019JGR": ["Meier2019JGR.bin", "Meier2019JGR_queries.bin", 256, 1, np.float32],
    "STEAD": ["STEAD.bin", "STEAD_queries.bin", 256, 1, np.float32],
    "TXED": ["TXED.bin", "TXED_queries.bin", 256, 1, np.float32],
}

all_threads = 36
n_segments = 16
alphabet_sizes = [2, 4, 8, 16, 32, 64, 128, 256]
all_csv_scores = {}

for alphabet_size in alphabet_sizes:
    all_csv_scores[alphabet_size] = []

for dataset in datasets:
    for alphabet_size in alphabet_sizes:
        csv_scores = all_csv_scores[alphabet_size]

        print("Running: ", dataset, alphabet_size)
        file_data, file_queries, d, path_switch, data_type = datasets[dataset]

        path = NORMAL_PATH if path_switch == 0 else SEISBENCH_PATH
        data = read(path + file_data, dim=d, data_type=data_type, count=100_000)  # data
        queries = read(path + file_queries, data_type=data_type, dim=d, count=100)  # queries

        print("\tDataset\t", dataset)
        print("\tData Shape\t", data.shape)
        print("\tQuery Shape\t", queries.shape)

        # ignore all future warnings
        simplefilter(action="ignore", category=FutureWarning)
        simplefilter(action="ignore", category=UserWarning)

        samples = samples[np.std(samples, axis=-1) > 1e-8]
        queries = queries[np.std(queries, axis=-1) > 1e-8]

        samples = (samples - np.mean(samples, axis=-1, keepdims=True)) / (
            np.std(samples, axis=-1, keepdims=True)
        )
        queries = (queries - np.mean(queries, axis=-1, keepdims=True)) / (
            np.std(queries, axis=-1, keepdims=True)
        )

        np.nan_to_num(samples, nan=0, copy=False)
        np.nan_to_num(queries, nan=0, copy=False)

        SAX_transform = SAX(n_segments=n_segments, alphabet_size=alphabet_size)
        SAX_samples = SAX_transform.fit_transform(samples).squeeze()
        PAA_queries = SAX_transform._get_paa(queries).squeeze()
        print("\tSAX done.")

        histograms = ["equi-width", "equi-depth"]
        variances = [True, False]

        method_names = ["isax"]
        all_breakpoints = []
        all_dfts = []
        all_words = []

        for histogram, variance in itertools.product(histograms, variances):
            sfa = SFAFast(
                word_length=n_segments,
                alphabet_size=alphabet_size,
                window_size=data.shape[-1],
                binning_method=histogram,
                norm=True,
                variance=variance,
                lower_bounding_distances=True,
                n_jobs=all_threads
            )

            sfa.fit(data)
            X_dfts = sfa.transform_mft(queries).squeeze()
            Y_words = sfa.transform_words(samples).squeeze()
            all_breakpoints.append(sfa.breakpoints.astype(np.float64))
            all_dfts.append(X_dfts.astype(np.float64))
            all_words.append(Y_words.astype(np.int32))

            method_names.append(f"sfa_{histogram}_{variance}")
            print(f"\tSFA {histogram} {variance} done.")

        sum_scores = {}
        for method_name in method_names:
            sum_scores[method_name] = {
                "dataset": [],
                "pruning_power": [],
            }

        pruning_power = compute_distances(
            queries,
            samples,
            PAA_queries,
            SAX_samples,
            SAX_transform.breakpoints,
            # sfa_transforms,
            np.array(all_breakpoints),
            np.array(all_dfts),
            np.array(all_words),
            np.array(method_names)
        )

        for a, method_name in enumerate(method_names):
            sum_scores[method_name]["dataset"].append(dataset)
            sum_scores[method_name]["pruning_power"].append(pruning_power[a])
            csv_scores.append((method_name, dataset, pruning_power[a]))

        print(f"\n\n---- Results using {alphabet_size}-----")
        for name, _ in sum_scores.items():
            print(f"---- Name {name}, \t PrP: {np.round(sum_scores[name]["pruning_power"], 3)}")

        # if server:
        pd.DataFrame.from_records(
            csv_scores,
            columns=[
                "Method",
                "Dataset",
                "Pruning_Power",
            ],
        ).to_csv(
            f"logs/pp_all_sofa_bench_{n_segments}_{alphabet_size}-24-10-24.csv",
            index=None)
