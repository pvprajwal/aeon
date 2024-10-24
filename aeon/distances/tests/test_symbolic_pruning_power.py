"""Test MinDist functions of symbolic representations."""

import itertools
import os
from warnings import simplefilter

import numpy as np
import pandas as pd
from scipy.stats import zscore

simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=UserWarning)

from joblib import Parallel, delayed, parallel_backend

from aeon.distances._dft_sfa_mindist import dft_sfa_mindist
from aeon.distances._paa_sax_mindist import paa_sax_mindist

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from aeon.transformations.collection.dictionary_based import SAX, SFAFast

dataset_names_full = [
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "BME",
    "Car",
    "CBF",
    "Chinatown",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "Crop",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    "DodgerLoopDay",
    "DodgerLoopGame",
    "DodgerLoopWeekend",
    "Earthquakes",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "EthanolLevel",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FordA",
    "FordB",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    "Fungi",
    "GestureMidAirD1",
    "GestureMidAirD2",
    "GestureMidAirD3",
    "GesturePebbleZ1",
    "GesturePebbleZ2",
    "GunPoint",
    "GunPointAgeSpan",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
    "Ham",
    "HandOutlines",
    "Haptics",
    "Herring",
    "HouseTwenty",
    "InlineSkate",
    "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "Meat",
    "MedicalImages",
    # "MelbournePedestrian",            # iSAX mindist error?
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW",
    "MixedShapesRegularTrain",
    "MixedShapesSmallTrain",
    "MoteStrain",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OliveOil",
    "OSULeaf",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "PickupGestureWiimoteZ",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    "PLAID",
    "Plane",
    "PowerCons",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "Rock",
    "ScreenType",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    "ShakeGestureWiimoteZ",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    # "SmoothSubspace",         # SFA mindist error?
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "StarLightCurves",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UMD",
    "UWaveGestureLibraryAll",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga",
]

dataset_names = [
    # "Chinatown"
    "ArrowHead",
    "Beef",
    # "BeetleFly",
    # "BirdChicken",
    # "Car",
    # "CBF",
    # "Coffee",
    # "DiatomSizeReduction",
    # "DistalPhalanxOutlineAgeGroup",
    # "DistalPhalanxOutlineCorrect",
    # "DistalPhalanxTW",
    # "ECG200",
    # "ECGFiveDays",
    # "FaceAll",
    # "FaceFour",
    # "FacesUCR",
    # "GunPoint",
    # "ItalyPowerDemand",
    # "MiddlePhalanxOutlineAgeGroup",
    # "MiddlePhalanxOutlineCorrect",
    # "MiddlePhalanxTW",
    # "OliveOil",
    # "Plane",
    # "ProximalPhalanxOutlineAgeGroup",
    # "ProximalPhalanxOutlineCorrect",
    # "ProximalPhalanxTW",
    # "SonyAIBORobotSurface1",
    # "SonyAIBORobotSurface2",
    # "SyntheticControl",
    # "TwoLeadECG",
    # "Wine",
]


def load_from_ucr_tsv_to_dataframe_plain(full_file_path_and_name):
    """Load UCR datasets."""
    df = pd.read_csv(
        full_file_path_and_name,
        sep=r"\s+|\t+|\s+\t+|\t+\s+",
        engine="python",
        header=None,
    )
    y = df.pop(0).values
    df.columns -= 1
    return df, y


DATA_PATH = "/Users/bzcschae/workspace/UCRArchive_2018/"
server = False

n_segments = 16
alphabet_sizes = [4,8,16,32,64,128,256]

if os.path.exists(DATA_PATH):
    parallel_jobs = 8
    DATA_PATH = "/Users/bzcschae/workspace/UCRArchive_2018/"
    used_dataset = dataset_names
# server
else:
    DATA_PATH = "/vol/fob-wbib-vol2/wbi/schaefpa/sktime/datasets/UCRArchive_2018"
    parallel_jobs = 80
    server = True
    used_dataset = dataset_names_full

if __name__ == "__main__":

    def _parallel_tlb(dataset_name, n_segments, alphabet_size):
        print(dataset_name)

        # ignore all future warnings
        simplefilter(action="ignore", category=FutureWarning)
        simplefilter(action="ignore", category=UserWarning)

        X_train, y_train = load_from_ucr_tsv_to_dataframe_plain(
            os.path.join(DATA_PATH, dataset_name, dataset_name + "_TRAIN.tsv")
        )
        X_test, y_test = load_from_ucr_tsv_to_dataframe_plain(
            os.path.join(DATA_PATH, dataset_name, dataset_name + "_TEST.tsv")
        )

        X_train.fillna(0, inplace=True)
        X_test.fillna(0, inplace=True)

        X_train = zscore(X_train.squeeze(), axis=1).values
        X_test = zscore(X_test.squeeze(), axis=1).values


        SAX_transform = SAX(n_segments=n_segments, alphabet_size=alphabet_size)
        _ = SAX_transform.fit_transform(X_train).squeeze()
        SAX_train = SAX_transform.transform(X_train).squeeze()
        PAA_test = SAX_transform._get_paa(X_test).squeeze()

        histograms = ["equi-width", "equi-depth"]
        variances = [True, False]

        method_names = ["isax"]
        sfa_transforms = []

        for histogram, variance in itertools.product(histograms, variances):
            sfa = SFAFast(
                word_length=n_segments,
                alphabet_size=alphabet_size,
                window_size=X_train.shape[-1],
                binning_method=histogram,
                norm=True,
                variance=variance,
                lower_bounding_distances=True,
            )

            sfa.fit(X_train)
            Y_words = sfa.transform_words(X_train)
            X_dfts = sfa.transform_mft(X_test).squeeze()
            sfa_transforms.append([sfa, X_dfts, Y_words])

            method_names.append(f"sfa_{histogram}_{variance}")

        sum_scores = {}
        for method_name in method_names:
            sum_scores[method_name] = {
                "dataset": [],
                "pruning_power": [],
            }
        pruning_power = np.zeros((len(method_names)), dtype=np.float64)

        for i in range(X_test.shape[0]):
            nn_dist = np.inf
            Y = X_test[i].reshape(1, -1)
            for j in range(X_train.shape[0]):
                X = X_train[j].reshape(1, -1)
                ed = np.linalg.norm(X[0] - Y[0])
                nn_dist = min(nn_dist, ed)

            for j in range(X_train.shape[0]):

                # SAX-PAA Min-Distance
                min_dist = paa_sax_mindist(
                    PAA_test[i], SAX_train[j], SAX_transform.breakpoints, X_train.shape[-1]
                )
                if min_dist > nn_dist:
                    pruning_power[0] += 1

                for a, (sfa, X_dfts, Y_words) in enumerate(sfa_transforms):
                    # DFT-SFA Min-Distance variants
                    min_dist =  dft_sfa_mindist(X_dfts[i], Y_words[j].squeeze(), sfa.breakpoints)
                    if min_dist > nn_dist:
                        pruning_power[a+1] += 1

        for i in range(len(pruning_power)):
            pruning_power[i] /= X_test.shape[0] * X_train.shape[0]

        for i, method_name in enumerate(method_names):
            sum_scores[method_name]["dataset"].append(dataset_name)
            sum_scores[method_name]["pruning_power"].append(pruning_power[i])

        return sum_scores

    for alphabet_size in alphabet_sizes:
        print(f"Running {alphabet_size}")

        with parallel_backend("threading", n_jobs=-1):
            parallel_res = Parallel(
                n_jobs=parallel_jobs, backend="threading", timeout=9999999, batch_size=1
            )(
                delayed(_parallel_tlb)(dataset, n_segments, alphabet_size)
                for dataset in used_dataset
            )

        sum_scores = {}
        for result in parallel_res:
            if not sum_scores:
                sum_scores = result
            else:
                for name, data in result.items():
                    if name not in sum_scores:
                        sum_scores[name] = {}
                    for key, value in data.items():
                        if key not in sum_scores[name]:
                            if type(value) == list:
                                sum_scores[name][key] = []
                            else:
                                sum_scores[name][key] = 0
                        sum_scores[name][key] += value

        print("\n\n---- Final results -----")

        for name, _ in sum_scores.items():
            print("---- Name", name, "-----")
            print("Total PP:", np.round(np.mean(sum_scores[name]["pruning_power"]), 3))
            print("-----------------")

        csv_scores = []
        for name, _ in sum_scores.items():
            all_pp = sum_scores[name]["pruning_power"]
            all_datasets = sum_scores[name]["dataset"]
            for tlb, dataset_name in zip(all_pp, all_datasets):
                csv_scores.append((name, dataset_name, tlb))

        # if server:
        pd.DataFrame.from_records(
            csv_scores,
            columns=[
                "Method",
                "Dataset",
                "Pruning Power",
            ],
        ).to_csv(f"logs/pp_all_ucr_{n_segments}_{alphabet_size}-14-10-24.csv", index=None)
