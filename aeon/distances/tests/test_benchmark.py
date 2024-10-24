"""Test MinDist functions of symbolic representations."""

import itertools
import os
from warnings import simplefilter

import time
import numpy as np
import pandas as pd
from scipy.stats import zscore

simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=UserWarning)

from joblib import Parallel, delayed, parallel_backend

from aeon.distances._dft_sfa_mindist import dft_sfa_mindist
from aeon.distances._paa_sax_mindist import paa_sax_mindist

import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import seaborn as sns

from aeon.transformations.collection.dictionary_based import SAX, SFAFast

dataset_names_full = [
    "ArrowHead",
    # "Beef",
    # "BeetleFly",
    # "BirdChicken",
    # "BME",
    # "Car",
    # "CBF",
    # "Chinatown",
    # "ChlorineConcentration",
    # "CinCECGTorso",
    # "Coffee",
    # "Computers",
    # "CricketX",
    # "CricketY",
    # "CricketZ",
    "Crop",
    # "DiatomSizeReduction",
    # "DistalPhalanxOutlineAgeGroup",
    # "DistalPhalanxOutlineCorrect",
    # "DistalPhalanxTW",
    # "DodgerLoopDay",
    # "DodgerLoopGame",
    # "DodgerLoopWeekend",
    # "Earthquakes",
    # "ECG200",
    # "ECG5000",
    # "ECGFiveDays",
    # "EOGHorizontalSignal",
    # "EOGVerticalSignal",
    # "EthanolLevel",
    # "FaceAll",
    # "FaceFour",
    # "FacesUCR",
    # "FiftyWords",
    # "Fish",
    "FordA",
    "FordB",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    # "Fungi",
    # "GestureMidAirD1",
    # "GestureMidAirD2",
    # "GestureMidAirD3",
    # "GesturePebbleZ1",
    # "GesturePebbleZ2",
    # "GunPoint",
    # "GunPointAgeSpan",
    # "GunPointMaleVersusFemale",
    # "GunPointOldVersusYoung",
    # "Ham",
    # "HandOutlines",
    # "Haptics",
    # "Herring",
    # "HouseTwenty",
    # "InlineSkate",
    # "InsectEPGRegularTrain",
    # "InsectEPGSmallTrain",
    # "InsectWingbeatSound",
    # "ItalyPowerDemand",
    # "LargeKitchenAppliances",
    # "Lightning2",
    # "Lightning7",
    # "Mallat",
    # "Meat",
    # "MedicalImages",
    # "MelbournePedestrian",            # iSAX mindist error?
    # "MiddlePhalanxOutlineAgeGroup",
    # "MiddlePhalanxOutlineCorrect",
    # "MiddlePhalanxTW",
    # "MixedShapesRegularTrain",
    # "MixedShapesSmallTrain",
    # "MoteStrain",
    # "NonInvasiveFetalECGThorax1",
    # "NonInvasiveFetalECGThorax2",
    # "OliveOil",
    # "OSULeaf",
    # "PhalangesOutlinesCorrect",
    # "Phoneme",
    # "PickupGestureWiimoteZ",
    # "PigAirwayPressure",
    # "PigArtPressure",
    # "PigCVP",
    # "PLAID",
    # "Plane",
    # "PowerCons",
    # "ProximalPhalanxOutlineAgeGroup",
    # "ProximalPhalanxOutlineCorrect",
    # "ProximalPhalanxTW",
    # "RefrigerationDevices",
    # "Rock",
    # "ScreenType",
    # "SemgHandGenderCh2",
    # "SemgHandMovementCh2",
    # "SemgHandSubjectCh2",
    # "ShakeGestureWiimoteZ",
    # "ShapeletSim",
    # "ShapesAll",
    # "SmallKitchenAppliances",
    # # "SmoothSubspace",         # SFA mindist error?
    # "SonyAIBORobotSurface1",
    # "SonyAIBORobotSurface2",
    "StarLightCurves",
    # "Strawberry",
    # "SwedishLeaf",
    # "Symbols",
    # "SyntheticControl",
    # "ToeSegmentation1",
    # "ToeSegmentation2",
    # "Trace",
    # "TwoLeadECG",
    # "TwoPatterns",
    # "UMD",
    # "UWaveGestureLibraryAll",
    # "UWaveGestureLibraryX",
    # "UWaveGestureLibraryY",
    # "Wafer",
    # "Wine",
    # "WordSynonyms",
    # "Worms",
    # "WormsTwoClass",
    # "Yoga",
]

dataset_names = [
    # "Chinatown"
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "Car",
    "CBF",
    "Coffee",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    "ECG200",
    "ECGFiveDays",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "GunPoint",
    "ItalyPowerDemand",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW",
    "OliveOil",
    "Plane",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "SyntheticControl",
    "TwoLeadECG",
    "Wine",
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
alphabet_size = 256

if os.path.exists(DATA_PATH):
    parallel_jobs = 8
    DATA_PATH = "/Users/bzcschae/workspace/UCRArchive_2018/"
    used_dataset = dataset_names_full
# server
else:
    DATA_PATH = "/vol/fob-wbib-vol2/wbi/schaefpa/sktime/datasets/UCRArchive_2018"
    parallel_jobs = 80
    server = True
    used_dataset = dataset_names_full


def simsearch(dataset_name, n_segments, alphabet_size):
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

    start = time.time()

    sfa = SFAFast(
        word_length=n_segments,
        alphabet_size=alphabet_size,
        window_size=X_train.shape[-1],
        binning_method="equi-width",
        norm=True,
        variance=True,
        lower_bounding_distances=True,
    )

    sfa.fit(X_train)
    X_train_dfts = sfa.transform_mft(X_train).squeeze()
    X_test_dfts = sfa.transform_mft(X_test).squeeze()

    bsf = np.inf
    best_idx = -1
    for i in range(min(X_train.shape[0], X_test.shape[0])):
        X_dft = X_train_dfts[i]  #.reshape(1, -1)
        Y_dft = X_test_dfts[i]  # .reshape(1, -1)
        # Compute Lower Bounding Distance
        buf = (X_dft - Y_dft)
        dist = np.sqrt(2 * buf @ buf)

        if dist < bsf:
            # Compute Euclidean Distance
            X = X_train[i].reshape(1, -1)
            Y = X_test[i].reshape(1, -1)
            ed = np.linalg.norm(X[0] - Y[0])
            if ed < bsf:
                bsf = ed
                best_idx = i


    duration = time.time() - start
    print(f"\tDuration LB: {duration:.2f}s, Best: {best_idx}")

    # Pure ED
    start = time.time()

    bsf = np.inf
    best_idx = -1
    for i in range(min(X_train.shape[0], X_test.shape[0])):
        # Compute Euclidean Distance
        X = X_train[i].reshape(1, -1)
        Y = X_test[i].reshape(1, -1)
        ed = np.linalg.norm(X[0] - Y[0])
        if ed < bsf:
            bsf = ed
            best_idx = i

    duration = time.time() - start
    print(f"\tDuration ED: {duration:.2f}s, Best: {best_idx}")



if __name__ == "__main__":
    for dataset in used_dataset:
        simsearch(dataset, n_segments, alphabet_size)
