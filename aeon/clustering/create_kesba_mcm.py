from aeon.visualisation.results import create_multi_comparison_matrix

if __name__ == "__main__":
    experiment_name = "kasba-numba"
    path_to_results = f"/Users/chrisholder/Documents/Research/SOTON/clustering-results/experiments/normalised/test-train-split/{experiment_name}/CLAcc/clacc_mean.csv"

    create_multi_comparison_matrix(
        df_results=path_to_results,
        dataset_column="Estimators:",
        output_dir="./",
        pdf_savename="test",
        font_size=22,
        pvalue_threshold=0.05,
        show_symetry=False,
        precision=3,
    )

    pass
