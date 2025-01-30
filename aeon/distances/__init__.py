"""Distance computation."""

__all__ = [
    "create_bounding_matrix",
    "squared_distance",
    "squared_pairwise_distance",
    "euclidean_distance",
    "euclidean_pairwise_distance",
    "manhattan_distance",
    "manhattan_pairwise_distance",
    "minkowski_distance",
    "minkowski_pairwise_distance",
    "adtw_distance",
    "adtw_pairwise_distance",
    "adtw_cost_matrix",
    "adtw_alignment_path",
    "dtw_distance",
    "dtw_pairwise_distance",
    "dtw_cost_matrix",
    "dtw_alignment_path",
    "ddtw_distance",
    "ddtw_pairwise_distance",
    "ddtw_alignment_path",
    "ddtw_cost_matrix",
    "wdtw_distance",
    "wdtw_pairwise_distance",
    "wdtw_cost_matrix",
    "wdtw_alignment_path",
    "wddtw_distance",
    "wddtw_pairwise_distance",
    "wddtw_alignment_path",
    "wddtw_cost_matrix",
    "lcss_distance",
    "lcss_pairwise_distance",
    "lcss_alignment_path",
    "lcss_cost_matrix",
    "erp_distance",
    "erp_pairwise_distance",
    "erp_alignment_path",
    "erp_cost_matrix",
    "edr_distance",
    "edr_pairwise_distance",
    "edr_alignment_path",
    "edr_cost_matrix",
    "twe_distance",
    "twe_pairwise_distance",
    "twe_alignment_path",
    "twe_cost_matrix",
    "msm_distance",
    "msm_alignment_path",
    "msm_cost_matrix",
    "msm_pairwise_distance",
    "distance",
    "pairwise_distance",
    "alignment_path",
    "cost_matrix",
    "get_cost_matrix_function",
    "get_distance_function",
    "get_distance_function_names",
    "get_pairwise_distance_function",
    "get_alignment_path_function",
    "shape_dtw_distance",
    "shape_dtw_cost_matrix",
    "shape_dtw_alignment_path",
    "shape_dtw_pairwise_distance",
    "sbd_distance",
    "sbd_pairwise_distance",
    "mp_distance",
    "mp_pairwise_distance",
    "mindist_paa_sax_distance",
    "mindist_sax_distance",
    "mindist_sfa_distance",
    "mindist_dft_sfa_distance",
    "shift_scale_invariant_distance",
    "shift_scale_invariant_pairwise_distance",
    "shift_scale_invariant_best_shift",
    "soft_dtw_distance",
    "soft_dtw_pairwise_distance",
    "soft_dtw_alignment_path",
    "soft_dtw_cost_matrix",
]

from aeon.distances._distance import (
    alignment_path,
    cost_matrix,
    distance,
    get_alignment_path_function,
    get_cost_matrix_function,
    get_distance_function,
    get_distance_function_names,
    get_pairwise_distance_function,
    pairwise_distance,
)
from aeon.distances._mpdist import mp_distance, mp_pairwise_distance
from aeon.distances._sbd import sbd_distance, sbd_pairwise_distance
from aeon.distances._shift_scale_invariant import (
    shift_scale_invariant_best_shift,
    shift_scale_invariant_distance,
    shift_scale_invariant_pairwise_distance,
)
from aeon.distances.elastic import (
    adtw_alignment_path,
    adtw_cost_matrix,
    adtw_distance,
    adtw_pairwise_distance,
    create_bounding_matrix,
    ddtw_alignment_path,
    ddtw_cost_matrix,
    ddtw_distance,
    ddtw_pairwise_distance,
    dtw_alignment_path,
    dtw_cost_matrix,
    dtw_distance,
    dtw_pairwise_distance,
    edr_alignment_path,
    edr_cost_matrix,
    edr_distance,
    edr_pairwise_distance,
    erp_alignment_path,
    erp_cost_matrix,
    erp_distance,
    erp_pairwise_distance,
    lcss_alignment_path,
    lcss_cost_matrix,
    lcss_distance,
    lcss_pairwise_distance,
    msm_alignment_path,
    msm_cost_matrix,
    msm_distance,
    msm_pairwise_distance,
    shape_dtw_alignment_path,
    shape_dtw_cost_matrix,
    shape_dtw_distance,
    shape_dtw_pairwise_distance,
    soft_dtw_alignment_path,
    soft_dtw_cost_matrix,
    soft_dtw_distance,
    soft_dtw_pairwise_distance,
    twe_alignment_path,
    twe_cost_matrix,
    twe_distance,
    twe_pairwise_distance,
    wddtw_alignment_path,
    wddtw_cost_matrix,
    wddtw_distance,
    wddtw_pairwise_distance,
    wdtw_alignment_path,
    wdtw_cost_matrix,
    wdtw_distance,
    wdtw_pairwise_distance,
)
from aeon.distances.mindist._dft_sfa import mindist_dft_sfa_distance
from aeon.distances.mindist._paa_sax import mindist_paa_sax_distance
from aeon.distances.mindist._sax import mindist_sax_distance
from aeon.distances.mindist._sfa import mindist_sfa_distance
from aeon.distances.pointwise._euclidean import (
    euclidean_distance,
    euclidean_pairwise_distance,
)
from aeon.distances.pointwise._manhattan import (
    manhattan_distance,
    manhattan_pairwise_distance,
)
from aeon.distances.pointwise._minkowski import (
    minkowski_distance,
    minkowski_pairwise_distance,
)
from aeon.distances.pointwise._squared import (
    squared_distance,
    squared_pairwise_distance,
)
