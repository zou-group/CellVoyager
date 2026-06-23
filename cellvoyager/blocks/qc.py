from __future__ import annotations

from typing import Any

import scanpy as sc


def qc_summary(
    adata: Any,
    groupby: list[str] | None = None,
    apply_filters: bool = False,
    apply_normalization: bool = False,
    apply_log1p: bool = False,
    apply_scaling: bool = False,
    min_genes: int = 200,
    min_counts: int = 500,
    max_counts: int = 50000,
    max_mito_pct: float | None = None,
    min_cell_per_gene: int = 3,
) -> tuple[Any, dict]:
    """
    This CellVoyager block is a template for QC on an AnnData object:

    Parameters:
        adata: AnnData object to perform QC on.
        groupby: List of columns in adata.obs to group by for QC summary.
        apply_filters: Whether to apply filtering based on QC metrics.
        min_genes: Minimum number of genes per cell.
        min_counts: Minimum number of counts per cell.
        max_counts: Maximum number of counts per cell.
        max_mito_pct: Maximum percentage of mitochondrial genes per cell.
        min_cell_per_gene: Minimum number of cells expressing a gene.

    Runs:
        1. Detects mitochondrial genes and calculates QC metrics.
        2. Calculate QC metrics.
        3. Generate a QC summary grouped by specified columns.
        4. Optionally apply filtering based on QC metrics.
    """

    if groupby is None:
        groupby = []

    cells_before_filtering = adata.n_obs
    genes_before_filtering = adata.n_vars

    filter_applied = False

    cells_after_filtering = cells_before_filtering
    genes_after_filtering = genes_before_filtering

    warnings: list[str] = []

    filter_log = {
        "cells_before": cells_before_filtering,
        "genes_before": genes_before_filtering,
        "cells_after": cells_after_filtering,
        "genes_after": genes_after_filtering,
        "min_genes": min_genes,
        "min_counts": min_counts,
        "max_counts": max_counts,
        "max_mito_pct": max_mito_pct,
        "min_cells_per_gene": min_cell_per_gene,
    }

    preprocessing_log = {
        "normalization_requested": apply_normalization,
        "log1p_requested": apply_log1p,
        "scaling_requested": apply_scaling,
        "normalization_applied": False,
        "log1p_applied": False,
        "scaling_applied": False,
        "normalization_layer": None,
        "log1p_layer": None,
        "scaled_layer": None,
    }

    steps_run: list[str] = []
    steps_skipped: list[str] = []

    processing_state = {
        "shape": (adata.n_obs, adata.n_vars),
        "layers": list(adata.layers.keys()),
        "obsm": list(adata.obsm.keys()),
        "varm": list(adata.varm.keys()),
        "uns_selected": [k for k in ["log1p", "pca", "neighbors"] if k in adata.uns],
        "raw_exists": adata.raw is not None,
        "has_pca": "X_pca" in adata.obsm,
        "has_pca_metadata": "pca" in adata.uns,
        "has_neighbors": "neighbors" in adata.uns,
        "leiden_columns": [c for c in adata.obs.columns if "leiden" in str(c).lower()],
        "has_hvgs": "highly_variable" in adata.var.columns,
        "had_mito_column_before_qc": "mt" in adata.var.columns,
    }

    if "highly_variable" in adata.var.columns:
        processing_state["n_hvgs"] = int(adata.var["highly_variable"].sum())
    else:
        processing_state["n_hvgs"] = None

    # Annotate the mt genes in the AnnData obj
    if "mt" not in adata.var.columns:
        adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")

    n_mt_genes = 0
    try:
        n_mt_genes = int(adata.var["mt"].sum())
        if n_mt_genes == 0:
            warnings.append(
                "No mitochondrial genes detected. QC metrics may be inaccurate."
            )

    except Exception:
        warnings.append("Could not determine the number of mitochondrial genes.")

    processing_state["has_mito_column_after_qc"] = "mt" in adata.var.columns
    processing_state["n_mito_genes"] = n_mt_genes

    # Calculate QC metrics
    required_qc_metrics = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
    missing_metrics = [
        metric for metric in required_qc_metrics if metric not in adata.obs.columns
    ]

    if missing_metrics:
        if adata.var["mt"].any():
            sc.pp.calculate_qc_metrics(
                adata,
                qc_vars=["mt"],
                percent_top=None,
                log1p=False,
                inplace=True,
            )
        else:
            warnings.append(
                "QC metrics were missing, but no mitochondrial genes were detected. "
                "Calculated standard QC metrics and set pct_counts_mt to 0.0."
            )
            sc.pp.calculate_qc_metrics(
                adata,
                qc_vars=[],
                percent_top=None,
                log1p=False,
                inplace=True,
            )
            adata.obs["pct_counts_mt"] = 0.0

    missing_after = [
        metric for metric in required_qc_metrics if metric not in adata.obs.columns
    ]
    if missing_after:
        return adata, {
            "status": "failed",
            "message": f"QC metrics calculation failed. Missing metrics: {', '.join(missing_after)}.",
            "warnings": warnings,
            "processing_state": processing_state,
            "results": {},
            "filter_applied": filter_applied,
            "filter_log": filter_log,
        }

    # Generate QC summary
    def make_summary(obs: Any) -> dict[str, float]:
        return {
            "n_cells": obs.shape[0],
            "mean_n_genes": float(obs["n_genes_by_counts"].mean()),
            "median_n_genes": float(obs["n_genes_by_counts"].median()),
            "mean_total_counts": float(obs["total_counts"].mean()),
            "median_total_counts": float(obs["total_counts"].median()),
            "mean_pct_mito": float(obs["pct_counts_mt"].mean()),
            "median_pct_mito": float(obs["pct_counts_mt"].median()),
        }

    results: dict[str, Any] = {}

    # Small funtion inside the list that only includes groupby columns that are actually present in adata.obs
    available_groupby = [col for col in groupby if col in adata.obs.columns]
    missing_groupby = [col for col in groupby if col not in adata.obs.columns]

    if missing_groupby:
        warnings.append(
            f"Some groupby columns are missing in adata.obs: {', '.join(missing_groupby)}."
        )

    if available_groupby:
        grouped = adata.obs.groupby(available_groupby, observed=True)
        for group_name, group_data in grouped:
            if isinstance(group_name, tuple):
                group_name = " | ".join(map(str, group_name))
            else:
                group_name = str(group_name)

            results[group_name] = make_summary(group_data)

    else:
        results["overall"] = make_summary(adata.obs)

    # Filtering

    if apply_filters:
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_cells(adata, min_counts=min_counts)
        sc.pp.filter_cells(adata, max_counts=max_counts)

        if max_mito_pct is not None:
            if "pct_counts_mt" in adata.obs.columns:
                adata = adata[adata.obs["pct_counts_mt"] <= max_mito_pct].copy()
            else:
                warnings.append(
                    "pct_counts_mt not found, so mitochondrial filtering was skipped."
                )
        else:
            warnings.append(
                "No max_mito_pct supplied, so mitochondrial filtering was skipped."
            )

        sc.pp.filter_genes(adata, min_cells=min_cell_per_gene)

        filter_applied = True

        cells_after_filtering = adata.n_obs
        genes_after_filtering = adata.n_vars
        filter_log["cells_after"] = cells_after_filtering
        filter_log["genes_after"] = genes_after_filtering
        filter_log["cells_removed"] = cells_before_filtering - cells_after_filtering
        filter_log["genes_removed"] = genes_before_filtering - genes_after_filtering

    has_normalization = "X_normalized" in adata.layers
    if apply_normalization and not has_normalization:
        if "counts" in adata.layers:
            adata.layers["X_normalized"] = adata.layers["counts"].copy()
            preprocessing_log["normalization_source"] = "adata.layers['counts']"
        else:
            warnings.append(
                "'counts' layer was not found. Using adata.X as the source for normalization. "
                "This assumes adata.X contains raw counts."
            )
            adata.layers["X_normalized"] = adata.X.copy()
            preprocessing_log["normalization_source"] = "adata.X"

        sc.pp.normalize_total(
            adata,
            target_sum=1e4,
            layer="X_normalized",
        )

        preprocessing_log["normalization_applied"] = True
        preprocessing_log["normalization_layer"] = "X_normalized"
        steps_run.append("normalize_total")

    else:
        if apply_normalization and has_normalization:
            warnings.append(
                "Normalization was requested, but 'X_normalized' already exists in adata.layers. Skipping normalization."
            )
            preprocessing_log["normalization_applied"] = False
            preprocessing_log["normalization_layer"] = "X_normalized"
            steps_skipped.append("normalize_total")

    has_log1p = "X_log1p" in adata.layers
    if apply_log1p and not has_log1p:
        if "X_normalized" not in adata.layers:
            warnings.append(
                "Log1p transformation was requested, but 'X_normalized' does not exist in adata.layers. Skipping log1p transformation."
            )
            steps_skipped.append("log1p")

        else:
            adata.layers["X_log1p"] = adata.layers["X_normalized"].copy()
            sc.pp.log1p(adata, layer="X_log1p")

            preprocessing_log["log1p_applied"] = True
            preprocessing_log["log1p_layer"] = "X_log1p"
            steps_run.append("log1p")

    elif apply_log1p and has_log1p:
        warnings.append(
            "Log1p transformation was requested, but 'X_log1p' already exists in adata.layers. "
            "Skipping log1p transformation."
        )

        preprocessing_log["log1p_applied"] = False
        preprocessing_log["log1p_layer"] = "X_log1p"
        steps_skipped.append("log1p")

    else:
        if apply_log1p and has_log1p:
            warnings.append(
                "Log1p transformation was requested, but 'X_log1p' already exists in adata.layers. Skipping log1p transformation."
            )

    has_scaled = "X_scaled" in adata.layers

    if apply_scaling and not has_scaled:
        if "X_log1p" not in adata.layers:
            warnings.append(
                "Scaling was requested, but 'X_log1p' does not exist in adata.layers. "
                "Skipping scaling."
            )
            steps_skipped.append("scale")

        else:
            adata.layers["X_scaled"] = adata.layers["X_log1p"].copy()

            sc.pp.scale(
                adata,
                layer="X_scaled",
                max_value=10,
            )

            preprocessing_log["scaling_applied"] = True
            preprocessing_log["scaled_layer"] = "X_scaled"
            steps_run.append("scale")

    elif apply_scaling and has_scaled:
        warnings.append(
            "Scaling was requested, but 'X_scaled' already exists in adata.layers. "
            "Skipping scaling."
        )
        preprocessing_log["scaling_applied"] = False
        preprocessing_log["scaled_layer"] = "X_scaled"
        steps_skipped.append("scale")

    # Summary
    overall_summary = make_summary(adata.obs)

    summary_text = (
        f"QC Summary:\n"
        f"Mitochondrial genes detected: {n_mt_genes}\n"
        f"Cells before filtering: {cells_before_filtering}\n"
        f"Genes before filtering: {genes_before_filtering}\n"
        f"Median genes per cell: {overall_summary['median_n_genes']:.2f}\n"
        f"Median counts per cell: {overall_summary['median_total_counts']:.2f}\n"
        f"Median mitochondrial percentage: {overall_summary['median_pct_mito']:.2f}\n"
        f"Available objs in adata obs: {', '.join(adata.obs.columns)}\n"
        f"Available layers in adata.layers: {', '.join(adata.layers.keys()) if adata.layers.keys() else 'None'}\n"
        f"\nPreprocessing:\n"
        f"Normalization requested: {apply_normalization}\n"
        f"Normalization applied: {preprocessing_log['normalization_applied']}\n"
        f"Normalization layer: {preprocessing_log['normalization_layer']}\n"
        f"Log1p requested: {apply_log1p}\n"
        f"Log1p applied: {preprocessing_log['log1p_applied']}\n"
        f"Log1p layer: {preprocessing_log['log1p_layer']}\n"
        f"Scaling requested: {apply_scaling}\n"
        f"Scaling applied: {preprocessing_log['scaling_applied']}\n"
        f"Scaled layer: {preprocessing_log['scaled_layer']}\n"
        f"Steps run: {', '.join(steps_run) if steps_run else 'None'}\n"
        f"Steps skipped: {', '.join(steps_skipped) if steps_skipped else 'None'}\n"
        f""
    )

    if filter_applied:
        summary_text += (
            f"Cells after filtering: {cells_after_filtering}\n"
            f"Genes after filtering: {genes_after_filtering}\n"
            f"Cells removed: {cells_before_filtering - cells_after_filtering}\n"
            f"Genes removed: {genes_before_filtering - genes_after_filtering}\n"
        )

    else:
        summary_text += "No filtering applied.\n"

    result = {
        "status": "success" if not warnings else "partial",
        "message": summary_text,
        "warnings": warnings,
        "results": results,
        "processing_state": processing_state,
        "filter_applied": filter_applied,
        "filter_log": filter_log,
        "preprocessing_log": preprocessing_log,
        "steps_run": steps_run,
        "steps_skipped": steps_skipped,
    }

    return adata, result
