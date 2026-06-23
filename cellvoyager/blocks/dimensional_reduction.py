from __future__ import annotations

from typing import Any

import scanpy as sc


def dimensional_reduction_summary(
    adata: Any,
    layer: str | None = None,
    use_hvgs: bool = True,
    n_top_genes: int = 2000,
    n_pcs: int = 50,
    neighbors_n_pcs: int = 25,
    n_neighbors: int = 15,
    run_pca: bool = True,
    run_neighbors: bool = True,
    run_umap: bool = True,
    run_tsne: bool = False,
    run_leiden: bool = True,
    leiden_resolution: float = 0.5,
    leiden_key: str = "leiden",
    force: bool = False,
) -> tuple[Any, dict]:
    """
    This CellVoyager block is a template for dimensional reduction on an AnnData object:

    Parameters:
        adata: AnnData object to perform dimensional reduction on.
        layer: Layer of the AnnData object to use for dimensional reduction.
        use_hvgs: Whether to use highly variable genes for dimensional reduction.
        compute_hvgs_if_missing: Whether to compute highly variable genes if they are missing.
        n_top_genes: Number of top highly variable genes to use.
        n_pcs: Number of principal components to compute.
        neighbors_n_pcs: Number of principal components to use for computing neighbors.
        n_neighbors: Number of neighbors to use for computing the neighborhood graph.
        run_umap: Whether to run UMAP for dimensional reduction.
        run_tsne: Whether to run t-SNE for dimensional reduction.
        run_leiden: Whether to run Leiden clustering after dimensional reduction.
        leiden_resolution: Resolution parameter for Leiden clustering.
        leiden_key: Key in adata.obs to store Leiden clustering results.
        force: Whether to force re-computation of dimensional reduction even if results already exist.

    Runs:
        1. Optionally computes highly variable genes if they are missing and use_hvgs is True.
        2. Computes PCA on the specified layer or the default layer of the AnnData object.
        3. Computes the neighborhood graph based on PCA results.
        4. Optionally runs UMAP and/or t-SNE for visualization.
        5. Optionally runs Leiden clustering and stores results in adata.obs under the specified key.
    """

    warnings: list[str] = []
    n_hvgs = 0

    if layer is not None and layer not in adata.layers:
        return adata, {
            "status": "failed",
            "message": f"Layer '{layer}' not found in AnnData object.",
            "warnings": warnings,
            "results": {},
        }

    processing_state_before = {
        "shape": (adata.n_obs, adata.n_vars),
        "layers": list(adata.layers.keys()),
        "obsm": list(adata.obsm.keys()),
        "uns_selected": [k for k in ["log1p", "pca", "neighbors"] if k in adata.uns],
        "has_hvgs": "highly_variable" in adata.var.columns,
        "has_pca": "X_pca" in adata.obsm,
        "has_pca_metadata": "pca" in adata.uns,
        "has_neighbors": "neighbors" in adata.uns,
        "has_umap": "X_umap" in adata.obsm,
        "has_tsne": "X_tsne" in adata.obsm,
        "leiden_columns": [c for c in adata.obs.columns if "leiden" in str(c).lower()],
    }

    steps_run: list[str] = []
    steps_skipped: list[str] = []

    if "log1p" not in adata.uns and layer is None:
        warnings.append(
            "No log1p transformation found in adata.uns. Consider running log1p before dimensional reduction."
        )

    has_hvgs = "highly_variable" in adata.var.columns

    if has_hvgs:
        n_hvgs = int(adata.var["highly_variable"].fillna(False).astype(bool).sum())
    else:
        n_hvgs = 0

    if use_hvgs and has_hvgs and n_hvgs > 0:
        n_features_for_pca = n_hvgs
        use_highly_variable_for_pca = True
    else:
        n_features_for_pca = int(adata.n_vars)
        use_highly_variable_for_pca = False

    if run_pca:
        safe_n_pcs = min(n_pcs, adata.n_obs - 1, n_features_for_pca - 1)

        if safe_n_pcs < n_pcs:
            warnings.append(
                f"Requested n_pcs={n_pcs}, but only {safe_n_pcs} PCs can safely be calculated "
                f"from this dataset."
            )

        safe_neighbors_n_pcs = min(neighbors_n_pcs, safe_n_pcs)

        if "X_pca" in adata.obsm and "pca" in adata.uns and not force:
            steps_skipped.append(
                "PCA skipped because X_pca and pca metadata already exist."
            )

        else:
            try:
                sc.tl.pca(
                    adata,
                    n_comps=safe_n_pcs,
                    use_highly_variable=use_highly_variable_for_pca,
                    svd_solver="arpack",
                    layer=layer,
                )
                steps_run.append("PCA")

            except Exception:
                if layer is not None:
                    warnings.append(
                        "This Scanpy version may not support the layer argument in sc.tl.pca. "
                        "Falling back to adata.X for PCA."
                    )

                sc.tl.pca(
                    adata,
                    n_comps=safe_n_pcs,
                    use_highly_variable=use_highly_variable_for_pca,
                    svd_solver="arpack",
                )
                steps_run.append("pca")

    if run_neighbors:
        if "neighbors" in adata.uns and not force:
            steps_skipped.append(
                "Neighbors skipped because neighbors already exist in adata.uns."
            )

        else:
            try:
                sc.pp.neighbors(
                    adata,
                    n_neighbors=n_neighbors,
                    n_pcs=safe_neighbors_n_pcs,
                    use_rep="X_pca",
                )
                steps_run.append("neighbors")

            except Exception as exc:
                return adata, {
                    "status": "failed",
                    "message": f"Neighbour graph construction failed: {exc}",
                    "warnings": warnings,
                    "processing_state_before": processing_state_before,
                    "results": {},
                }

    if run_umap:
        if "X_umap" in adata.obsm and not force:
            steps_skipped.append(
                "UMAP skipped because X_umap already exists in adata.obsm."
            )

        else:
            try:
                sc.tl.umap(adata)
                steps_run.append("umap")

            except Exception as exc:
                warnings.append(f"UMAP failed: {exc}")

    if run_tsne:
        if "X_tsne" in adata.obsm and not force:
            steps_skipped.append(
                "t-SNE skipped because X_tsne already exists in adata.obsm."
            )

        else:
            try:
                sc.tl.tsne(adata, n_pcs=safe_neighbors_n_pcs, inplace=True)
                steps_run.append("tsne")

            except Exception as exc:
                warnings.append(f"t-SNE failed: {exc}")

    if run_leiden:
        if leiden_key in adata.obs.columns and not force:
            steps_skipped.append(
                f"Leiden clustering skipped because {leiden_key} already exists in adata.obs."
            )

        else:
            try:
                sc.tl.leiden(
                    adata,
                    resolution=leiden_resolution,
                    key_added=leiden_key,
                )
                steps_run.append("leiden")

            except Exception as exc:
                warnings.append(f"Leiden clustering failed: {exc}")

    processing_state_after = {
        "shape": (adata.n_obs, adata.n_vars),
        "layers": list(adata.layers.keys()),
        "obsm": list(adata.obsm.keys()),
        "uns_selected": [k for k in ["log1p", "pca", "neighbors"] if k in adata.uns],
        "has_hvgs": "highly_variable" in adata.var.columns,
        "n_hvgs": int(adata.var["highly_variable"].sum()),
        "has_pca": "X_pca" in adata.obsm,
        "has_pca_metadata": "pca" in adata.uns,
        "has_neighbors": "neighbors" in adata.uns,
        "has_umap": "X_umap" in adata.obsm,
        "has_tsne": "X_tsne" in adata.obsm,
        "leiden_columns": [c for c in adata.obs.columns if "leiden" in str(c).lower()],
    }

    results: dict[str, Any] = {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_hvgs": processing_state_after["n_hvgs"],
        "n_pcs_requested": n_pcs,
        "n_pcs_used": safe_n_pcs,
        "neighbors_n_pcs_used": safe_neighbors_n_pcs,
        "n_neighbors": n_neighbors,
        "layer_used": layer,
        "use_hvgs": use_hvgs,
        "steps_run": steps_run,
        "steps_skipped": steps_skipped,
    }

    if leiden_key in adata.obs.columns:
        cluster_counts = adata.obs[leiden_key].value_counts().to_dict()
        results["leiden_key"] = leiden_key
        results["n_leiden_clusters"] = len(cluster_counts)
        results["leiden_cluster_counts"] = {
            str(k): int(v) for k, v in cluster_counts.items()
        }

    summary_text = (
        "Dimensional Reduction Summary:\n"
        f"Cells: {adata.n_obs}\n"
        f"Genes: {adata.n_vars}\n"
        f"Layer used: {layer if layer is not None else 'adata.X'}\n"
        f"HVGs used for PCA: {use_highly_variable_for_pca}\n"
        f"Number of HVGs: {results['n_hvgs']}\n"
        f"PCs calculated/used: {safe_n_pcs}\n"
        f"PCs used for neighbours: {safe_neighbors_n_pcs}\n"
        f"Neighbour graph present: {'neighbors' in adata.uns}\n"
        f"UMAP present: {'X_umap' in adata.obsm}\n"
        f"t-SNE present: {'X_tsne' in adata.obsm}\n"
        f"Leiden columns: {', '.join(processing_state_after['leiden_columns']) if processing_state_after['leiden_columns'] else 'None'}\n"
        f"Steps run: {', '.join(steps_run) if steps_run else 'None'}\n"
        f"Steps skipped: {', '.join(steps_skipped) if steps_skipped else 'None'}\n"
    )

    status = "success" if not warnings else "partial"

    result = {
        "status": status,
        "message": summary_text,
        "warnings": warnings,
        "results": results,
        "processing_state_before": processing_state_before,
        "processing_state_after": processing_state_after,
    }

    return adata, result
