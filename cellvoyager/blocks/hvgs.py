from __future__ import annotations

from typing import Any

import scanpy as sc
from anndata import AnnData


def compute_hvgs(
    adata: AnnData,
    flavor: str = "seurat_v3",
    n_top_genes: int = 2000,
    subset: bool = False,
    layer: str | None = None,
    force: bool = False,
    **kwargs: Any,
) -> tuple[AnnData, dict[str, Any]]:
    """
    Compute highly variable genes (HVGs) for an AnnData object.

    This CellVoyager block follows the standard template format:

    Parameters:
    - adata: AnnData
    - flavor: str, method for HVG selection (default: 'seurat_v3')
    - n_top_genes: int, number of top HVGs to select (default: 2000)
    - subset: bool, whether to subset the AnnData object to HVGs (default:
    False)
    - layer: str or None, layer to use for HVG computation (default: None)
    - force: bool, whether to force recomputation of HVGs even if they already exist (default: False)
    - **kwargs: additional keyword arguments for scanpy's highly_variable
    """

    warnings: list[str] = []
    steps_run: list[str] = []
    steps_skipped: list[str] = []

    processing_state_before = {
        "shape": (int(adata.n_obs), int(adata.n_vars)),
        "layers": list(adata.layers.keys()),
        "has_hvgs": "highly_variable" in adata.var.columns,
        "n_hvgs": (
            int(adata.var["highly_variable"].sum())
            if "highly_variable" in adata.var.columns
            else 0
        ),
    }

    if layer is not None and layer not in adata.layers:
        return adata, {
            "status": "failed",
            "message": f"Layer '{layer}' not found in AnnData object.",
            "warnings": warnings,
            "results": {},
            "processing_state_before": processing_state_before,
            "processing_state_after": processing_state_before,
            "steps_run": steps_run,
            "steps_skipped": steps_skipped,
        }

    has_existing_hvgs = "highly_variable" in adata.var.columns

    if has_existing_hvgs and not force:
        steps_skipped.append("compute_hvgs")
        n_hvgs = int(adata.var["highly_variable"].sum())

    else:
        try:
            sc.pp.highly_variable_genes(
                adata,
                flavor=flavor,
                n_top_genes=n_top_genes,
                layer=layer,
                subset=False,
                inplace=True,
                **kwargs,
            )
            steps_run.append("compute_hvgs")
            n_hvgs = int(adata.var["highly_variable"].sum())

        except Exception as exc:
            warnings.append(
                f"Failed to compute highly variable genes with flavor='{flavor}': {exc}"
            )

            processing_state_after = {
                "shape": (int(adata.n_obs), int(adata.n_vars)),
                "layers": list(adata.layers.keys()),
                "has_hvgs": "highly_variable" in adata.var.columns,
                "n_hvgs": (
                    int(adata.var["highly_variable"].sum())
                    if "highly_variable" in adata.var.columns
                    else 0
                ),
            }

            return adata, {
                "status": "failed",
                "message": (f"HVG computation failed with flavor='{flavor}'. "),
                "warnings": warnings,
                "results": {
                    "n_cells": int(adata.n_obs),
                    "n_genes": int(adata.n_vars),
                    "n_hvgs": processing_state_after["n_hvgs"],
                    "flavor": flavor,
                    "n_top_genes": int(n_top_genes),
                    "layer_used": layer,
                    "subset": bool(subset),
                    "force": bool(force),
                    "steps_run": steps_run,
                    "steps_skipped": steps_skipped,
                },
                "processing_state_before": processing_state_before,
                "processing_state_after": processing_state_after,
                "steps_run": steps_run,
                "steps_skipped": steps_skipped,
            }

    n_hvgs = int(adata.var["highly_variable"].sum())

    if subset:
        if "highly_variable" in adata.var.columns and n_hvgs > 0:
            adata = adata[:, adata.var["highly_variable"].values].copy()
            steps_run.append("subset_to_hvgs")
        else:
            warnings.append("subset=True was requested, but no HVGs were available.")

    processing_state_after = {
        "shape": (int(adata.n_obs), int(adata.n_vars)),
        "layers": list(adata.layers.keys()),
        "has_hvgs": "highly_variable" in adata.var.columns,
        "n_hvgs": (
            int(adata.var["highly_variable"].sum())
            if "highly_variable" in adata.var.columns
            else int(adata.n_vars)
            if subset
            else 0
        ),
    }

    results: dict[str, Any] = {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_hvgs": int(n_hvgs),
        "flavor": flavor,
        "n_top_genes": int(n_top_genes),
        "layer_used": layer,
        "subset": bool(subset),
        "force": bool(force),
        "steps_run": steps_run,
        "steps_skipped": steps_skipped,
    }

    summary_text = (
        f"HVG summary: {n_hvgs} highly variable genes selected using "
        f"flavor='{flavor}', n_top_genes={n_top_genes}, "
        f"layer={layer if layer is not None else 'adata.X'}, "
        f"subset={subset}."
    )

    status = "success" if not warnings else "partial"

    result = {
        "status": status,
        "message": summary_text,
        "warnings": warnings,
        "results": results,
        "processing_state_before": processing_state_before,
        "processing_state_after": processing_state_after,
        "steps_run": steps_run,
        "steps_skipped": steps_skipped,
    }

    return adata, result
