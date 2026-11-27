"""The frozen config must carry Appendix B's values (BRIEF.md "Independence rules").

Appendix B is embedded verbatim below. Every key and value must match `configs/landslide.json`, except `name`,
`status` and `paths`, which refer to the spike's repo. The config may add keys only in its own top-level blocks
(`replication`, `sources`), never inside an Appendix B block, and may not drop or change any Appendix B value.
"""

import json

from landslide.config import load_config

APPENDIX_B = r"""
{
  "name": "11_landslide_kill_test",
  "status": "FROZEN 2026-11-25 at the pre-registration commit (before any CNN training, including the smoke test)",
  "crs": "EPSG:5514",
  "paths": {
    "labels": "data/01_geohazards_mining/svahove_nestability_polygons.gpkg",
    "surveyed_sheets": "data/01_geohazards_mining/prehled_mapovani_svahove_nestability.geojson",
    "data_dir": "data/11_landslide",
    "results_dir": "results/11",
    "smoke_dir": "data/11_landslide/smoke"
  },
  "dtm": {
    "service": "https://ags.cuzk.gov.cz/arcgis2/rest/services/dmr5g/ImageServer (DMR 5G, native 2 m)",
    "cell_m": 2.0,
    "margin_m": 250,
    "invalid_rule": "z <= 0 (ImageServer returns 0.0 outside the national border) or z > 2000 -> NaN"
  },
  "labels": {
    "skupina": 1,
    "in_mapped": true,
    "min_area_m2": 3000
  },
  "patch": {
    "size_m": 300,
    "px": 150
  },
  "channels": [
    "slope",
    "lrm_s",
    "lrm_l",
    "curv_prof",
    "curv_plan",
    "svf",
    "detr"
  ],
  "channel_params": {
    "lrm_small_px": 11,
    "lrm_large_px": 51,
    "curv_sigma_px": 1.0,
    "curv_k0": 0.001,
    "svf_radius_px": 20,
    "svf_dirs": 16
  },
  "flysch": {
    "source": "ČGS geologická mapa 1:500 000 (Geologie/geologicka_mapa500_CR, layer 7 Horniny)",
    "subregions": [
      "Marinní terciér vnějších Karpat",
      "Mezozoické sedimenty Karpat",
      "Marinní mezozoikum a terciér vnějších a vnitřních Karpat",
      "Vulkanity vnějších Karpat",
      "Vulkanity vnějších a vnitřních Karpat"
    ],
    "closing_m": 2000,
    "sheet_rule": "sheet is flysch if >= 50% of its area lies inside the closed flysch outline",
    "min_sheet_frac": 0.5
  },
  "sampling": {
    "seed": 11,
    "max_pos": 4000,
    "max_neg": 4000,
    "pts_area_unit_m2": 40000,
    "max_pts_per_poly": 3,
    "min_pt_spacing_m": 100,
    "neg_min_dist_to_polygon_m": 150,
    "neg_exclude_zabaged_landslide_m": 150,
    "neg_candidate_spacing_m": 60,
    "neg_min_spacing_m": 150,
    "slope_bin_deg": 2.0,
    "lookalike_frac": 0.12,
    "lookalike_types": [
      "quarry",
      "spoil",
      "rock",
      "cut"
    ],
    "lookalike_max_per_polygon": 2,
    "cut_point_step_m": 100,
    "cut_window_px": 21,
    "cut_slope_p90_deg": 20.0
  },
  "splits": {
    "block_km": 15,
    "n_folds_a": 5,
    "inner_val_frac": 0.15,
    "transfer_train_kraje": [
      "Zlínský",
      "Moravskoslezský"
    ],
    "sheet_kraj_rule": "majority gen_kraj (first-listed kraj) of all register polygons intersecting the sheet",
    "stage2": {
      "n_flysch": 10,
      "n_nonflysch": 10,
      "min_eligible_polygons": 5,
      "full_sheets_only": true,
      "seed": 2026,
      "require_valid_dtm": "whole cached raster (sheet bbox + 250 m) valid: no 0-m cells (outside the border) or NaN"
    }
  },
  "model": {
    "arch": "resnet18",
    "init": "torchvision IMAGENET1K_V1; conv1 = mean of RGB filters replicated over channels x 3/C",
    "in_channels": 7,
    "head": "single logit"
  },
  "train": {
    "loss": "BCEWithLogits",
    "optimizer": "AdamW",
    "lr": 0.0003,
    "weight_decay": 0.01,
    "batch_size": 64,
    "max_epochs": 40,
    "warmup_epochs": 2,
    "schedule": "linear warmup then cosine to 0 over max_epochs",
    "early_stopping": {
      "metric": "inner-val ROC AUC",
      "patience": 8,
      "restore_best": true
    },
    "amp": "bf16 autocast",
    "augmentation": "random dihedral D4 (8 rotations/flips, exact on the pixel grid)",
    "normalisation": "per-channel mean/std from the run's training patches",
    "tta": "mean of sigmoid over the 8 D4 views",
    "seeds": [
      0,
      1,
      2
    ],
    "ensemble": "mean of seed probabilities",
    "deterministic": true
  },
  "eval": {
    "primary_a": "pooled out-of-fold ROC AUC over all held-out flysch patches (5 folds)",
    "primary_b": "ROC AUC on all patches of non-Zlínský/non-Moravskoslezský sheets",
    "bootstrap": {
      "unit": "sheet",
      "n": 2000,
      "ci": 0.95,
      "method": "percentile",
      "seed": 123
    },
    "thresholds": {
      "a_auc_min": 0.75,
      "b_auc_min": 0.65,
      "stage2_recall": 0.5,
      "stage2_precision_min": 0.3
    }
  },
  "stage2": {
    "model": "same configuration retrained on all stage-1 patches (matched) PLUS the unmatched background sample (4,000 patches, uniform over the stage-1 sheets' feasible area, >= 150 m from polygons) as extra negatives; 15% of blocks as inner validation; seeds [0,1,2]; ensemble",
    "background_n": 4000,
    "stride_px": 25,
    "nms_radius_m": 150,
    "match_buffer_m": 50,
    "ignore_buffer_m": 150,
    "object_rule": "detections = greedy NMS (radius 150 m) of the 50 m window-score lattice; TP = first detection inside a target (eligible polygon assigned to the sheet) buffered 50 m; duplicate on a matched target = ignored; within 150 m of any register polygon = ignored; else FP",
    "precision_rule": "precision at the highest threshold whose object recall >= 0.5 (primary); interpolated max precision at recall >= 0.5 (secondary)",
    "rule_change_note": "3x3 local maxima (first draft) capped max recall at 0.41-0.45 for smooth baseline maps; replaced before any CNN training using baseline maps only"
  },
  "runner": {
    "wall_clock_h": 6.0,
    "min_free_disk_gb": 10,
    "num_workers": 0
  },
  "smoke": {
    "patch_frac": 0.05,
    "max_epochs": 2,
    "folds": [
      0
    ],
    "seeds": [
      0
    ],
    "runs": [
      "a_f0",
      "b",
      "s2"
    ],
    "stage2_sheets": 1,
    "note": "proves the code runs end to end; smoke metrics are never used to change the configuration"
  }
}
"""

EXEMPT = {"name", "status", "paths"}


def _flatten(d, prefix=()):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flatten(v, prefix + (k,))
        else:
            yield prefix + (k,), v


def test_appendix_b_values_verbatim():
    spec = json.loads(APPENDIX_B)
    cfg = load_config()
    for key, value in _flatten(spec):
        if key[0] in EXEMPT:
            continue
        node = cfg
        for part in key:
            assert part in node, f"missing config key {'.'.join(key)}"
            node = node[part]
        assert node == value and type(node) is type(value), (
            f"{'.'.join(key)}: config has {node!r}, Appendix B has {value!r}")


def test_no_appendix_b_section_is_dropped():
    spec = json.loads(APPENDIX_B)
    cfg = load_config()
    assert set(spec) - EXEMPT <= set(cfg)


def test_replication_block():
    rep = load_config()["replication"]
    streams = rep["streams"]
    assert len(set(streams.values())) == len(streams), "stream ids must be unique"
    assert rep["verdict_words"] == ["REPLICATED", "NOT REPLICATED", "INCOMPLETE"]
    assert rep["provenance"]["n_sheets"] >= 10


#: Top-level blocks that are ours, not Appendix B's.
OWN_BLOCKS = {"replication", "sources"}


def _keys(d, prefix=()):
    for k, v in d.items():
        yield prefix + (k,)
        if isinstance(v, dict):
            yield from _keys(v, prefix + (k,))


def test_no_key_added_inside_an_appendix_b_block():
    spec = json.loads(APPENDIX_B)
    cfg = load_config()
    assert set(cfg) - set(spec) == OWN_BLOCKS, "a new top-level block that is neither replication nor sources"
    want = {k for k in _keys(spec) if k[0] not in EXEMPT}
    got = {k for k in _keys(cfg) if k[0] not in EXEMPT | OWN_BLOCKS}
    assert got == want, f"keys added inside Appendix B blocks: {sorted('.'.join(k) for k in got - want)}"


def test_sources_block_is_the_briefs():
    """BRIEF.md "Data sources" (rows 1-6, 8), notes 1-5 and "Provenance check"; P1-16 moved them from module
    constants into the config without changing a value."""
    cfg = load_config()
    src = cfg["sources"]
    assert src["dtm"]["service"] == cfg["dtm"]["service"].split()[0]  # the URL in Appendix B's dtm.service
    assert src["dtm"]["grid_origin_m"] == [0.4, -0.12]
    assert src["dtm"]["package_valid_range_m"] == [-100.0, 2000.0]
    assert src["dtm"]["provenance_min_sheets"] == 10 <= cfg["replication"]["provenance"]["n_sheets"]
    base = "https://mapy.geology.cz/arcgis/rest/services/"
    assert src["register"] == {"url": base + "Geohazardy/svahove_nestability/MapServer/1", "expected_count": 21356}
    assert src["sheets"] == {"url": base + "Prozkoumanost/prehled_mapovani_svahove_nestability/MapServer/0",
                             "expected_count": 860}
    assert src["geology"] == {"url": base + "Geologie/geologicka_mapa500_CR/MapServer/7",
                              "fields": ["objectid", "region", "subregion", "hor_karto"]}
    su = src["susceptibility"]
    assert su["service"] == base + "Geohazardy/sesuvna_nachylnost/MapServer"
    assert (su["layer_id"], su["class_field"], su["alpha_min"], su["dpi"], su["format"]) == (0, "struktura", 128, 96,
                                                                                            "png32")
    assert su["classes"] == {"mid": {"code": 1, "like": "Třída střední%"}, "high": {"code": 2, "like": "Třída vysoké%"}}
    assert su["low"] == {"code": 0, "like": "Třída nízké%"}
    z = src["zabaged"]
    assert z["service"] == "https://ags.cuzk.gov.cz/arcgis/rest/services/ZABAGED_POLOHOPIS/MapServer"
    assert z["layers"] == {"quarry": 118, "spoil": 119, "rock": 130, "landslide": 129, "road": 79, "railway": 75}
    w = src["weights"]
    assert w["file"] == "resnet18-f37072fd.pth" and len(w["sha256"]) == 64
    assert w["sha256"].startswith(w["file"].split("-")[1].split(".")[0])  # torch.hub's hash prefix in the name


def test_replication_keys_added_at_the_freeze():
    """The keys tasks requested (P1-05, P1-09, P1-10, P1-11, P1-13, P1-14), with the values the code used before."""
    cfg = load_config()
    rep = cfg["replication"]
    assert rep["streams"]["verify_sample"] == 11
    assert rep["sampling"]["stage2_buffer_m"] == 150 == cfg["patch"]["size_m"] / 2  # Appendix A §4: 150 m
    assert [s["name"] for s in rep["baselines"]["scorers"]] == ["slope_only", "susceptibility", "slope_susc_lr",
                                                                "terrain_stats_lr", "terrain_stats_gbm"]
    assert rep["eval"] == {"secondary_bootstrap_unit": "block", "stage2_bootstrap_not_reached": "precision_0"}
    assert rep["runner"] == {"min_free_gpu_gb": 7, "eval_reserve_min": 15}  # Appendix A §11
