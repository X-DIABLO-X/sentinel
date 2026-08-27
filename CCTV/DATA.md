# Data provenance and licensing

No organiser dataset was supplied for this challenge, so everything here is
public, licensed for research, or generated. Each row states what it was used for
and under what terms; where a licence could not be verified it says so rather
than guessing.

**Nothing listed here is redistributed in this repository.** `scripts/` contains
download helpers; the data stays with its original host.

## Detection models

| Asset | Source | Licence | Used for |
|---|---|---|---|
| `yolo26n/s/m.pt` | Ultralytics, arXiv:2606.03748 | AGPL-3.0, enterprise option available | Primary detector. **Licence review required before any commercial ELCIA pilot.** |
| `UVH-26-MV-YOLOv11-S/X.pt` | `iisc-aim/UVH-26` on HuggingFace | Open-source per model card — verify exact terms before commercial use | Bengaluru-domain detector: fine-tuned on ~2,800 Safe City CCTV cameras, 14 India-specific classes |

## Video used for development and evaluation

| Asset | Source | Licence | Used for |
|---|---|---|---|
| `data/problems/Accidents/*.mp4` (15) | Supplied by the team | Not stated — treated as private, not redistributed | Positives in the labelled evaluation set |
| `data/problems/Traffic/*.mp4` (16) | Supplied by the team | Not stated — treated as private, not redistributed | Negatives — confirmed crash-free |
| `collision_rearend_jilin.webm` | Wikimedia Commons | **Public domain** | Real rear-end collision, development |
| `india_cuttack_linkroad.webm` | Wikimedia Commons | **CC BY-SA 3.0** | Indian urban traffic, calibration development |
| `india_gangtok_congestion.webm` | Wikimedia Commons | **CC BY-SA 4.0** | Indian congestion, queue-engine development |

Per-file attribution (author, licence, source URL) is recorded in
`data/raw/wikimedia_manifest.json`.

## Corpora identified for evaluation and future fine-tuning

| Corpus | Scale | Access | Relevance |
|---|---|---|---|
| **AI City 2021 Track 4** | 100 train + 150 test videos, ~15 min each, 410p | Direct Google Drive — no form, no password | The benchmark four of our six reference papers compete on. Downloaded, 15.1 GB. Anomalies are stalled vehicles and crashes; ordinary congestion is explicitly excluded. |
| **ACCIDENT (2026)** | 2,027 real + 2,211 CARLA clips | Kaggle · annotations CC BY 4.0, code Apache-2.0 | Current CCTV accident benchmark with when/where/what labels and IID, geo-OOD, zero-shot splits |
| **UVH-26** | 26,646 images, 1.8M boxes | HuggingFace | Bengaluru Safe City CCTV, 14 Indian vehicle classes |
| **BMD-45** | large-scale | HuggingFace, `iisc-aim/BMD-45` | Bengaluru Mobility Dataset, operational CCTV |
| **UA-DETRAC** | 100 sequences, 1.21M boxes | Roboflow mirror is CC BY 4.0 — **verify the version taken** | Tracking and congestion; covers what Track 4 excludes |
| **SO-TAD** | 2,186 samples, 282 accidents | GitHub; 25.5 GB via Baidu | Surveillance-oriented accident benchmark |
| **TU-DAT** | ~280 clips | GitHub — **research-only, non-commercial** | Aggressive driving: tailgating, weaving, speeding |
| **DriveIndia / IDD** | 66,986 / ~10k images | Public | Indian roads, but **dashcam viewpoint** — superseded by UVH-26 for this fixed-camera use case |

## Synthetic controls

Wrong-side movement has no reliable public corpus, so ground truth is generated
and **labelled as synthetic** wherever it is reported:

- **Time-reversal control** — a one-way corridor played backwards. Every vehicle
  is then genuinely travelling against the configured direction, with exact ground
  truth and zero labelling cost. Yields recall on reversed clips and
  false-positive rate on forward clips.
- **Corridor-swap control** — one corridor's direction vector deliberately
  inverted, testing the geometry independently of the tracker.

A synthetic control clearly labelled as one is rigour. Presented as real footage
it would be the opposite.
