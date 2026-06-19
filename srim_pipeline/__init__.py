"""
srim_pipeline
=============
Simultaneous Modal Parameter Anomaly Detection Pipeline for Structural Health Monitoring.

Implementation of Tran & Ozer (2020), Sensors 20(17):4752.
Applied to Harness Bridge 10-day accelerometer dataset.

Classes:
    DataSegmenter   - overlapping time-window segmentation
    SRIMIdentifier  - output-only SRIM state-space system identification
    ModalClusterer  - stabilization, MPC clearance, hierarchical clustering, MAD
    AnomalyDetector - Gaussian anomaly detection with k-sigma thresholds
    MetricsEvaluator- TPR/TNR/PPV/NPV/FPR/FDR/FNR/FOR evaluation

Runner:
    srim_shm_pipeline.py  - orchestrates the full pipeline
"""

from .data_segmenter   import DataSegmenter
from .srim_identifier  import SRIMIdentifier
from .modal_clusterer  import ModalClusterer
from .anomaly_detector import AnomalyDetector
from .metrics_evaluator import MetricsEvaluator

__all__ = [
    "DataSegmenter",
    "SRIMIdentifier",
    "ModalClusterer",
    "AnomalyDetector",
    "MetricsEvaluator",
]
