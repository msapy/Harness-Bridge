"""
data_segmenter.py
=================
DataSegmenter class: loads multi-channel accelerometer CSV files recorded from the
Harness Bridge monitoring campaign and divides the continuous time-history into
short, overlapping segments suitable for per-window system identification.

Usage
-----
>>> seg = DataSegmenter(fs=128.0, segment_length_s=20.0, overlap=0.5)
>>> seg.load_csv("path/to/day1 Clean.csv")
>>> segments = seg.get_segments()
>>> # Each element: {'data': np.ndarray(N_samples, n_channels),
>>> #                'start_sample': int,
>>> #                'start_time': str,
>>> #                'seg_index': int}
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The CSV format produced by SensorConnect/Dewesoft has 25 header rows before
# the actual column headers and data begin.
CSV_HEADER_ROWS = 25

# Time column name as it appears in every clean CSV.
TIME_COL = "Time"


class DataSegmenter:
    """
    Loads a multi-channel structural acceleration CSV and segments it into
    overlapping short-duration windows.

    Parameters
    ----------
    fs : float
        Nominal sampling frequency in Hz (default 128.0 for Harness Bridge).
    segment_length_s : float
        Duration of each segment in seconds (default 20 s - matches Bridge 2
        in Tran & Ozer 2020; increase to 60 s for richer frequency resolution).
    overlap : float
        Fractional overlap between consecutive segments [0, 1).
        0.5 means 50 % overlap (half the segment is shared with next window).
    skiprows : int
        Number of metadata header rows to skip before column names appear.
    detrend : bool
        If True, subtract the mean from each channel within every segment
        (removes quasi-static offset and DC bias).
    max_rows : int or None
        If set, limits the number of data rows loaded (useful for quick tests).
    """

    def __init__(
        self,
        fs: float = 128.0,
        segment_length_s: float = 20.0,
        overlap: float = 0.5,
        skiprows: int = CSV_HEADER_ROWS,
        detrend: bool = True,
        max_rows: Optional[int] = None,
    ):
        if not (0.0 <= overlap < 1.0):
            raise ValueError("overlap must be in [0, 1)")

        self.fs = fs
        self.segment_length_s = segment_length_s
        self.overlap = overlap
        self.skiprows = skiprows
        self.detrend = detrend
        self.max_rows = max_rows

        # Derived quantities
        self.n_samples_per_seg: int = int(round(segment_length_s * fs))
        self.hop_size: int = int(round(self.n_samples_per_seg * (1.0 - overlap)))

        # Set after load_csv()
        self._raw_data: Optional[np.ndarray] = None   # (N, m) float64
        self._time_strings: Optional[pd.Series] = None
        self._sensor_names: Optional[List[str]] = None
        self._n_samples: int = 0
        self._n_channels: int = 0
        self._csv_path: str = ""

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def load_csv(self, csv_path: str) -> "DataSegmenter":
        """
        Read the CSV file produced by SensorConnect and cache raw data.

        The file format is:
            rows 0-24  : metadata (FILE_INFO, SESSION_INFO, ...)
            row  25    : column headers (Time, 29124:ch3, ...)
            rows 26+   : numeric data

        Parameters
        ----------
        csv_path : str or Path
            Absolute or relative path to a ``*Clean.csv`` file.

        Returns
        -------
        self  (for method chaining)
        """
        csv_path = str(csv_path)
        self._csv_path = csv_path

        print(f"[DataSegmenter] Loading: {Path(csv_path).name}")

        nrows = self.max_rows if self.max_rows else None

        # Auto-detect skip rows: scan for the header line that begins with 'Time'
        # Some files (days 9/10) have an extra 'DATA_START' row shifting headers by 1.
        skiprows = self.skiprows
        try:
            with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh):
                    if line.strip().startswith("Time,") or line.strip().startswith("Time\t"):
                        skiprows = i   # this line IS the header → skip everything before it
                        break
                    if i > 50:        # give up after 50 rows
                        break
        except OSError:
            pass   # fall back to self.skiprows

        df = pd.read_csv(csv_path, skiprows=skiprows, nrows=nrows)

        # Identify sensor columns (everything except Time / ParsedTime)
        sensor_cols = [c for c in df.columns if c not in (TIME_COL, "ParsedTime")]
        self._sensor_names = sensor_cols
        self._n_channels = len(sensor_cols)

        # Drop rows where any sensor reading is NaN
        df = df.dropna(subset=sensor_cols).reset_index(drop=True)

        self._time_strings = df[TIME_COL].astype(str)
        self._raw_data = df[sensor_cols].values.astype(np.float64)  # (N, m)
        self._n_samples = len(self._raw_data)

        dur_s = self._n_samples / self.fs
        n_segs = max(0, (self._n_samples - self.n_samples_per_seg) // self.hop_size + 1)

        print(
            f"[DataSegmenter]  Channels : {self._n_channels}  ({', '.join(sensor_cols)})"
        )
        print(
            f"[DataSegmenter]  Samples  : {self._n_samples:,}  ({dur_s/60:.1f} min)"
        )
        print(
            f"[DataSegmenter]  Segments : {n_segs}  "
            f"({self.segment_length_s:.0f}s, {self.overlap*100:.0f}% overlap)"
        )

        return self

    def get_segments(self) -> List[Dict]:
        """
        Return a list of segment dictionaries, one per window.

        Each dictionary contains:
            'data'         : np.ndarray of shape (n_samples_per_seg, n_channels)
                             Optionally zero-mean (detrended) if self.detrend=True.
            'start_sample' : int - index into the full record where this window starts.
            'start_time'   : str - timestamp string at window start.
            'seg_index'    : int - sequential segment number (0-based).

        Returns
        -------
        List[Dict]
        """
        if self._raw_data is None:
            raise RuntimeError("Call load_csv() before get_segments().")

        segments: List[Dict] = []
        start = 0
        seg_idx = 0

        while start + self.n_samples_per_seg <= self._n_samples:
            end = start + self.n_samples_per_seg
            chunk = self._raw_data[start:end, :].copy()   # (N_seg, m)

            # Subtract column-wise mean to remove DC offset / slow drift
            if self.detrend:
                chunk -= chunk.mean(axis=0)

            segments.append(
                {
                    "data": chunk,
                    "start_sample": start,
                    "start_time": self._time_strings.iloc[start],
                    "seg_index": seg_idx,
                }
            )

            start += self.hop_size
            seg_idx += 1

        return segments

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def sensor_names(self) -> List[str]:
        """List of sensor column names, set after load_csv()."""
        if self._sensor_names is None:
            raise RuntimeError("Call load_csv() first.")
        return self._sensor_names

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def n_segments(self) -> int:
        if self._raw_data is None:
            return 0
        return max(
            0,
            (self._n_samples - self.n_samples_per_seg) // self.hop_size + 1,
        )

    def __repr__(self) -> str:
        return (
            f"DataSegmenter(fs={self.fs}, seg={self.segment_length_s}s, "
            f"overlap={self.overlap}, channels={self._n_channels}, "
            f"n_segs={self.n_segments})"
        )
