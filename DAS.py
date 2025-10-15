# DAS.py
import numpy as np
import h5py
from datetime import datetime, timezone
import Utilities

def _decode_attr(val):
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8")
        except Exception:
            return str(val)
    return val

def _parse_start_time_attr(raw):
    """
    Parse common DAS time forms into tz-aware UTC datetime:
      - ISO 8601 strings/bytes, possibly with 'Z' and >6-digit fractional seconds
      - numeric epochs (s/ms/us/ns)
      - numpy.datetime64
    Returns datetime|None (UTC).
    """
    if raw is None:
        return None

    # numpy.datetime64?
    if isinstance(raw, (np.datetime64,)):
        try:
            ns = raw.astype('datetime64[ns]').astype('int64')
            sec, nsec = divmod(ns, 1_000_000_000)
            return datetime.fromtimestamp(sec, tz=timezone.utc).replace(microsecond=nsec // 1000)
        except Exception:
            pass

    # bytes/str -> ISO handling with long fractional seconds
    val = _decode_attr(raw)
    if isinstance(val, str):
        s = val.strip()
        try:
            # normalize trailing Z
            if s.endswith('Z'):
                s = s[:-1] + '+00:00'
            # if there is a fractional second with >6 digits, truncate
            # find the time part up to timezone sign (+/-) after date 'YYYY-MM-DD'
            if 'T' in s:
                # split off timezone offset if present (keeps sign)
                main, tz = s, ''
                plus = s.rfind('+')
                minus = s.rfind('-')
                cut = max(plus, minus if minus > 9 else -1)  # ignore date hyphens
                if cut > 9:
                    main, tz = s[:cut], s[cut:]
                # trim fractional seconds in main
                if '.' in main:
                    head, frac = main.split('.', 1)
                    frac_digits = ''.join(ch for ch in frac if ch.isdigit())
                    main = head + ('.' + frac_digits[:6] if frac_digits else '')
                s_for_dt = main + tz
            else:
                s_for_dt = s
            dt = datetime.fromisoformat(s_for_dt)
            # ensure tz-aware UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except Exception:
            pass

    # numeric epoch?
    if isinstance(val, (int, float, np.integer, np.floating)):
        x = float(val)
        # heuristics by magnitude
        if x > 1e17:    # ns
            sec, nsec = divmod(int(x), 1_000_000_000)
            return datetime.fromtimestamp(sec, tz=timezone.utc).replace(microsecond=nsec // 1000)
        elif x > 1e14:  # us
            sec, usec = divmod(int(x), 1_000_000)
            return datetime.fromtimestamp(sec, tz=timezone.utc).replace(microsecond=usec)
        elif x > 1e11:  # ms
            sec, msec = divmod(int(x), 1_000)
            return datetime.fromtimestamp(sec, tz=timezone.utc).replace(microsecond=msec * 1000)
        else:           # s
            return datetime.fromtimestamp(x, tz=timezone.utc)

    # last try: numpy conversion from stringy inputs
    try:
        dt64 = np.datetime64(val)
        ns = dt64.astype('datetime64[ns]').astype('int64')
        sec, nsec = divmod(ns, 1_000_000_000)
        return datetime.fromtimestamp(sec, tz=timezone.utc).replace(microsecond=nsec // 1000)
    except Exception:
        return None



class DAS:
    """
    Multi-vendor DAS reader.
    Provides:
      - self.data: ndarray [channels, time]
      - self.meta: dict with 'fs','dt','dx','start_time_dt','start_time_iso', and light extras
    """
    def __init__(self, file, select_channels=None, vendor="OptaSense"):
        self.file = file
        self.vendor = vendor
        self.select_channels = select_channels  # None => read all channels
        self.data = None
        self.meta = {}

        if str(vendor).lower() == "optasense":
            self._read_optasense()
        elif str(vendor).lower() == "silixa":
            self._read_silixa()
        else:
            raise ValueError(f"Unknown vendor '{vendor}'. Use 'OptaSense' or 'Silixa'.")

    # ----------------------
    # OptaSense reader
    # ----------------------
    def _read_optasense(self):
        RAW_PATH = "Acquisition/Raw[0]/RawData"
        RAW0_GRP = "Acquisition/Raw[0]"
        ACQ_GRP  = "Acquisition"

        with h5py.File(self.file, "r") as f:
            raw_ds = f[RAW_PATH]              # (time, loci)
            arr = raw_ds[...].T               # -> (channels, time)
            if self.select_channels is not None:
                arr = arr[self.select_channels, :]
            self.data = arr

            dx = float(f[ACQ_GRP].attrs["SpatialSamplingInterval"])
            fs = float(f[RAW0_GRP].attrs["OutputDataRate"])
            dt = 1.0 / fs

            start_dt = _parse_start_time_attr(f[ACQ_GRP].attrs.get("MeasurementStartTime"))
            if start_dt is None and "Acquisition/Raw[0]/RawDataTime" in f:
                start_dt = _parse_start_time_attr(f["Acquisition/Raw[0]/RawDataTime"].attrs.get("StartTime"))

            part_start_dt = _parse_start_time_attr(raw_ds.attrs.get("PartStartTime"))
            part_end_dt   = _parse_start_time_attr(raw_ds.attrs.get("PartEndTime"))

            self.meta.update({
                "dx": dx,
                "fs": fs,
                "dt": dt,
                "start_time_dt": start_dt,
                "start_time_iso": None if start_dt is None else start_dt.isoformat(),
                "part_start_time_dt": part_start_dt,
                "part_end_time_dt":   part_end_dt,
                "vendor": _decode_attr(f[ACQ_GRP].attrs.get("VendorCode")),
                "gauge_length_m": float(f[ACQ_GRP].attrs.get("GaugeLength", np.nan)),
                "raw_unit": _decode_attr(f[RAW0_GRP].attrs.get("RawDataUnit")),
                "raw_description": _decode_attr(f[RAW0_GRP].attrs.get("RawDescription")),
                "raw_path": RAW_PATH,
                "raw_dtype": str(raw_ds.dtype),
                "raw_shape": tuple(raw_ds.shape),  # original (time, loci)
            })

    # ----------------------
    # Silixa (DAS-RCN style) reader
    # ----------------------
    def _read_silixa(self):
        # Paths from your example
        META_ROOT   = "DasMetadata"
        META_ACQ    = "DasMetadata/Interrogator/Acquisition"
        RAW_GRP     = "DasRawData"
        RAW_PATH    = "DasRawData/RawData"        # (time step, locus)
        TIME_ARRAY  = "DasRawData/DasTimeArray"   # (time,), attrs StartTime/EndTime

        with h5py.File(self.file, "r") as f:
            raw_ds = f[RAW_PATH]                  # (time, locus)
            arr = raw_ds[...].T                   # -> (channels, time)
            if self.select_channels is not None:
                arr = arr[self.select_channels, :]
            self.data = arr

            # Core meta
            # AcquisitionSampleRate can be a string; SpatialSamplingInterval too.
            fs_raw = f[META_ACQ].attrs.get("AcquisitionSampleRate")
            dx_raw = f[META_ACQ].attrs.get("SpatialSamplingInterval")
            fs = float(_decode_attr(fs_raw))      # e.g., '1000' -> 1000.0
            dx = float(_decode_attr(dx_raw))      # e.g., '1.021' -> 1.021
            dt = 1.0 / fs

            # Start times (prefer explicit StartTime from DasTimeArray; else AcquisitionStartTime)
            part_start_dt = _parse_start_time_attr(f[TIME_ARRAY].attrs.get("StartTime"))
            part_end_dt   = _parse_start_time_attr(f[TIME_ARRAY].attrs.get("EndTime"))

            start_dt = _parse_start_time_attr(f[META_ACQ].attrs.get("AcquisitionStartTime"))
            if start_dt is None:
                start_dt = part_start_dt

            # Light extras
            vendor_name = _decode_attr(f.get(f"{META_ROOT}/Interrogator", {}).attrs.get("InterrogatorManufacturer")) \
                          if f"{META_ROOT}/Interrogator" in f else "Silixa"

            self.meta.update({
                "dx": dx,
                "fs": fs,
                "dt": dt,
                "start_time_dt": start_dt,
                "start_time_iso": None if start_dt is None else start_dt.isoformat(),
                "part_start_time_dt": part_start_dt,
                "part_end_time_dt":   part_end_dt,
                "vendor": vendor_name,
                "gauge_length_m": float(_decode_attr(f[META_ACQ].attrs.get("GaugeLength", "nan"))) if META_ACQ in f else np.nan,
                "raw_unit": None,  # not standardized here; can be added if present
                "raw_description": None,
                "raw_path": RAW_PATH,
                "raw_dtype": str(raw_ds.dtype),
                "raw_shape": tuple(raw_ds.shape),  # original (time, locus)
            })

    # ----------------------
    # Plot wrappers (unchanged)
    # ----------------------
    def plot(self, start_time=None, end_time=None, title=None, preprocess=None, target_fs=None):
        from preprocessing import make_preprocess
        default_pp = make_preprocess(f_lo=1.0, f_hi=20.0, order=5)

        data = default_pp(self.data, self.meta["fs"]) if preprocess is None else preprocess(self.data, self.meta["fs"])
        if target_fs is not None:
            data = Utilities.downsample_data(data, self.meta["fs"], target_fs)
            dt_plot = 1.0 / target_fs
        else:
            dt_plot = self.meta["dt"]
        channels = np.arange(data.shape[0]) if self.select_channels is None else np.asarray(self.select_channels)
        # default title: part start time if available
        if title is None:
            title = self.meta.get("part_start_time_dt", self.meta.get("start_time_dt"))
        Utilities.plot_das_data(data, channels, self.meta["dx"], dt_plot, start_time, end_time, title=title)

    def plot_single(self, ch, start_time=None, end_time=None, preprocess=None):
        import Utilities
        data = self.data if preprocess is None else preprocess(self.data, self.meta["fs"])
        Utilities.plot_single(data, ch, self.meta["dx"], self.meta["dt"], start_time, end_time)


# ----------------------
# MulDAS: pass vendor through
# ----------------------
class MulDAS(DAS):
    def __init__(self, file_list, select_channels=None, vendor="OptaSense"):
        self.files = list(file_list)  # original, unsorted
        self.vendor = vendor
        self.ordered_files = None
        self.file = None
        self.select_channels = select_channels
        self.data = None
        self.meta = {}
        self._load_sort_concat()

    def _load_sort_concat(self):
        # read once per file with the chosen vendor reader
        items = []
        for f in self.files:
            d = DAS(f, select_channels=self.select_channels, vendor=self.vendor)
            key_dt = d.meta.get("part_start_time_dt") or d.meta.get("start_time_dt")
            if key_dt is None:
                raise ValueError(f"{f}: missing per-file start time")
            items.append((key_dt, d))

        if not items:
            raise ValueError("MulDAS: empty file list.")

        # sort by per-part start time, tie-break by filename
        items.sort(key=lambda tup: (tup[0], tup[1].file))
        self.ordered_files = [d.file for (_, d) in items]
        self.file = self.ordered_files  # compatibility alias

        # sanity checks
        first = items[0][1]
        fs0 = first.meta["fs"]; dx0 = first.meta["dx"]; ch0 = first.data.shape[0]
        for _, d in items[1:]:
            if d.data.shape[0] != ch0:
                raise ValueError(f"Channel mismatch: first={ch0}, {d.file}={d.data.shape[0]}")
            if not np.isclose(d.meta["fs"], fs0, atol=1e-9):
                raise ValueError(f"fs mismatch: {fs0} vs {d.meta['fs']} in {d.file}")
            if not np.isclose(d.meta["dx"], dx0, atol=1e-12):
                raise ValueError(f"dx mismatch: {dx0} vs {d.meta['dx']} in {d.file}")

        # concat along time
        self.data = np.concatenate([d.data for (_, d) in items], axis=1)

        # unified meta
        self.meta = {
            "dx": dx0,
            "fs": fs0,
            "dt": 1.0 / fs0,
            "start_time_dt": items[0][1].meta.get("part_start_time_dt") or first.meta.get("start_time_dt"),
            "start_time_iso": (items[0][1].meta.get("part_start_time_dt") or first.meta.get("start_time_dt")).isoformat(),
            "part_start_time_dt": first.meta.get("start_time_dt"),
            "vendor": first.meta.get("vendor"),
            "gauge_length_m": first.meta.get("gauge_length_m"),
            "raw_unit": first.meta.get("raw_unit"),
            "raw_description": first.meta.get("raw_description"),
        }
