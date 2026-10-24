from pathlib import Path

import neurokit2 as nk  # type: ignore
import numpy as np
import sleepecg  # type: ignore
import wfdb  # type: ignore

database_name = "mimic3wdb/1.0"


def has_continuous_constant_values(data, threshold=5):
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
            if count > threshold:
                return True
        else:
            count = 1
    return False


def split_segment(segment_path, split_sec=5):
    segment = Path(segment_path)
    segment_name = segment.name
    segment_dir = f"{database_name}/{segment.parent}".replace("\\", "/")
    segment_data = wfdb.rdrecord(record_name=segment_name, pn_dir=segment_dir)

    # Exact "II": a substring match also selects lead III.
    ppg_sig_no = abp_sig_no = ecg_sig_no = None
    for sig_no, sig_name in enumerate(segment_data.sig_name):
        if "Pleth" in sig_name or "PLETH" in sig_name:
            ppg_sig_no = sig_no
        if "ABP" in sig_name:
            abp_sig_no = sig_no
        if sig_name.upper() == "II":
            ecg_sig_no = sig_no
    if ppg_sig_no is None or abp_sig_no is None or ecg_sig_no is None:
        raise ValueError(
            f"{segment_name} missing Pleth, ABP, or II; has {segment_data.sig_name}"
        )

    ppg = segment_data.p_signal[:, ppg_sig_no]
    abp = segment_data.p_signal[:, abp_sig_no]
    ecg = segment_data.p_signal[:, ecg_sig_no]
    fs = segment_data.fs

    ecg = nk.signal_fillmissing(ecg)
    ppg = nk.signal_fillmissing(ppg)
    abp = nk.signal_fillmissing(abp)

    ecg = nk.signal_filter(ecg, fs, 0.6)
    ppg = nk.signal_filter(ppg, fs, 0.7, 10)
    abp = nk.signal_filter(abp, fs, 0.5, 20)

    split_samples = int(fs * split_sec)
    num_segments = len(ppg) // split_samples

    valid_segments = []

    for i in range(num_segments):
        start_idx = i * split_samples
        end_idx = start_idx + split_samples

        ppg_segment = ppg[start_idx:end_idx]
        abp_segment = abp[start_idx:end_idx]
        ecg_segment = ecg[start_idx:end_idx]

        if (
            not np.any(np.isnan(ppg_segment))
            and not np.any(np.isnan(abp_segment))
            and not np.any(np.isnan(ecg_segment))
            and not has_continuous_constant_values(ppg_segment)
            and not has_continuous_constant_values(abp_segment)
            and not has_continuous_constant_values(ecg_segment)
        ):
            valid_segments.append(
                {"ECG": ecg_segment, "PPG": ppg_segment, "ABP": abp_segment}
            )

    return valid_segments, fs


def calculate_hr(corrected_peaks, fs):
    peak_intervals = np.diff(corrected_peaks) / fs
    hr = 60 / np.mean(peak_intervals) if len(peak_intervals) > 0 else np.nan
    return hr


def calculate_peak_ratio(signal, corrected_peaks, method="min"):
    peak_amplitudes = signal[corrected_peaks]
    max_peak = np.max(peak_amplitudes) if len(peak_amplitudes) > 0 else np.nan
    if method == "min":
        min_peak = np.min(peak_amplitudes) if len(peak_amplitudes) > 0 else np.nan
        peak_ratio = max_peak / min_peak if min_peak != 0 else np.nan
    else:
        mean_peak = np.mean(peak_amplitudes) if len(peak_amplitudes) > 0 else np.nan
        peak_ratio = max_peak / mean_peak if mean_peak != 0 else np.nan

    return peak_ratio


def calculate_btb_ratio(corrected_peaks, fs):
    btb_intervals = np.diff(corrected_peaks) / fs
    max_btb = np.max(btb_intervals) if len(btb_intervals) > 0 else np.nan
    min_btb = np.mean(btb_intervals) if len(btb_intervals) > 0 else np.nan

    btb_ratio = max_btb / min_btb if min_btb != 0 else np.nan

    return btb_ratio


def calculate_pulse_pressure(abp_signal, corrected_peaks, corrected_valleys):
    systolic_pressure = (
        np.mean(abp_signal[corrected_peaks]) if len(corrected_peaks) > 0 else np.nan
    )
    diastolic_pressure = (
        np.mean(abp_signal[corrected_valleys]) if len(corrected_valleys) > 0 else np.nan
    )

    pulse_pressure = (
        systolic_pressure - diastolic_pressure
        if np.isfinite(systolic_pressure) and np.isfinite(diastolic_pressure)
        else np.nan
    )

    return pulse_pressure, systolic_pressure, diastolic_pressure


def correct_peaks(signal, peak_inds, search_radius):
    corrected_peak_inds = []

    for peak in peak_inds:
        start = max(peak - search_radius, 0)
        end = min(peak + search_radius, len(signal) - 1)
        max_peak_in_range = np.argmax(signal[start : end + 1])
        corrected_peak_inds.append(start + max_peak_in_range)

    return np.array(corrected_peak_inds)


def signal_selection_algorithm(groups, fs):
    valid_records = []

    for group in groups:
        ecg = group["ECG"]
        ppg = group["PPG"]
        abp = group["ABP"]

        ecg_peak_indices = sleepecg.detect_heartbeats(ecg, fs=fs)
        ppg_peak_indices = sleepecg.detect_heartbeats(ppg, fs=fs)
        abp_peak_indices = sleepecg.detect_heartbeats(abp, fs=fs)
        abp_valley_indices = sleepecg.detect_heartbeats(-abp, fs=fs)

        search_radius = int(0.33 * fs)
        corrected_ecg_peaks = correct_peaks(ecg, ecg_peak_indices, search_radius)
        corrected_ppg_peaks = correct_peaks(ppg, ppg_peak_indices, search_radius)
        corrected_abp_peaks = correct_peaks(abp, abp_peak_indices, search_radius)
        corrected_abp_valleys = correct_peaks(-abp, abp_valley_indices, search_radius)

        hr_ecg = calculate_hr(corrected_ecg_peaks, fs)
        hr_ppg = calculate_hr(corrected_ppg_peaks, fs)

        ppg_peak_ratio = calculate_peak_ratio(ppg, corrected_ppg_peaks)
        ecg_peak_ratio = calculate_peak_ratio(ecg, corrected_ecg_peaks, method="mean")
        ppg_btb_ratio = calculate_btb_ratio(corrected_ppg_peaks, fs)
        ecg_btb_ratio = calculate_btb_ratio(corrected_ecg_peaks, fs)

        pulse_pressure, true_sbp, true_dbp = calculate_pulse_pressure(
            abp, corrected_abp_peaks, corrected_abp_valleys
        )

        if (
            abs(hr_ppg - hr_ecg) <= 5
            and 40 < hr_ppg < 180
            and ppg_peak_ratio < 1.5
            and ecg_peak_ratio < 1.5
            and ppg_btb_ratio < 1.5
            and ecg_btb_ratio < 1.5
            and 20 < pulse_pressure < 60
        ):
            group["true_sbp"] = true_sbp
            group["true_dbp"] = true_dbp
            valid_records.append(group)

    return valid_records
