from pathlib import Path

import neurokit2 as nk  # type: ignore
import numpy as np
import sleepecg  # type: ignore
import wfdb  # type: ignore

# database_name = 'mimic4wdb/0.1.0'
database_name = "mimic3wdb/1.0"


def has_continuous_constant_values(data, threshold=5):
    count = 1  # Initialize counter
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
            if count > threshold:  # Check if count exceeds the threshold
                return True
        else:
            count = 1  # Reset counter if values differ
    return False


def split_segment(segment_path, split_sec=5):
    # Read the segment data
    segment = Path(segment_path)
    segment_name = segment.name
    segment_dir = f"{database_name}/{segment.parent}".replace("\\", "/")
    segment_data = wfdb.rdrecord(record_name=segment_name, pn_dir=segment_dir)

    # Identify signal indices for Pleth (PPG), ABP, and lead II.
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

    # Extract signal data
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
        # Split each signal into 5-second segments
        start_idx = i * split_samples
        end_idx = start_idx + split_samples

        ppg_segment = ppg[start_idx:end_idx]
        abp_segment = abp[start_idx:end_idx]
        ecg_segment = ecg[start_idx:end_idx]

        # Check for NaN or flat signals in any of the segments
        if (
            not np.any(np.isnan(ppg_segment))
            and not np.any(np.isnan(abp_segment))
            and not np.any(np.isnan(ecg_segment))
            and not has_continuous_constant_values(ppg_segment)
            and not has_continuous_constant_values(abp_segment)
            and not has_continuous_constant_values(ecg_segment)
        ):
            # If valid, add to the list of valid segments
            valid_segments.append(
                {"ECG": ecg_segment, "PPG": ppg_segment, "ABP": abp_segment}
            )

    return valid_segments, fs


def calculate_hr(corrected_peaks, fs):
    """
    Calculate heart rate from corrected peaks.

    Parameters:
    corrected_peaks (array-like): Indices of the corrected peaks.
    fs (int): Sampling frequency.

    Returns:
    hr (float): The heart rate in beats per minute (bpm).
    """
    # Calculate time differences between corrected peaks
    peak_intervals = np.diff(corrected_peaks) / fs

    # Calculate heart rate (HR) in beats per minute (bpm)
    hr = 60 / np.mean(peak_intervals) if len(peak_intervals) > 0 else np.nan

    return hr


def calculate_peak_ratio(signal, corrected_peaks, method="min"):
    """
    Calculate the peak-to-peak amplitude ratio using the corrected peaks.

    Parameters:
    signal (array-like): The input signal (e.g., ECG or PPG).
    corrected_peaks (array-like): Indices of the corrected peaks.

    Returns:
    peak_ratio (float): Ratio of maximum peak to the mean peak height.
    """
    # Extract the peak amplitudes from the signal using corrected peak indices
    peak_amplitudes = signal[corrected_peaks]

    # Calculate the max and mean peak amplitude
    max_peak = np.max(peak_amplitudes) if len(peak_amplitudes) > 0 else np.nan
    if method == "min":
        min_peak = np.min(peak_amplitudes) if len(peak_amplitudes) > 0 else np.nan
        peak_ratio = max_peak / min_peak if min_peak != 0 else np.nan
    else:
        mean_peak = np.mean(peak_amplitudes) if len(peak_amplitudes) > 0 else np.nan
        peak_ratio = max_peak / mean_peak if mean_peak != 0 else np.nan

    return peak_ratio


def calculate_btb_ratio(corrected_peaks, fs):
    """
    Calculate the beat-to-beat (BTB) interval ratio using corrected peaks.

    Parameters:
    corrected_peaks (array-like): Indices of the corrected peaks.
    fs (int): Sampling frequency.

    Returns:
    btb_ratio (float): Ratio of the maximum to the mean beat-to-beat intervals.
    """
    # Calculate the beat-to-beat intervals from the corrected peaks
    btb_intervals = np.diff(corrected_peaks) / fs

    # Calculate the max and mean beat-to-beat intervals
    max_btb = np.max(btb_intervals) if len(btb_intervals) > 0 else np.nan
    min_btb = np.mean(btb_intervals) if len(btb_intervals) > 0 else np.nan

    btb_ratio = max_btb / min_btb if min_btb != 0 else np.nan

    return btb_ratio


def calculate_pulse_pressure(abp_signal, corrected_peaks, corrected_valleys):
    """
    Calculate pulse pressure using corrected ABP peaks and valleys.

    Parameters:
    abp_signal (array-like): The arterial blood pressure signal (ABP).
    corrected_peaks (array-like): Indices of the corrected ABP peaks.
    corrected_valleys (array-like): Indices of the corrected ABP valleys.

    Returns:
    pulse_pressure (float): Pulse pressure, calculated as the difference between systolic and diastolic pressures.
    """
    # Extract systolic (peak) and diastolic (valley) pressures
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
        # Define the search range, ensuring it stays within the signal bounds
        start = max(peak - search_radius, 0)
        end = min(peak + search_radius, len(signal) - 1)

        # Find the index of the maximum value within the search range
        max_peak_in_range = np.argmax(
            signal[start : end + 1]
        )  # +1 because of the inclusive end index
        corrected_peak_inds.append(start + max_peak_in_range)

    return np.array(corrected_peak_inds)


def signal_selection_algorithm(groups, fs):
    """
    Input: groups: List of dictionaries with 'ECG', 'PPG', 'ABP' signals from split_segment function
                    fs: Sampling frequency
    Output: List of valid records (those with record_quality = 1)
    """
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

        # Calculate heart rates from corrected ECG and PPG peaks
        hr_ecg = calculate_hr(corrected_ecg_peaks, fs)
        hr_ppg = calculate_hr(corrected_ppg_peaks, fs)

        # Calculate peak and BTB ratios from corrected peaks
        ppg_peak_ratio = calculate_peak_ratio(ppg, corrected_ppg_peaks)
        ecg_peak_ratio = calculate_peak_ratio(ecg, corrected_ecg_peaks, method="mean")
        ppg_btb_ratio = calculate_btb_ratio(corrected_ppg_peaks, fs)
        ecg_btb_ratio = calculate_btb_ratio(corrected_ecg_peaks, fs)

        # Calculate pulse pressure from ABP signal, corrected peaks, and valleys
        pulse_pressure, true_sbp, true_dbp = calculate_pulse_pressure(
            abp, corrected_abp_peaks, corrected_abp_valleys
        )

        # Apply selection criteria from Algorithm 1
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
            valid_records.append(group)  # Valid record

    return valid_records
