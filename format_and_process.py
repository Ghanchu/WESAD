import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch, find_peaks  # welch kept for EMG PSD
from scipy.integrate import trapezoid
import neurokit2 as nk
import time
import warnings
warnings.filterwarnings('ignore')

FS = 700
WINDOW_60S = 60 * FS    # 42000 samples
WINDOW_5S  =  5 * FS    # 3500 samples
SHIFT      = int(0.25 * FS)  # 175 samples


# 1=baseline, 2=stress, 3=amusement  (0/4/6/7 = transient or meditation)
VALID_LABELS = {1, 2, 3}


# ── filters ────────────────────────────────────────────────────────────────

def _lp(signal, cutoff, order=4):
    sos = butter(order, cutoff / (FS / 2), btype='low', output='sos')
    return sosfiltfilt(sos, signal)

def _hp(signal, cutoff, order=4):
    sos = butter(order, cutoff / (FS / 2), btype='high', output='sos')
    return sosfiltfilt(sos, signal)

def _bp(signal, low, high, order=4):
    sos = butter(order, [low / (FS / 2), high / (FS / 2)], btype='band', output='sos')
    return sosfiltfilt(sos, signal)

def _band_power(freqs, psd, low, high):
    mask = (freqs >= low) & (freqs < high)
    return float(trapezoid(psd[mask], freqs[mask])) if mask.sum() >= 2 else 0.0


# ── per-window feature extractors ──────────────────────────────────────────

def _burg_psd(x, order=16, nfft=512, fs=4.0):
    """Burg AR method — better spectral resolution than Welch on short RR series."""
    x = x - x.mean()
    n = len(x)
    ef, eb = x.copy(), x.copy()
    a = np.zeros(order + 1)
    a[0] = 1.0
    P = float(np.dot(x, x)) / n
    for m in range(1, order + 1):
        efp, ebp = ef[1:], eb[:-1]
        den = np.dot(efp, efp) + np.dot(ebp, ebp)
        km  = -2.0 * np.dot(efp, ebp) / den if abs(den) > 1e-10 else 0.0
        ef, eb = efp + km * ebp, ebp + km * efp
        a_old = a[:m].copy()
        a[m]  = km
        for k in range(1, m):
            a[k] = a_old[k] + km * a_old[m - k]
        P *= (1.0 - km ** 2)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    h     = np.fft.rfft(a, n=nfft)
    psd   = (P / fs) / np.maximum(np.abs(h) ** 2, 1e-10)
    return freqs, psd


def _ecg_features(r_peaks_in_window):
    if len(r_peaks_in_window) < 4:
        return None

    rr = np.diff(r_peaks_in_window) / FS          # seconds
    hr = 60.0 / rr
    diff_rr = np.diff(rr)

    nn50 = int(np.sum(np.abs(diff_rr) > 0.050))
    pnn50 = nn50 / len(diff_rr) * 100.0 if len(diff_rr) > 0 else 0.0
    rmssd = float(np.sqrt(np.mean(diff_rr ** 2)))

    # TINN: least-squares triangular interpolation of NN histogram (HRV standard)
    rr_ms = rr * 1000.0
    bin_w = 1000.0 / 128.0
    min_b = np.floor(rr_ms.min() / bin_w) * bin_w
    max_b = np.ceil(rr_ms.max()  / bin_w) * bin_w + bin_w
    bins  = np.arange(min_b, max_b, bin_w)
    if len(bins) >= 3:
        hist, edges = np.histogram(rr_ms, bins=bins)
        centers   = (edges[:-1] + edges[1:]) / 2
        peak_idx  = int(np.argmax(hist))
        peak_h    = float(hist[peak_idx])
        peak_c    = centers[peak_idx]
        best_err, best_tinn = np.inf, 0.0
        for n_i in range(0, peak_idx + 1):
            for m_i in range(peak_idx, len(hist)):
                N, M = centers[n_i], centers[m_i]
                if M <= N:
                    continue
                tri = np.where(centers <= peak_c,
                               peak_h * (centers - N) / (peak_c - N + 1e-9),
                               peak_h * (M - centers) / (M - peak_c + 1e-9))
                tri = np.clip(tri, 0, None)
                tri[centers < N] = 0
                tri[centers > M] = 0
                err = float(np.sum((hist - tri) ** 2))
                if err < best_err:
                    best_err, best_tinn = err, M - N
        tinn = best_tinn
    else:
        tinn = 0.0

    # Frequency domain: interpolate RR to 4 Hz, then Welch
    rr_times = np.cumsum(rr) - rr[0]
    duration  = rr_times[-1]
    if duration < 10.0:
        ulf = lf = hf = uhf = lf_hf = total = rel = lfn = hfn = 0.0
    else:
        t_i  = np.arange(0, duration, 1 / 4.0)
        rr_i = np.interp(t_i, rr_times, rr)
        freqs, psd = _burg_psd(rr_i, order=16, nfft=512, fs=4.0)
        ulf   = _band_power(freqs, psd, 0.01, 0.04)
        lf    = _band_power(freqs, psd, 0.04, 0.15)
        hf    = _band_power(freqs, psd, 0.15, 0.40)
        uhf   = _band_power(freqs, psd, 0.40, 1.00)
        total = ulf + lf + hf + uhf
        lf_hf = lf / hf if hf > 0 else 0.0
        rel   = total
        lfn   = lf  / (lf + hf) * 100.0 if (lf + hf) > 0 else 0.0
        hfn   = hf  / (lf + hf) * 100.0 if (lf + hf) > 0 else 0.0

    return {
        'ECG_HR_mean':          float(np.mean(hr)),
        'ECG_HR_std':           float(np.std(hr)),
        'ECG_HRV_mean':         float(np.mean(rr)),
        'ECG_HRV_std':          float(np.std(rr)),
        'ECG_NN50':             nn50,
        'ECG_pNN50':            pnn50,
        'ECG_TINN':             tinn,
        'ECG_rmsHRV':           rmssd,
        'ECG_HRV_ULF':          ulf,
        'ECG_HRV_LF':           lf,
        'ECG_HRV_HF':           hf,
        'ECG_HRV_UHF':          uhf,
        'ECG_HRV_LF_HF_ratio':  lf_hf,
        'ECG_HRV_sum_ULF_HF':   total,
        'ECG_HRV_rel_power':    rel,
        'ECG_HRV_LFnorm':       lfn,
        'ECG_HRV_HFnorm':       hfn,
    }


def _eda_features(eda_w, scl_w, scr_w):
    eda = eda_w.flatten()
    scl = scl_w.flatten()
    scr = scr_w.flatten()
    t   = np.arange(len(eda)) / FS

    slope = float(np.polyfit(t, eda, 1)[0])
    scl_corr = float(np.corrcoef(t, scl)[0, 1]) if scl.std() > 0 else 0.0

    # SCR peaks
    peaks, props = find_peaks(scr, height=max(scr.std() * 0.5, 1e-6),
                               distance=int(FS * 1.0))
    n_scr = len(peaks)
    if n_scr > 0:
        amps = props['peak_heights']
        mag_sum  = float(np.sum(amps))
        dur_sum  = float(n_scr)          # ~1 s per response (rough)
        area     = float(np.trapezoid(np.maximum(scr, 0))) / FS
    else:
        mag_sum = dur_sum = area = 0.0

    return {
        'EDA_mean':              float(eda.mean()),
        'EDA_std':               float(eda.std()),
        'EDA_min':               float(eda.min()),
        'EDA_max':               float(eda.max()),
        'EDA_slope':             slope,
        'EDA_range':             float(eda.max() - eda.min()),
        'EDA_SCL_mean':          float(scl.mean()),
        'EDA_SCL_std':           float(scl.std()),
        'EDA_SCR_std':           float(scr.std()),
        'EDA_SCL_corr':          scl_corr,
        'EDA_SCR_count':         n_scr,
        'EDA_SCR_magnitude_sum': mag_sum,
        'EDA_SCR_duration_sum':  dur_sum,
        'EDA_SCR_area':          area,
    }


def _emg_chain1(emg_5s):
    emg = emg_5s.flatten()
    if len(emg) > 20:
        emg = _hp(emg, 20.0)

    nperseg = min(len(emg), 256)
    freqs, psd = welch(emg, fs=FS, nperseg=nperseg)
    total_p = psd.sum()

    if total_p > 0:
        freq_mean   = float(np.sum(freqs * psd) / total_p)
        cum         = np.cumsum(psd)
        freq_median = float(freqs[np.searchsorted(cum, cum[-1] * 0.5)])
    else:
        freq_mean = freq_median = 0.0

    band_edges = np.linspace(0, 350, 8)
    psd_bands  = {f'EMG_PSD_band{i+1}': _band_power(freqs, psd,
                                                      band_edges[i], band_edges[i+1])
                  for i in range(7)}
    return {
        'EMG_mean':        float(emg.mean()),
        'EMG_std':         float(emg.std()),
        'EMG_range':       float(emg.max() - emg.min()),
        'EMG_integral':    float(np.sum(np.abs(emg))) / FS,
        'EMG_median':      float(np.median(emg)),
        'EMG_p10':         float(np.percentile(emg, 10)),
        'EMG_p90':         float(np.percentile(emg, 90)),
        'EMG_freq_mean':   freq_mean,
        'EMG_freq_median': freq_median,
        'EMG_freq_peak':   float(freqs[np.argmax(psd)]),
        **psd_bands,
    }


def _emg_chain2(emg_60s):
    emg = emg_60s.flatten()
    if len(emg) > 20:
        emg_lp = _lp(emg, 50.0)
    else:
        emg_lp = emg

    rect    = np.abs(emg_lp)
    thresh  = float(np.percentile(rect, 75))   # Wijsman et al.: upper-quartile threshold
    peaks, _ = find_peaks(rect, height=thresh, distance=int(FS * 0.1))
    n = len(peaks)

    if n > 0:
        amps     = rect[peaks]
        duration = len(emg) / FS
        return {
            'EMG_peaks_count':      n,
            'EMG_peak_amp_mean':    float(amps.mean()),
            'EMG_peak_amp_std':     float(amps.std()),
            'EMG_peak_amp_sum':     float(amps.sum()),
            'EMG_peak_amp_norm_sum': float(amps.sum()) / duration,
        }
    return {
        'EMG_peaks_count': 0,
        'EMG_peak_amp_mean': 0.0, 'EMG_peak_amp_std': 0.0,
        'EMG_peak_amp_sum': 0.0,  'EMG_peak_amp_norm_sum': 0.0,
    }


def _resp_features(resp_w, all_peaks, all_troughs, start, end):
    w_peaks   = all_peaks  [(all_peaks   >= start) & (all_peaks   < end)]
    w_troughs = all_troughs[(all_troughs >= start) & (all_troughs < end)]

    inhal, exhal = [], []
    for pk in w_peaks:
        prev = w_troughs[w_troughs < pk]
        nxt  = w_troughs[w_troughs > pk]
        if len(prev): inhal.append((pk - prev[-1]) / FS)
        if len(nxt):  exhal.append((nxt[0]  - pk) / FS)

    resp = resp_w.flatten()

    i_mean = float(np.mean(inhal)) if inhal else 0.0
    i_std  = float(np.std(inhal))  if len(inhal) > 1 else 0.0
    e_mean = float(np.mean(exhal)) if exhal else 0.0
    e_std  = float(np.std(exhal))  if len(exhal) > 1 else 0.0
    ie     = i_mean / e_mean if e_mean > 0 else 0.0
    rate   = float(len(w_peaks))   # peaks per 60-s window = breaths/min

    return {
        'RESP_inhal_mean': i_mean,
        'RESP_inhal_std':  i_std,
        'RESP_exhal_mean': e_mean,
        'RESP_exhal_std':  e_std,
        'RESP_IE_ratio':   ie,
        'RESP_stretch':    float(resp.max() - resp.min()),
        'RESP_vol_insp':   float(np.sum(np.maximum(resp - resp.mean(), 0))) / FS,
        'RESP_rate':       rate,
        'RESP_duration':   float(sum(inhal) + sum(exhal)),
    }


def _temp_features(temp_w):
    temp = temp_w.flatten().astype(float)
    t    = np.arange(len(temp)) / FS
    return {
        'TEMP_mean':  float(temp.mean()),
        'TEMP_std':   float(temp.std()),
        'TEMP_min':   float(temp.min()),
        'TEMP_max':   float(temp.max()),
        'TEMP_range': float(temp.max() - temp.min()),
        'TEMP_slope': float(np.polyfit(t, temp, 1)[0]),
    }


# ── main entry point ────────────────────────────────────────────────────────

def format_and_process_data(data, subject_id):
    """
    Extract sliding-window features from one subject's raw WESAD pkl data.
    Returns a DataFrame (one row per valid window).
    Binary label: 1 = stress (condition 2), 0 = non-stress (conditions 1 & 3).
    """
    chest  = data['signal']['chest']
    labels = data['label'].flatten()

    ecg  = chest['ECG'].flatten().astype(float)
    emg  = chest['EMG'].flatten().astype(float)
    eda  = chest['EDA'].flatten().astype(float)
    temp = chest['Temp'].flatten().astype(float)
    resp = chest['Resp'].flatten().astype(float)

    n = len(ecg)
    subject_start = time.time()
    total_possible = (n - WINDOW_60S) // SHIFT
    print(f'  [{subject_id}] Signal length: {n/FS:.0f} s  |  up to {total_possible:,} candidate windows')

    # ── global pre-processing (done once per subject) ──

    t0 = time.time()
    print(f'  [{subject_id}] (1/4) Filtering and decomposing EDA (Choi et al.) ...')
    eda_f = _lp(eda, 5.0)
    try:
        eda_decomp = nk.eda_phasic(eda_f, sampling_rate=FS, method='highpass')
        eda_scl = eda_decomp['EDA_Tonic'].values
        eda_scr = eda_decomp['EDA_Phasic'].values
    except Exception:
        eda_scl = _lp(eda_f, 0.05)
        eda_scr = eda_f - eda_scl
    print(f'  [{subject_id}]       done ({time.time()-t0:.1f}s)')

    t0 = time.time()
    print(f'  [{subject_id}] (2/4) Detecting ECG R-peaks (Pan-Tompkins) ...')
    try:
        _, ecg_info = nk.ecg_peaks(ecg, sampling_rate=FS, method='pantompkins1985')
        all_r_peaks = ecg_info['ECG_R_Peaks']
        print(f'  [{subject_id}]       {len(all_r_peaks):,} R-peaks found ({time.time()-t0:.1f}s)')
    except Exception as e:
        print(f'  [{subject_id}]       R-peak detection failed: {e}')
        all_r_peaks = np.array([], dtype=int)

    t0 = time.time()
    print(f'  [{subject_id}] (3/4) Detecting respiration peaks ...')
    try:
        resp_f        = _bp(resp, 0.1, 0.35)
        resp_peaks,  _ = find_peaks( resp_f, distance=int(FS * 1.5))
        resp_troughs, _ = find_peaks(-resp_f, distance=int(FS * 1.5))
        print(f'  [{subject_id}]       {len(resp_peaks):,} peaks / {len(resp_troughs):,} troughs ({time.time()-t0:.1f}s)')
    except Exception:
        resp_f, resp_peaks, resp_troughs = resp, np.array([]), np.array([])
        print(f'  [{subject_id}]       RESP processing failed, using raw signal')

    # ── sliding window ──────────────────────────────────────────────────────

    rows      = []
    window_id = 0
    log_every = max(1, total_possible // 10)  # print at every ~10%

    t0 = time.time()
    print(f'  [{subject_id}] (4/4) Sliding window extraction ...')
    for i, start in enumerate(range(0, n - WINDOW_60S, SHIFT)):
        end = start + WINDOW_60S

        if i > 0 and i % log_every == 0:
            pct     = i / total_possible * 100
            elapsed = time.time() - t0
            rate    = i / elapsed if elapsed > 0 else 0
            eta     = (total_possible - i) / rate if rate > 0 else 0
            print(f'  [{subject_id}]       {pct:4.0f}%  |  {len(rows):,} valid windows so far  |  ETA {eta:.0f}s')

        # keep window if >95% of samples share the same valid label
        win_labels = labels[start:end]
        vals, cnts = np.unique(win_labels, return_counts=True)
        majority = int(vals[cnts.argmax()])
        if cnts.max() / len(win_labels) < 0.95 or majority not in VALID_LABELS:
            continue

        binary_label = 1 if majority == 2 else 0

        # ECG
        w_peaks = all_r_peaks[(all_r_peaks >= start) & (all_r_peaks < end)]
        ecg_f   = _ecg_features(w_peaks)
        if ecg_f is None:
            continue

        # EDA
        eda_f_w = _eda_features(eda_f[start:end], eda_scl[start:end], eda_scr[start:end])

        # EMG: chain 1 uses last 5 s of the 60-s window; chain 2 uses full 60 s
        emg_c1 = _emg_chain1(emg[end - WINDOW_5S : end])
        emg_c2 = _emg_chain2(emg[start:end])

        # RESP
        resp_f_w = _resp_features(resp_f[start:end], resp_peaks, resp_troughs, start, end)

        # TEMP
        temp_f = _temp_features(temp[start:end])

        rows.append({
            'Participant ID': subject_id,
            'Window ID':      window_id,
            **ecg_f,
            **eda_f_w,
            **emg_c1,
            **emg_c2,
            **resp_f_w,
            **temp_f,
            'Label': binary_label,
        })
        window_id += 1

    elapsed_total = time.time() - subject_start
    print(f'  [{subject_id}] Done — {len(rows):,} windows extracted in {elapsed_total:.1f}s')
    return pd.DataFrame(rows)
