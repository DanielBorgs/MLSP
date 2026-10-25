"""
02471 Machine Learning for Signal Processing -- Problem set 2, problems 2.5-2.7.

Usage:   python solutions.py [data_dir] [out_dir]
         data_dir (default ./data) must contain
             problem2_5_signal.mat, problem2_5_noise.mat,
             problem2_6.wav,
             problem2_7_speech.mat, problem2_7_lpir.mat,
             problem2_7_hpir.mat,  problem2_7_bpir.mat
Output:  out_dir/figs/*.pdf   figures used by exercise_solutions.tex
         out_dir/results.tex  numbers used by exercise_solutions.tex
         out_dir/audio/*.wav  noisy / cleaned speech for the listening test
"""
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.signal as sg
from scipy.cluster.vq import kmeans2
from scipy.io import wavfile
from scipy.linalg import solve_toeplitz

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "data"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE
FIGS, AUDIO = OUT / "figs", OUT / "audio"
FIGS.mkdir(parents=True, exist_ok=True)
AUDIO.mkdir(parents=True, exist_ok=True)
FS = 8000
RNG = np.random.default_rng(0)
R = []  # lines of results.tex


def macro(name, value):
    R.append(f"\\newcommand{{\\{name}}}{{{value}}}")


def load_mat(name):
    """Load the (single) numeric variable of a .mat file as a 1-D float array."""
    m = sio.loadmat(DATA / name)
    arrs = [np.asarray(v) for k, v in m.items()
            if not k.startswith("__") and np.issubdtype(np.asarray(v).dtype, np.number)]
    return max(arrs, key=lambda a: a.size).astype(float).ravel()


def db(x):
    return 10 * np.log10(x)


# ----------------------------------------------------------------------------
# Problem 2.5  FIR Wiener filter
# ----------------------------------------------------------------------------
def xcorr(a, b, L):
    """Biased estimate of E[a_n b_{n-k}], k = 0..L-1."""
    N = len(a)
    return np.array([a[k:] @ b[:N - k] for k in range(L)]) / N


def wiener_fir(x, d, L):
    """Solve Sigma_x theta = p, Sigma_x = Toeplitz(r_x), p_k = E[d_n x_{n-k}]."""
    return solve_toeplitz(xcorr(x, x, L), xcorr(d, x, L))


def problem_2_5():
    s, w = load_mat("problem2_5_signal.mat"), load_mat("problem2_5_noise.mat")
    n = min(len(s), len(w))
    s, w = s[:n], w[:n]
    x = s + w
    half = n // 2                       # estimate on first half, test on second
    Ls = np.unique(np.round(np.geomspace(1, 512, 30)).astype(int))
    mse_tr, mse_te = [], []
    for L in Ls:
        th = wiener_fir(x[:half], s[:half], L)
        sh = sg.lfilter(th, 1, x)
        mse_tr.append(np.mean((s[:half] - sh[:half]) ** 2))
        mse_te.append(np.mean((s[half:] - sh[half:]) ** 2))
    mse_tr, mse_te = np.array(mse_tr), np.array(mse_te)
    # smallest L whose held-out MSE is within 2 % of the best held-out MSE
    L_opt = int(Ls[np.argmax(mse_te <= 1.02 * mse_te.min())])

    th = wiener_fir(x, s, L_opt)        # final filter on all data
    sh = sg.lfilter(th, 1, x)
    snr_in = db(np.sum(s ** 2) / np.sum(w ** 2))
    snr_out = db(np.sum(s ** 2) / np.sum((s - sh) ** 2))

    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.semilogx(Ls, db(mse_tr), "o-", ms=3, label="estimation half (train)")
    ax.semilogx(Ls, db(mse_te), "s-", ms=3, label="held-out half (test)")
    ax.axvline(L_opt, color="k", ls="--", lw=.8, label=f"chosen $L={L_opt}$")
    ax.set_xlabel("filter length $L$"); ax.set_ylabel("MSE [dB]"); ax.legend()
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(FIGS / "p25_mse_vs_L.pdf")

    nper = 512
    f, Ps = sg.welch(s, FS, nperseg=nper); _, Pw = sg.welch(w, FS, nperseg=nper)
    _, Px = sg.welch(x, FS, nperseg=nper); _, Pe = sg.welch(sh, FS, nperseg=nper)
    fh, H = sg.freqz(th, 1, worN=512, fs=FS)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for P, lab in [(Ps, "$s_n$ (signal)"), (Pw, "$w_n$ (noise)"),
                   (Px, "$x_n=s_n+w_n$"), (Pe, r"$\hat s_n$ (filter output)")]:
        ax[0].plot(f, db(P), label=lab)
    ax[0].set_xlabel("frequency [Hz]"); ax[0].set_ylabel("PSD [dB/Hz]")
    ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax[0].grid(alpha=.3)
    ax[1].plot(fh, db(np.abs(H) ** 2 + 1e-12), label=f"FIR Wiener, $L={L_opt}$")
    ax[1].plot(f, db(Ps / (Ps + Pw)), "--", label=r"ideal non-causal $S_s/(S_s+S_w)$")
    ax[1].set_ylim(-60, 5)
    ax[1].set_xlabel("frequency [Hz]"); ax[1].set_ylabel(r"$|H(f)|^2$ [dB]")
    ax[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
    ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "p25_freq.pdf", bbox_inches="tight")

    macro("resWienerL", L_opt)
    macro("resWienerSNRin", f"{snr_in:.1f}")
    macro("resWienerSNRout", f"{snr_out:.1f}")
    macro("resWienerGain", f"{snr_out - snr_in:.1f}")
    macro("resWienerLtestMSE", f"{db(mse_te[Ls == L_opt][0]):.1f}")
    macro("resWienerMinMSE", f"{db(mse_te.min()):.1f}")
    macro("resWienerLmin", int(Ls[np.argmin(mse_te)]))


# ----------------------------------------------------------------------------
# Problem 2.6  Spectrogram of audio
# ----------------------------------------------------------------------------
def problem_2_6():
    fs, y = wavfile.read(DATA / "problem2_6.wav")
    y = y.astype(float)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y /= np.max(np.abs(y))
    nper, hop = 1024, 128
    f, t, S = sg.spectrogram(y, fs, window="hann", nperseg=nper,
                             noverlap=nper - hop, mode="magnitude")
    SdB = 20 * np.log10(S + 1e-10)

    # onset detection: spectral flux of log-compressed magnitude
    C = np.log1p(100 * S / S.max())
    flux = np.maximum(np.diff(C, axis=1), 0).sum(axis=0)
    flux = np.convolve(flux, np.hanning(5) / np.hanning(5).sum(), "same")
    min_dist = int(0.06 * fs / hop)
    pk, _ = sg.find_peaks(flux, height=flux.mean() + 1.0 * flux.std(), distance=min_dist)
    onsets = t[pk + 1]
    ioi = np.diff(onsets)

    # per-onset spectral signature: mean log-spectrum in 12 log-spaced bands
    edges = np.geomspace(40, fs / 2, 13)
    feats, spec = [], []
    for i in pk + 1:
        seg = S[:, i:i + 8].mean(axis=1)
        spec.append(seg)
        b = [seg[(f >= lo) & (f < hi)].sum() ** 2 for lo, hi in zip(edges[:-1], edges[1:])]
        b = db(np.array(b) + 1e-12)
        feats.append(b - b.max())
    feats, spec = np.array(feats), np.array(spec)
    K = 3
    _, lab = kmeans2(feats, K, minit="++", seed=0)
    cent = np.array([np.sum(f * spec[lab == k].mean(0)) / np.sum(spec[lab == k].mean(0))
                     if np.any(lab == k) else np.nan for k in range(K)])
    order = np.argsort(cent)                     # low -> high
    rank = {int(k): r for r, k in enumerate(order)}
    lab = np.array([rank[int(l)] for l in lab])
    names = ["low", "mid", "high"]
    cols = ["tab:red", "tab:green", "tab:cyan"]

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1]})
    im = ax[0].pcolormesh(t, f / 1e3, SdB, shading="auto", cmap="magma",
                          vmin=SdB.max() - 80, vmax=SdB.max())
    for r in range(K):
        ax[0].plot(onsets[lab == r], np.full((lab == r).sum(), f[-1] / 1e3 * .97), "v",
                   color=cols[r], ms=5, label=f"{names[r]} instrument")
    ax[0].set_ylabel("frequency [kHz]"); ax[0].legend(loc="lower left", bbox_to_anchor=(0, 1.01), fontsize=8, ncol=3, frameon=False)
    fig.colorbar(im, ax=ax, label="dB", pad=0.01)
    ax[1].plot(t[1:], flux, lw=.7); ax[1].plot(onsets, flux[pk], "kx")
    ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("spectral flux")
    fig.savefig(FIGS / "p26_spectrogram.pdf", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    for r in range(K):
        if np.any(lab == r):
            ax.semilogx(f[1:], db(spec[lab == r].mean(0)[1:] ** 2 + 1e-12),
                        color=cols[r], label=f"{names[r]} (n={np.sum(lab == r)})")
    ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("mean spectrum [dB]")
    ax.grid(alpha=.3, which="both"); ax.legend(); fig.tight_layout()
    fig.savefig(FIGS / "p26_signatures.pdf")

    rows = []
    for r in range(K):
        m = lab == r
        if not m.any():
            continue
        ms = spec[m].mean(0)
        rows.append(f"{names[r]} & {m.sum()} & {np.sum(f * ms) / np.sum(ms):.0f} & "
                    f"{f[1:][np.argmax(ms[1:])]:.0f} \\\\")
    med = np.median(ioi)
    macro("resSixFs", fs)
    macro("resSixDur", f"{len(y) / fs:.2f}")
    macro("resSixDf", f"{fs / nper:.1f}")
    macro("resSixWinMs", f"{1e3 * nper / fs:.0f}")
    macro("resSixHopMs", f"{1e3 * hop / fs:.1f}")
    macro("resSixNons", len(onsets))
    macro("resSixIOI", f"{med:.3f}")
    macro("resSixBPM", f"{60 / med:.0f}")
    macro("resSixRows", " ".join(rows))


# ----------------------------------------------------------------------------
# Problem 2.7  Adaptive noise cancelling: LMS, NLMS, RLS
# ----------------------------------------------------------------------------
def regressors(d, M):
    """Row n = [d_n, d_{n-1}, ..., d_{n-M+1}] (zero padded)."""
    dp = np.concatenate([np.zeros(M - 1), d])
    return np.lib.stride_tricks.sliding_window_view(dp, M)[:, ::-1]


def lms(d, p, M, mu):
    U, w, e = regressors(d, M), np.zeros(M), np.zeros(len(d))
    for n in range(len(d)):
        e[n] = p[n] - w @ U[n]
        w += mu * e[n] * U[n]
    return e, w


def nlms(d, p, M, mu, eps=1e-6):
    U, w, e = regressors(d, M), np.zeros(M), np.zeros(len(d))
    for n in range(len(d)):
        e[n] = p[n] - w @ U[n]
        w += mu * e[n] * U[n] / (eps + U[n] @ U[n])
    return e, w


def rls(d, p, M, lam, delta=0.01):
    U, w, e = regressors(d, M), np.zeros(M), np.zeros(len(d))
    P = np.eye(M) / (delta * np.var(d))
    for n in range(len(d)):
        u = U[n]
        Pu = P @ u
        k = Pu / (lam + u @ Pu)
        e[n] = p[n] - w @ u
        w += k * e[n]
        P = (P - np.outer(k, Pu)) / lam
    return e, w


def snr(s, resid):
    return db(np.sum(s ** 2) / np.sum(resid ** 2))


def smooth_db(x, N=1000):
    return db(np.convolve(x ** 2, np.ones(N) / N, "valid") + 1e-14)


def problem_2_7():
    s = load_mat("problem2_7_speech.mat")
    s = s / np.max(np.abs(s)) * 0.5
    N = len(s)
    irs = {k: load_mat(f"problem2_7_{k}ir.mat") for k in ("lp", "hp", "bp")}
    names = {"lp": "low-pass", "hp": "high-pass", "bp": "band-pass"}
    SNR_IN = 0.0                                      # dB at the primary microphone
    v = RNG.standard_normal(N)                        # reference noise (unit variance)
    results, curves = [], {}
    rows = []
    sweep_rows = {}

    for key, h in irs.items():
        noise = sg.lfilter(h, 1, v)
        g = np.sqrt(np.sum(s ** 2) / np.sum(noise ** 2) * 10 ** (-SNR_IN / 10))
        d = g * v                                     # reference microphone
        nz = g * noise                                # noise at primary microphone
        p = s + nz                                    # primary microphone
        M = len(h)
        wtrue = h                                     # noise at primary = h * d, so the ideal w is h
        s2d = np.var(d)
        wavfile.write(AUDIO / f"noisy_{key}.wav", FS, (p / np.abs(p).max() * .9 * 32767).astype(np.int16))

        # step-size / forgetting-factor sweeps (final SNR over the last half)
        def final_snr(e):
            return snr(s[N // 2:], (e - s)[N // 2:])
        mu_l = np.array([.01, .03, .1, .3, .6]) * 2 / (M * s2d) * 0.5
        mu_n = np.array([.01, .02, .05, .1, .25, .5, 1.0])
        lam_r = np.array([.99, .995, .999, .9995, 1.0])
        sw_l = [final_snr(lms(d, p, M, m)[0]) for m in mu_l]
        sw_n = [final_snr(nlms(d, p, M, m)[0]) for m in mu_n]
        sw_r = [final_snr(rls(d, p, M, l)[0]) for l in lam_r]
        best = dict(LMS=(mu_l[int(np.argmax(sw_l))], lms),
                    NLMS=(mu_n[int(np.argmax(sw_n))], nlms),
                    RLS=(lam_r[int(np.argmax(sw_r))], rls))
        sweep_rows[key] = (mu_l, sw_l, mu_n, sw_n, lam_r, sw_r)

        curves[key] = {}
        for alg, (par, fn) in best.items():
            e, w = fn(d, p, M, par)
            curves[key][alg] = (e, w)
            mis = db(np.sum((w - wtrue) ** 2) / np.sum(wtrue ** 2))
            rows.append(f"{names[key]} & {alg} & {par:.3g} & {snr(s, p - s):.1f} & "
                        f"{snr(s[N // 2:], (e - s)[N // 2:]):.1f} & {mis:.1f} \\\\")
            wavfile.write(AUDIO / f"clean_{key}_{alg}.wav", FS,
                          (e / np.abs(p).max() * .9 * 32767).astype(np.int16))
        curves[key]["_noise"] = nz
        curves[key]["_p"] = p
        curves[key]["_d"] = d
        curves[key]["_h"] = wtrue

    # --- figures -----------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for a, key in zip(ax, irs):
        nz = curves[key]["_noise"]
        a.plot(smooth_db(nz), "k", lw=.8, label="noise at primary mic")
        for alg in ("LMS", "NLMS", "RLS"):
            e = curves[key][alg][0]
            a.plot(smooth_db(e - s), label=f"{alg}: residual noise")
        a.set_title(names[key]); a.set_xlabel("sample $n$"); a.grid(alpha=.3)
    ax[0].set_ylabel("smoothed power [dB]"); ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "p27_learning.pdf")

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for a, key in zip(ax, irs):
        M = len(irs[key])
        fh, Ht = sg.freqz(curves[key]["_h"], 1, worN=512, fs=FS)
        a.plot(fh, db(np.abs(Ht) ** 2 + 1e-12), "k", lw=1.5, label="true $h$")
        for alg in ("LMS", "NLMS", "RLS"):
            fw, Hw = sg.freqz(curves[key][alg][1], 1, worN=512, fs=FS)
            a.plot(fw, db(np.abs(Hw) ** 2 + 1e-12), "--", label=alg)
        a.set_title(names[key]); a.set_xlabel("frequency [Hz]"); a.grid(alpha=.3)
    ax[0].set_ylabel(r"$|H(f)|^2$ [dB]"); ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "p27_filters.pdf")

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    key0 = list(irs)[0]
    mu_l, sw_l, mu_n, sw_n, lam_r, sw_r = sweep_rows[key0]
    ax[0].semilogx(mu_l, sw_l, "o-"); ax[0].set_xlabel(r"LMS $\mu$")
    ax[1].semilogx(mu_n, sw_n, "o-"); ax[1].set_xlabel(r"NLMS $\tilde\mu$")
    ax[2].plot(lam_r, sw_r, "o-"); ax[2].set_xlabel(r"RLS $\lambda$")
    ax[0].set_ylabel("output SNR [dB]")
    for a in ax:
        a.grid(alpha=.3)
    fig.suptitle(f"Parameter sweep, {names[key0]} noise path", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "p27_sweep.pdf")

    # --- experiment: reference microphone picks up the speech ---------------
    key = "bp"
    d, p, h = curves[key]["_d"], curves[key]["_p"], curves[key]["_h"]
    M = len(irs[key])
    leak = []
    for alpha in (0.0, 0.1, 0.3):
        dd = d + alpha * s * np.std(d) / np.std(s)
        e, _ = nlms(dd, p, M, 0.5)
        leak.append((alpha, snr(s[N // 2:], (e - s)[N // 2:])))
    macro("resLeakRows", " ".join(f"{a:.1f} & {x:.1f} \\\\" for a, x in leak))

    macro("resSevenN", N)
    macro("resSevenM", ", ".join(f"{names[k]}: {len(h)}" for k, h in irs.items()))
    macro("resSevenSNRin", f"{SNR_IN:.0f}")
    macro("resSevenRows", " ".join(rows))


if __name__ == "__main__":
    for fn in (problem_2_5, problem_2_6, problem_2_7):
        fn()
        print("done", fn.__name__)
    (OUT / "results.tex").write_text("\n".join(R) + "\n")