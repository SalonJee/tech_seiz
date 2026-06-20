import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import welch


class EDFReader:
    """Minimal EDF file reader for basic channel extraction."""

    @staticmethod
    def _read_ascii_block(file, length):
        return file.read(length).decode("ascii", "ignore")

    @staticmethod
    def _strip_text(text):
        if isinstance(text, bytes):
            text = text.decode("ascii", "ignore")
        return text.rstrip(" \t\r\n\x00")

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            header = f.read(256)
            if len(header) < 256:
                raise ValueError("EDF header too short")

            num_records = int(cls._strip_text(header[236:244]))
            duration = float(cls._strip_text(header[244:252]))
            num_signals = int(cls._strip_text(header[252:256]))

            signal_labels = [cls._strip_text(f.read(16)) for _ in range(num_signals)]
            transducer_types = [cls._strip_text(f.read(80)) for _ in range(num_signals)]
            physical_dims = [cls._strip_text(f.read(8)) for _ in range(num_signals)]
            physical_mins = [float(cls._strip_text(f.read(8))) for _ in range(num_signals)]
            physical_maxs = [float(cls._strip_text(f.read(8))) for _ in range(num_signals)]
            digital_mins = [int(cls._strip_text(f.read(8))) for _ in range(num_signals)]
            digital_maxs = [int(cls._strip_text(f.read(8))) for _ in range(num_signals)]
            prefilterings = [cls._strip_text(f.read(80)) for _ in range(num_signals)]
            samples_per_record = [int(cls._strip_text(f.read(8))) for _ in range(num_signals)]
            reserved = [cls._strip_text(f.read(32)) for _ in range(num_signals)]

            total_samples_per_record = sum(samples_per_record)
            data_dtype = np.dtype('<i2')
            total_records = num_records * total_samples_per_record
            raw_data = np.fromfile(f, dtype=data_dtype, count=total_records)

            if raw_data.size != total_records:
                raise ValueError("EDF file does not contain the expected number of samples")

            raw_data = raw_data.reshape((num_records, total_samples_per_record))

            channels = []
            offset = 0
            for chan_idx in range(num_signals):
                count = samples_per_record[chan_idx]
                channel_data = raw_data[:, offset:offset + count].reshape(-1)
                offset += count
                digital_min = digital_mins[chan_idx]
                digital_max = digital_maxs[chan_idx]
                physical_min = physical_mins[chan_idx]
                physical_max = physical_maxs[chan_idx]
                scale = (physical_max - physical_min) / (digital_max - digital_min)
                channel_phys = physical_min + (channel_data - digital_min) * scale
                channels.append(channel_phys)

            sampling_rate = samples_per_record[0] / duration
            return {
                "labels": signal_labels,
                "data": np.vstack(channels),
                "fs": sampling_rate,
                "record_duration": duration,
                "num_records": num_records,
            }

    @classmethod
    def load_channel(cls, path, channel_index=0, max_seconds=None):
        data = cls.load(path)
        if channel_index < 0 or channel_index >= data["data"].shape[0]:
            raise IndexError("channel_index out of range")
        channel = data["data"][channel_index]
        if max_seconds is not None:
            max_samples = int(round(max_seconds * data["fs"]))
            channel = channel[:max_samples]
        return {
            "name": data["labels"][channel_index],
            "signal": channel,
            "fs": data["fs"],
        }


# Thalamocortical model constants
TAU = 0.01
W_LOOP = np.array([
    [0, -25, 35, 0],
    [25, 0, 0, 0],
    [25, 0, 0, -45],
    [25, 0, 25, 0]
])


def sigmoid(V, gain=2.5):
    return 1.0 / (1.0 + np.exp(-gain * (V - 2.0)))


def suffczynski_ode(t, y, gain=1.0, noise_amp=1.5, bias=2.0):
    u_noise = bias + noise_amp * np.random.normal()
    F = sigmoid(y, gain=gain)
    dVdt = (-y + np.dot(W_LOOP, F)) / TAU
    dVdt[2] += u_noise
    return dVdt


def haghighi_ode(t, y, freq=10.0, gain=2.5, sine_amp=12.0, bias=2.0):
    u_sine = bias + sine_amp * np.cos(2 * np.pi * freq * t)
    F = sigmoid(y, gain=gain)
    dVdt = (-y + np.dot(W_LOOP, F)) / TAU
    dVdt[2] += u_sine
    return dVdt


def run_model(model, duration=10.0, steps=5000, **kwargs):
    t_span = (0.0, duration)
    t_eval = np.linspace(0.0, duration, steps)
    y0 = np.zeros(4)
    if model == "suffczynski":
        sol = solve_ivp(
            lambda t, y: suffczynski_ode(t, y, **kwargs),
            t_span,
            y0,
            t_eval=t_eval,
            method="RK45",
        )
    elif model == "haghighi":
        sol = solve_ivp(
            lambda t, y: haghighi_ode(t, y, **kwargs),
            t_span,
            y0,
            t_eval=t_eval,
            method="RK45",
        )
    else:
        raise ValueError("Unknown model: %s" % model)
    return sol.t, sol.y[0]


def plot_time_series(ax, time, signal, label, color):
    ax.plot(time, signal, label=label, color=color, linewidth=1)
    ax.set_ylabel(label)
    ax.grid(True)


def plot_psd(ax, signal, fs, label, color, nperseg=None):
    nperseg = nperseg or min(2048, len(signal))
    f, Pxx = welch(signal, fs=fs, nperseg=nperseg)
    ax.semilogy(f, Pxx, label=label, color=color)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)


def trim_signal(signal, fs, max_seconds):
    max_samples = int(round(max_seconds * fs))
    return signal[:max_samples]


def main():
    data_folder = os.path.join("eeg data", "chb01_eeg")
    normal_path = os.path.join(data_folder, "chb01_01.edf")
    seizure_path = os.path.join(data_folder, "chb01_03.edf")

    if not os.path.exists(normal_path) or not os.path.exists(seizure_path):
        raise FileNotFoundError(
            "EEG data files not found. Place chb01_01.edf and chb01_03.edf in eeg data/chb01_eeg/."
        )

    # chb01_01 and chb01_03 are file names for recordings, not EEG channels.
    # Each EDF file contains many channels (electrode pairs), and the script
    # currently loads the first channel in the file.
    channel_index = 0
    max_seconds = 60.0
    normal = EDFReader.load_channel(normal_path, channel_index=channel_index, max_seconds=max_seconds)
    seizure = EDFReader.load_channel(seizure_path, channel_index=channel_index, max_seconds=max_seconds)

    fs = normal["fs"]
    t_eeg = np.arange(len(normal["signal"])) / fs

    model_duration = 10.0
    steps = int(model_duration * 500)
    t_suff, v_suff = run_model(
        "suffczynski",
        duration=model_duration,
        steps=steps,
        gain=1.0,
        noise_amp=2.0,
        bias=2.0,
    )
    t_hagh_quiet, v_hagh_quiet = run_model(
        "haghighi",
        duration=model_duration,
        steps=steps,
        freq=2.0,
        gain=2.5,
        sine_amp=8.0,
        bias=2.0,
    )
    t_hagh_seiz, v_hagh_seiz = run_model(
        "haghighi",
        duration=model_duration,
        steps=steps,
        freq=10.0,
        gain=2.5,
        sine_amp=12.0,
        bias=2.0,
    )

    os.makedirs("graphs", exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    fig.suptitle("EEG Channel 1 Comparison: Normal vs Seizure", fontsize=16, y=0.98)
    axes[0].plot(t_eeg, normal["signal"], label="Normal EEG (chb01_01) — actual data", color="tab:blue")
    axes[0].plot(t_eeg, seizure["signal"], label="Seizure EEG (chb01_03) — actual data", color="tab:red", alpha=0.75)
    axes[0].set_title(f"Time series of EEG Channel 1 ({normal['name']})")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Voltage")
    axes[0].legend(loc="upper right")
    axes[0].grid(True)

    plot_psd(axes[1], normal["signal"], fs, "Normal EEG (chb01_01) — actual data", "tab:blue")
    plot_psd(axes[1], seizure["signal"], fs, "Seizure EEG (chb01_03) — actual data", "tab:red")
    axes[1].set_title("Power Spectral Density of EEG Channel 1")
    axes[1].legend()
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("graphs/eeg_channel1_normal_vs_seizure.png", dpi=150)

    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig2.suptitle("Thalamocortical Model Time Traces (Suffczynski and Haghighi)", fontsize=16, y=0.98)
    plot_time_series(axes2[0], t_suff, v_suff, "Suffczynski model: stochastic normal", "tab:blue")
    axes2[0].set_title("Suffczynski model output (stochastic input) — assumed normal-like state")
    axes2[0].legend()
    plot_time_series(axes2[1], t_hagh_quiet, v_hagh_quiet, "Haghighi model: quiet 2 Hz", "tab:green")
    axes2[1].set_title("Haghighi model output (2 Hz sinusoidal input) — quiet/interictal-like state")
    axes2[1].legend()
    plot_time_series(axes2[2], t_hagh_seiz, v_hagh_seiz, "Haghighi model: seizure 10 Hz", "tab:red")
    axes2[2].set_title("Haghighi model output (10 Hz sinusoidal input) — seizure/ictal-like state")
    axes2[2].legend()
    axes2[2].set_xlabel("Time (s)")
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig("graphs/model_traces.png", dpi=150)

    fig3, axes3 = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
    fig3.suptitle("Power Spectral Density Comparison: EEG and Model", fontsize=16, y=0.98)
    plot_psd(axes3[0], normal["signal"], fs, "Normal EEG (chb01_01) — actual data", "tab:blue")
    axes3[0].set_title("EEG Channel 1 PSD — normal data")
    axes3[0].legend()
    plot_psd(axes3[1], seizure["signal"], fs, "Seizure EEG (chb01_03) — actual data", "tab:red")
    axes3[1].set_title("EEG Channel 1 PSD — seizure data")
    axes3[1].legend()
    plot_psd(axes3[2], v_hagh_seiz, 1.0 / (t_hagh_seiz[1] - t_hagh_seiz[0]), "Haghighi model PSD — seizure state", "tab:purple")
    axes3[2].set_title("Haghighi model PSD — seizure/ictal-like state")
    axes3[2].legend()
    fig3.tight_layout(rect=[0, 0, 1, 0.96])
    fig3.savefig("graphs/model_and_eeg_psd.png", dpi=150)

    def trapezoidal_integral(y, x):
        if len(x) < 2 or len(y) < 2:
            return 0.0
        dx = np.diff(x)
        return np.sum((y[:-1] + y[1:]) * dx * 0.5)

    def power_band(signal, fs, low, high):
        f, Pxx = welch(signal, fs=fs, nperseg=min(2048, len(signal)))
        idx = np.logical_and(f >= low, f <= high)
        return trapezoidal_integral(Pxx[idx], f[idx])

    normal_band = power_band(normal["signal"], fs, 3, 12)
    seizure_band = power_band(seizure["signal"], fs, 3, 12)
    print("=== Validation summary ===")
    print(f"Recording file normal: {normal_path}")
    print(f"Recording file seizure: {seizure_path}")
    print(f"Using EDF channel index: {channel_index} (label: {normal['name']})")
    print(f"Sampling rate: {fs:.2f} Hz")
    print(f"Normal 3-12 Hz power: {normal_band:.3e}")
    print(f"Seizure 3-12 Hz power: {seizure_band:.3e}")
    print(f"Seizure / Normal power ratio: {seizure_band / (normal_band + 1e-12):.2f}")
    print("Saved graphs:")
    print(" - graphs/eeg_channel1_normal_vs_seizure.png")
    print(" - graphs/model_traces.png")
    print(" - graphs/model_and_eeg_psd.png")

    plt.show()


if __name__ == "__main__":
    main()
