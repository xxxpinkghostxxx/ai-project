"""Audio output module — oscillator bank synthesizer.

Reads workspace energy from the audio workspace grid regions and synthesizes
stereo audio via a bank of sine oscillators.  Each frequency bin maps to one
oscillator; energy controls amplitude.  Phase accumulators ensure glitch-free
output, and EMA smoothing prevents clicks on rapid amplitude changes.
"""

import logging
import math
import threading
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    sd = None  # type: ignore[assignment]
    SD_AVAILABLE = False
    logger.warning("sounddevice not available — audio output disabled")


class AudioOutput:
    """Oscillator-bank synthesizer driven by workspace energy.

    Parameters
    ----------
    n_bins : int
        Number of frequency bins per channel (default 256).
    sample_rate : int
        Output sample rate in Hz (default 44100).
    buffer_size : int
        sounddevice block size (default 1024).
    min_freq : float
        Lowest oscillator frequency in Hz (default 80).
    max_freq : float
        Highest oscillator frequency in Hz (default 8000).
    master_volume : float
        Global volume multiplier in [0, 1] (default 0.3).
    """

    def __init__(
        self,
        n_bins: int = 256,
        sample_rate: int = 44100,
        buffer_size: int = 1024,
        min_freq: float = 80.0,
        max_freq: float = 8000.0,
        master_volume: float = 0.3,
    ) -> None:
        self.n_bins = n_bins
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.master_volume = np.clip(master_volume, 0.0, 1.0)

        # Log-spaced frequencies for each oscillator
        self.frequencies: NDArray[np.floating[Any]] = np.logspace(
            math.log10(min_freq), math.log10(max_freq), n_bins, dtype=np.float64,
        )

        # Phase accumulators (one per bin, in radians)
        self._phase_L: NDArray[np.floating[Any]] = np.zeros(n_bins, dtype=np.float64)
        self._phase_R: NDArray[np.floating[Any]] = np.zeros(n_bins, dtype=np.float64)

        # Current and target amplitudes (EMA-smoothed)
        self._amp_L: NDArray[np.floating[Any]] = np.zeros(n_bins, dtype=np.float32)
        self._amp_R: NDArray[np.floating[Any]] = np.zeros(n_bins, dtype=np.float32)
        self._target_amp_L: NDArray[np.floating[Any]] = np.zeros(n_bins, dtype=np.float32)
        self._target_amp_R: NDArray[np.floating[Any]] = np.zeros(n_bins, dtype=np.float32)

        # EMA smoothing factor (higher = faster response, more clicks)
        self._smoothing = 0.15

        # Pre-compute phase increments per sample for each frequency
        self._phase_inc: NDArray[np.floating[Any]] = (
            2.0 * np.pi * self.frequencies / self.sample_rate
        )

        # Stream management
        self._stream: Optional[Any] = None
        self._running = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start audio output stream."""
        if not SD_AVAILABLE:
            logger.error("sounddevice not installed — cannot start audio output")
            return
        if self._running:
            return

        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.buffer_size,
                channels=2,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            self._running = True
            logger.info(
                "Audio output started (sr=%d, bins=%d, vol=%.2f)",
                self.sample_rate, self.n_bins, self.master_volume,
            )
        except Exception as e:
            logger.error("Failed to start audio output: %s", e)

    def stop(self) -> None:
        """Stop audio output stream."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.debug("Error closing audio output stream: %s", e)
            self._stream = None
        logger.info("Audio output stopped")

    def update_amplitudes(
        self,
        energy_L: NDArray[np.floating[Any]],
        energy_R: NDArray[np.floating[Any]],
    ) -> None:
        """Update target oscillator amplitudes from workspace energy grids.

        Parameters
        ----------
        energy_L, energy_R : NDArray
            2-D arrays of shape ``(rows, n_bins)`` from the audio workspace
            region.  Columns are averaged down to ``n_bins`` amplitude values
            and normalised to [0, 1].
        """
        try:
            # Average rows → (n_bins,)
            amp_L = np.mean(energy_L, axis=0).astype(np.float32)
            amp_R = np.mean(energy_R, axis=0).astype(np.float32)

            # Clamp to expected bin count
            amp_L = amp_L[: self.n_bins]
            amp_R = amp_R[: self.n_bins]

            # Pad if shorter
            if amp_L.shape[0] < self.n_bins:
                amp_L = np.pad(amp_L, (0, self.n_bins - amp_L.shape[0]))
            if amp_R.shape[0] < self.n_bins:
                amp_R = np.pad(amp_R, (0, self.n_bins - amp_R.shape[0]))

            # Normalise to [0, 1]
            max_val = max(amp_L.max(), amp_R.max(), 1e-6)
            amp_L /= max_val
            amp_R /= max_val

            with self._lock:
                self._target_amp_L = amp_L
                self._target_amp_R = amp_R

        except Exception as e:
            logger.debug("update_amplitudes error: %s", e)

    def set_master_volume(self, volume: float) -> None:
        """Set master volume (0.0 to 1.0)."""
        self.master_volume = float(np.clip(volume, 0.0, 1.0))

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        outdata: NDArray[np.floating[Any]],
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """sounddevice OutputStream callback — vectorized additive synthesis."""
        if status:
            logger.debug("Audio output status: %s", status)

        try:
            # EMA smooth amplitudes toward targets
            with self._lock:
                target_L = self._target_amp_L.copy()
                target_R = self._target_amp_R.copy()

            alpha = self._smoothing
            self._amp_L += alpha * (target_L - self._amp_L)
            self._amp_R += alpha * (target_R - self._amp_R)

            # Build sample indices: (frames,)
            t = np.arange(frames, dtype=np.float64)

            # Phase matrix: (frames, n_bins) = outer(t, phase_inc) + phase
            phases_L = np.outer(t, self._phase_inc) + self._phase_L[np.newaxis, :]
            phases_R = np.outer(t, self._phase_inc) + self._phase_R[np.newaxis, :]

            # Sine synthesis: (frames, n_bins)
            sines_L = np.sin(phases_L)  # (frames, n_bins)
            sines_R = np.sin(phases_R)

            # Weight by amplitudes and sum across bins → (frames,)
            left = (sines_L * self._amp_L[np.newaxis, :]).sum(axis=1)
            right = (sines_R * self._amp_R[np.newaxis, :]).sum(axis=1)

            # Advance phase accumulators (mod 2π to prevent float drift)
            self._phase_L = (self._phase_L + self._phase_inc * frames) % (2.0 * np.pi)
            self._phase_R = (self._phase_R + self._phase_inc * frames) % (2.0 * np.pi)

            # Normalise by peak amplitude to prevent clipping while preserving
            # relative spectral balance (bin-count division was non-linear).
            peak_L = max(np.abs(left).max(), 1e-8)
            peak_R = max(np.abs(right).max(), 1e-8)
            left /= peak_L
            right /= peak_R

            # Apply master volume and write to output buffer
            vol = self.master_volume
            outdata[:, 0] = (left * vol).astype(np.float32)
            outdata[:, 1] = (right * vol).astype(np.float32)

        except Exception as e:
            logger.debug("Audio output callback error: %s", e)
            outdata[:] = 0.0
