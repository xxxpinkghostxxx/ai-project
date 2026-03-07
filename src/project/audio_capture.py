"""Audio capture module for FFT-based spectral input.

Provides threaded audio capture from system loopback (WASAPI) or microphone,
computing stereo FFT magnitude spectra for injection into the neural grid.
Mirrors the AsyncScreenCapture / ThreadedScreenCapture pattern.
"""

import threading
import time
import logging
from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    sd = None  # type: ignore[assignment]
    SD_AVAILABLE = False
    logger.warning("sounddevice not available — audio capture disabled")


class AudioCapture:
    """Threaded audio capture with real-time FFT spectral analysis.

    Captures stereo audio from either system loopback (WASAPI) or a
    microphone input, computes per-channel FFT magnitude spectra, and
    exposes the latest spectrum via a thread-safe ``get_latest()`` call.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz (default 44100).
    fft_size : int
        FFT window size in samples (default 512).  Produces ``fft_size // 2``
        unique frequency bins per channel.
    buffer_size : int
        sounddevice block size (default 1024).
    source : str
        ``"loopback"`` for system audio or ``"microphone"`` for mic input.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        fft_size: int = 512,
        buffer_size: int = 1024,
        source: str = "loopback",
    ) -> None:
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.fft_bins = fft_size // 2  # unique magnitudes (Nyquist excluded)
        self.buffer_size = buffer_size
        self._source = source

        # Latest spectrum: shape (2, fft_bins) — row 0 = L, row 1 = R
        self._lock = threading.Lock()
        self._latest: NDArray[np.floating[Any]] = np.zeros(
            (2, self.fft_bins), dtype=np.float32
        )

        # Hann window for FFT (pre-computed)
        self._window: NDArray[np.floating[Any]] = np.hanning(fft_size).astype(np.float32)
        # Window amplitude sum for correct FFT normalization (Hann sum ≈ N/2)
        self._window_sum: float = float(self._window.sum())

        # Ring buffer to accumulate samples between callbacks
        self._ring: NDArray[np.floating[Any]] = np.zeros(
            (fft_size, 2), dtype=np.float32
        )
        self._ring_pos = 0
        self._ring_filled = False  # True once ring has been fully written at least once

        # Stream management
        self._stream: Optional[Any] = None
        self._running = False
        self._error_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start audio capture stream."""
        if not SD_AVAILABLE:
            logger.error("sounddevice not installed — cannot start audio capture")
            return
        if self._running:
            return

        try:
            device, channels = self._resolve_device()
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.buffer_size,
                device=device,
                channels=channels,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            self._running = True
            logger.info(
                "Audio capture started (source=%s, device=%s, sr=%d, fft=%d)",
                self._source, device, self.sample_rate, self.fft_size,
            )
        except Exception as e:
            logger.error("Failed to start audio capture: %s", e)

    def stop(self) -> None:
        """Stop audio capture stream."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.debug("Error closing audio stream: %s", e)
            self._stream = None
        logger.info("Audio capture stopped")

    def get_latest(self) -> NDArray[np.floating[Any]]:
        """Return the latest stereo FFT magnitude spectrum.

        Returns
        -------
        NDArray
            Shape ``(2, fft_bins)`` float32 array normalised to roughly [0, 1].
            Row 0 = left channel, row 1 = right channel.
        """
        with self._lock:
            return self._latest.copy()

    def set_source(self, source: str) -> None:
        """Switch between ``"loopback"`` and ``"microphone"`` at runtime."""
        if source not in ("loopback", "microphone"):
            logger.warning("Unknown audio source '%s'; ignoring", source)
            return
        if source == self._source:
            return
        was_running = self._running
        if was_running:
            self.stop()
        self._source = source
        if was_running:
            self.start()

    @property
    def source(self) -> str:
        return self._source

    @property
    def is_running(self) -> bool:
        return self._running

    @staticmethod
    def get_device_list() -> List[dict]:
        """Return list of available audio devices."""
        if not SD_AVAILABLE:
            return []
        return sd.query_devices()  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_device(self) -> Tuple[Optional[int], int]:
        """Pick the sounddevice device index and channel count."""
        if self._source == "loopback":
            return self._find_loopback_device()
        # Microphone: use default input
        return None, 2  # None = default device

    def _find_loopback_device(self) -> Tuple[Optional[int], int]:
        """Find a loopback / stereo-mix capture device across all host APIs.

        Search order:
        1. WASAPI device with "loopback" or "stereo mix" in name (best latency)
        2. Any host-API device with "loopback" or "stereo mix" in name
        3. Default input device (microphone fallback)
        """
        if not SD_AVAILABLE:
            return None, 2
        try:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()

            # Find WASAPI host API index (preferred for low latency)
            wasapi_idx: Optional[int] = None
            for i, api in enumerate(hostapis):
                if "wasapi" in api["name"].lower():
                    wasapi_idx = i
                    break

            # Pass 1: WASAPI loopback devices
            if wasapi_idx is not None:
                for i, dev in enumerate(devices):
                    if dev["hostapi"] == wasapi_idx and dev["max_input_channels"] >= 2:
                        name_lower = dev["name"].lower()
                        if "loopback" in name_lower or "stereo mix" in name_lower:
                            logger.info("Using WASAPI loopback device: %s", dev["name"])
                            return i, min(int(dev["max_input_channels"]), 2)

            # Pass 2: loopback / stereo-mix on ANY host API
            for i, dev in enumerate(devices):
                if dev["max_input_channels"] >= 2:
                    name_lower = dev["name"].lower()
                    if "loopback" in name_lower or "stereo mix" in name_lower:
                        api_name = hostapis[dev["hostapi"]]["name"] if dev["hostapi"] < len(hostapis) else "?"
                        logger.info("Using loopback device: %s (api=%s)", dev["name"], api_name)
                        return i, min(int(dev["max_input_channels"]), 2)

        except Exception as e:
            logger.warning("Loopback device detection failed: %s", e)

        logger.info("No loopback device found; falling back to default input")
        return None, 2

    def _audio_callback(
        self,
        indata: NDArray[np.floating[Any]],
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """sounddevice InputStream callback — accumulates samples and runs FFT."""
        if status:
            self._error_count += 1
            if self._error_count <= 5:
                logger.debug("Audio callback status: %s", status)

        try:
            # indata shape: (frames, channels)
            n_channels = indata.shape[1] if indata.ndim == 2 else 1

            # Ensure stereo
            if n_channels == 1:
                stereo = np.column_stack([indata.ravel(), indata.ravel()])
            else:
                stereo = indata[:, :2].copy()

            # Feed into ring buffer
            n = stereo.shape[0]
            buf_len = self._ring.shape[0]

            if n >= buf_len:
                # More samples than buffer — just take the last buf_len
                self._ring[:] = stereo[-buf_len:]
                self._ring_pos = 0
                self._ring_filled = True
                self._compute_fft()
            else:
                # Partial fill
                space = buf_len - self._ring_pos
                if n <= space:
                    self._ring[self._ring_pos : self._ring_pos + n] = stereo
                    self._ring_pos += n
                else:
                    self._ring[self._ring_pos:] = stereo[:space]
                    remainder = n - space
                    self._ring[:remainder] = stereo[space:]
                    self._ring_pos = remainder
                    self._ring_filled = True

                # Compute FFT every callback once the ring has been filled
                if self._ring_filled:
                    self._compute_fft()

        except Exception as e:
            self._error_count += 1
            if self._error_count <= 5:
                logger.error("Audio callback error: %s", e)

    def _compute_fft(self) -> None:
        """Compute windowed FFT magnitudes for both channels."""
        # Reorder ring buffer so oldest sample is first
        buf = np.roll(self._ring, -self._ring_pos, axis=0)

        spectrum = np.zeros((2, self.fft_bins), dtype=np.float32)
        for ch in range(2):
            windowed = buf[:, ch] * self._window
            fft_result = np.fft.rfft(windowed)
            magnitudes = np.abs(fft_result[1 : self.fft_bins + 1])  # skip DC
            # Normalise by window sum so a full-scale sine → ~1.0.
            # Factor of 2 accounts for one-sided spectrum.
            magnitudes *= 2.0 / self._window_sum
            spectrum[ch] = magnitudes

        with self._lock:
            self._latest = spectrum
