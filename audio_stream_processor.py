"""
audio_stream_processor.py

Real-time audio stream processing for neural network training.
Handles audio capture, preprocessing, and feature extraction.
"""

import numpy as np
import torch
import torch.nn as nn
import librosa
import scipy.signal
from typing import Optional, Callable, Tuple
import logging
from logging_utils import log_step

class AudioPreprocessor:
    """Real-time audio preprocessing for neural network training."""
    
    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048, 
                 hop_length: int = 512, n_mels: int = 128):
        """
        Initialize audio preprocessor.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel frequency bins
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Pre-compute mel filter bank
        self.mel_filter_bank = librosa.filters.mel(
            sr=sample_rate, 
            n_fft=n_fft, 
            n_mels=n_mels
        )
        
        log_step("AudioPreprocessor initialized", 
                sample_rate=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels)
    
    def extract_mel_spectrogram(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram from audio data."""
        try:
            # Compute STFT
            stft = librosa.stft(
                audio_data, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length
            )
            
            # Convert to magnitude
            magnitude = np.abs(stft)
            
            # Apply mel filter bank
            mel_spec = np.dot(self.mel_filter_bank, magnitude)
            
            # Convert to log scale
            log_mel_spec = np.log(mel_spec + 1e-8)
            
            return log_mel_spec.astype(np.float32)
            
        except Exception as e:
            log_step("Mel spectrogram extraction error", error=str(e))
            # Return zero array as fallback
            return np.zeros((self.n_mels, 10), dtype=np.float32)
    
    def extract_mfcc(self, audio_data: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """Extract MFCC features from audio data."""
        try:
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            return mfcc.astype(np.float32)
            
        except Exception as e:
            log_step("MFCC extraction error", error=str(e))
            return np.zeros((n_mfcc, 10), dtype=np.float32)
    
    def extract_spectral_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract various spectral features."""
        try:
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data, sr=self.sample_rate
            )[0]
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=self.sample_rate
            )[0]
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            
            # Combine features
            features = np.stack([
                spectral_centroids,
                spectral_rolloff,
                zcr
            ], axis=0)
            
            return features.astype(np.float32)
            
        except Exception as e:
            log_step("Spectral features extraction error", error=str(e))
            return np.zeros((3, 10), dtype=np.float32)
    
    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio data to [-1, 1] range."""
        if np.max(np.abs(audio_data)) > 0:
            return audio_data / np.max(np.abs(audio_data))
        return audio_data
    
    def apply_noise_reduction(self, audio_data: np.ndarray, 
                            noise_factor: float = 0.1) -> np.ndarray:
        """Apply simple noise reduction."""
        # Simple high-pass filter to remove low-frequency noise
        b, a = scipy.signal.butter(4, 0.1, btype='high')
        filtered = scipy.signal.filtfilt(b, a, audio_data)
        return filtered.astype(np.float32)

class AudioFeatureExtractor:
    """Extract features from audio for neural network training."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.preprocessor = AudioPreprocessor(sample_rate)
        
        log_step("AudioFeatureExtractor initialized", sample_rate=sample_rate)
    
    def extract_comprehensive_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract comprehensive audio features."""
        try:
            # Normalize audio
            normalized_audio = self.preprocessor.normalize_audio(audio_data)
            
            # Extract different feature types
            mel_spec = self.preprocessor.extract_mel_spectrogram(normalized_audio)
            mfcc = self.preprocessor.extract_mfcc(normalized_audio)
            spectral_features = self.preprocessor.extract_spectral_features(normalized_audio)
            
            # Flatten and combine features
            mel_flat = mel_spec.flatten()
            mfcc_flat = mfcc.flatten()
            spectral_flat = spectral_features.flatten()
            
            # Combine all features
            combined_features = np.concatenate([
                mel_flat,
                mfcc_flat,
                spectral_flat
            ])
            
            # Pad or truncate to fixed size
            target_size = 2048  # Fixed feature vector size
            if len(combined_features) > target_size:
                combined_features = combined_features[:target_size]
            else:
                # Pad with zeros
                padding = target_size - len(combined_features)
                combined_features = np.pad(combined_features, (0, padding))
            
            return combined_features.astype(np.float32)
            
        except Exception as e:
            log_step("Comprehensive feature extraction error", error=str(e))
            return np.zeros(2048, dtype=np.float32)

class AudioAutoencoder(nn.Module):
    """Neural network for audio reconstruction and feature learning."""
    
    def __init__(self, input_size: int = 2048, hidden_size: int = 512, 
                 latent_size: int = 128):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, latent_size),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, input_size),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        log_step("AudioAutoencoder created", 
                input_size=input_size,
                hidden_size=hidden_size,
                latent_size=latent_size)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Get encoded representation."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode from latent representation."""
        return self.decoder(z)

class AudioClassifier(nn.Module):
    """Neural network for audio classification."""
    
    def __init__(self, input_size: int = 2048, num_classes: int = 10, 
                 hidden_size: int = 512):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes),
            nn.Softmax(dim=1)
        )
        
        log_step("AudioClassifier created", 
                input_size=input_size,
                num_classes=num_classes,
                hidden_size=hidden_size)
    
    def forward(self, x):
        return self.classifier(x)

# Example usage and testing
if __name__ == "__main__":
    print("🎵 Testing Audio Stream Processor...")
    
    # Create test audio data
    duration = 1.0  # 1 second
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate test signal (sine wave + noise)
    frequency = 440  # A4 note
    test_audio = np.sin(2 * np.pi * frequency * t) + 0.1 * np.random.randn(len(t))
    
    print(f"Generated test audio: {len(test_audio)} samples")
    
    # Test feature extraction
    feature_extractor = AudioFeatureExtractor(sample_rate)
    features = feature_extractor.extract_comprehensive_features(test_audio)
    
    print(f"Extracted features: {features.shape}")
    print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
    
    # Test neural networks
    autoencoder = AudioAutoencoder(input_size=2048)
    classifier = AudioClassifier(input_size=2048, num_classes=5)
    
    # Test forward pass
    input_tensor = torch.tensor(features).unsqueeze(0)
    
    with torch.no_grad():
        autoencoder_output = autoencoder(input_tensor)
        classifier_output = classifier(input_tensor)
        
        print(f"Autoencoder output shape: {autoencoder_output.shape}")
        print(f"Classifier output shape: {classifier_output.shape}")
        print(f"Classifier predictions: {classifier_output.numpy()}")
    
    print("✅ Audio Stream Processor test completed!")
