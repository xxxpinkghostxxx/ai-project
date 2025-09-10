"""
live_training_interface.py

Comprehensive live training interface with multiple data sources.
Integrates audio streams, visual data, and real-time neural network training.
"""

import torch
import torch.nn as nn
import numpy as np
import threading
import time
import queue
from typing import Dict, List, Optional, Any, Callable
import logging
from logging_utils import log_step, log_runtime

# Import our custom modules
from live_training_engine import LiveTrainingEngine, DataStream, TrainingMetrics
from audio_stream_processor import AudioFeatureExtractor, AudioAutoencoder, AudioClassifier
from performance_optimizer_v2 import PerformanceOptimizer

class LiveTrainingInterface:
    """
    Comprehensive live training interface for real-time neural network training.
    Supports multiple data sources including audio, visual, and custom streams.
    """
    
    def __init__(self, model_type: str = "autoencoder", learning_rate: float = 0.001):
        """
        Initialize the live training interface.
        
        Args:
            model_type: Type of neural network model ('autoencoder', 'classifier', 'custom')
            learning_rate: Learning rate for training
        """
        self.model_type = model_type
        self.learning_rate = learning_rate
        
        # Initialize components
        self.training_engine = None
        self.audio_processor = None
        self.performance_optimizer = PerformanceOptimizer()
        
        # Training state
        self.training_active = False
        self.training_thread = None
        
        # Data sources
        self.audio_streams = {}
        self.visual_streams = {}
        self.custom_streams = {}
        
        # Models
        self.models = {}
        self.current_model = None
        
        # Callbacks
        self.training_callbacks = []
        self.metrics_callbacks = []
        self.ui_callbacks = []
        
        # Performance tracking
        self.start_time = time.time()
        self.samples_processed = 0
        
        log_step("LiveTrainingInterface initialized", 
                model_type=model_type,
                learning_rate=learning_rate)
    
    def initialize_models(self):
        """Initialize neural network models based on model type."""
        try:
            if self.model_type == "autoencoder":
                # Audio autoencoder for feature learning
                self.models["audio_autoencoder"] = AudioAutoencoder(
                    input_size=2048, 
                    hidden_size=512, 
                    latent_size=128
                )
                self.current_model = self.models["audio_autoencoder"]
                
            elif self.model_type == "classifier":
                # Audio classifier for real-time classification
                self.models["audio_classifier"] = AudioClassifier(
                    input_size=2048, 
                    num_classes=10, 
                    hidden_size=512
                )
                self.current_model = self.models["audio_classifier"]
                
            elif self.model_type == "custom":
                # Custom model - user will provide
                log_step("Custom model type selected - user must provide model")
                return
            
            # Initialize training engine with current model
            if self.current_model:
                self.training_engine = LiveTrainingEngine(
                    self.current_model, 
                    self.learning_rate
                )
                
                # Add callbacks
                self.training_engine.add_training_callback(self._on_training_update)
                self.training_engine.add_metrics_callback(self._on_metrics_update)
            
            # Initialize audio processor
            self.audio_processor = AudioFeatureExtractor()
            
            log_step("Models initialized successfully", 
                    model_type=self.model_type,
                    model_params=sum(p.numel() for p in self.current_model.parameters()) if self.current_model else 0)
            
        except Exception as e:
            log_step("Model initialization error", error=str(e))
            raise
    
    def add_audio_stream(self, name: str, sample_rate: int = 44100, 
                        buffer_size: int = 1024, channels: int = 1):
        """
        Add a real-time audio stream for training.
        
        Args:
            name: Unique name for the stream
            sample_rate: Audio sample rate (Hz)
            buffer_size: Buffer size for audio processing
            channels: Number of audio channels
        """
        try:
            if not self.training_engine:
                self.initialize_models()
            
            # Add audio stream to training engine
            self.training_engine.add_audio_stream(
                name=name,
                sample_rate=sample_rate,
                buffer_size=buffer_size,
                channels=channels,
                preprocessing=self._audio_preprocessing
            )
            
            self.audio_streams[name] = {
                "sample_rate": sample_rate,
                "buffer_size": buffer_size,
                "channels": channels,
                "active": True
            }
            
            log_step("Audio stream added", 
                    name=name, 
                    sample_rate=sample_rate,
                    buffer_size=buffer_size)
            
        except Exception as e:
            log_step("Failed to add audio stream", error=str(e), name=name)
            raise
    
    def add_visual_stream(self, name: str, width: int = 320, height: int = 240, fps: int = 30):
        """
        Add a real-time visual stream for training.
        
        Args:
            name: Unique name for the stream
            width: Frame width
            height: Frame height
            fps: Frames per second
        """
        try:
            if not self.training_engine:
                self.initialize_models()
            
            # Add visual stream to training engine
            self.training_engine.add_visual_stream(
                name=name,
                width=width,
                height=height,
                fps=fps,
                preprocessing=self._visual_preprocessing
            )
            
            self.visual_streams[name] = {
                "width": width,
                "height": height,
                "fps": fps,
                "active": True
            }
            
            log_step("Visual stream added", 
                    name=name, 
                    width=width, 
                    height=height, 
                    fps=fps)
            
        except Exception as e:
            log_step("Failed to add visual stream", error=str(e), name=name)
            raise
    
    def add_custom_stream(self, name: str, data_generator: Callable, 
                         sample_rate: int = 1, buffer_size: int = 1000):
        """
        Add a custom data stream with user-defined generator.
        
        Args:
            name: Unique name for the stream
            data_generator: Function that generates data
            sample_rate: Data generation rate
            buffer_size: Buffer size for data
        """
        try:
            if not self.training_engine:
                self.initialize_models()
            
            # Add custom stream to training engine
            self.training_engine.add_custom_stream(
                name=name,
                callback=data_generator,
                sample_rate=sample_rate,
                buffer_size=buffer_size,
                preprocessing=self._custom_preprocessing
            )
            
            self.custom_streams[name] = {
                "sample_rate": sample_rate,
                "buffer_size": buffer_size,
                "active": True
            }
            
            log_step("Custom stream added", 
                    name=name, 
                    sample_rate=sample_rate,
                    buffer_size=buffer_size)
            
        except Exception as e:
            log_step("Failed to add custom stream", error=str(e), name=name)
            raise
    
    def _audio_preprocessing(self, audio_data: np.ndarray) -> np.ndarray:
        """Preprocess audio data for neural network training."""
        try:
            if self.audio_processor:
                return self.audio_processor.extract_comprehensive_features(audio_data)
            else:
                # Fallback: simple normalization
                return np.clip(audio_data, -1.0, 1.0).astype(np.float32)
        except Exception as e:
            log_step("Audio preprocessing error", error=str(e))
            return np.zeros(2048, dtype=np.float32)
    
    def _visual_preprocessing(self, visual_data: np.ndarray) -> np.ndarray:
        """Preprocess visual data for neural network training."""
        try:
            # Simple preprocessing: flatten and normalize
            if visual_data.ndim > 1:
                visual_data = visual_data.flatten()
            
            # Normalize to [0, 1]
            if visual_data.max() > visual_data.min():
                visual_data = (visual_data - visual_data.min()) / (visual_data.max() - visual_data.min())
            
            # Pad or truncate to fixed size
            target_size = 2048
            if len(visual_data) > target_size:
                visual_data = visual_data[:target_size]
            else:
                padding = target_size - len(visual_data)
                visual_data = np.pad(visual_data, (0, padding))
            
            return visual_data.astype(np.float32)
            
        except Exception as e:
            log_step("Visual preprocessing error", error=str(e))
            return np.zeros(2048, dtype=np.float32)
    
    def _custom_preprocessing(self, custom_data: np.ndarray) -> np.ndarray:
        """Preprocess custom data for neural network training."""
        try:
            # Convert to numpy array if needed
            if not isinstance(custom_data, np.ndarray):
                custom_data = np.array(custom_data)
            
            # Flatten if multi-dimensional
            if custom_data.ndim > 1:
                custom_data = custom_data.flatten()
            
            # Normalize
            if custom_data.max() > custom_data.min():
                custom_data = (custom_data - custom_data.min()) / (custom_data.max() - custom_data.min())
            
            # Pad or truncate to fixed size
            target_size = 2048
            if len(custom_data) > target_size:
                custom_data = custom_data[:target_size]
            else:
                padding = target_size - len(custom_data)
                custom_data = np.pad(custom_data, (0, padding))
            
            return custom_data.astype(np.float32)
            
        except Exception as e:
            log_step("Custom preprocessing error", error=str(e))
            return np.zeros(2048, dtype=np.float32)
    
    def start_training(self):
        """Start live training from all configured data sources."""
        try:
            if not self.training_engine:
                self.initialize_models()
            
            if not self.training_engine:
                raise RuntimeError("No training engine available")
            
            self.training_active = True
            self.start_time = time.time()
            self.samples_processed = 0
            
            # Start training engine
            self.training_engine.start_training()
            
            # Start performance monitoring
            self.performance_optimizer.add_optimization_callback(self._on_performance_update)
            
            log_step("Live training started", 
                    streams=len(self.audio_streams) + len(self.visual_streams) + len(self.custom_streams),
                    model_type=self.model_type)
            
            # Notify UI callbacks
            for callback in self.ui_callbacks:
                try:
                    callback("training_started", {"status": "active"})
                except Exception as e:
                    log_step("UI callback error", error=str(e))
            
        except Exception as e:
            log_step("Failed to start training", error=str(e))
            raise
    
    def stop_training(self):
        """Stop live training."""
        try:
            self.training_active = False
            
            if self.training_engine:
                self.training_engine.stop_training()
            
            log_step("Live training stopped")
            
            # Notify UI callbacks
            for callback in self.ui_callbacks:
                try:
                    callback("training_stopped", {"status": "inactive"})
                except Exception as e:
                    log_step("UI callback error", error=str(e))
            
        except Exception as e:
            log_step("Failed to stop training", error=str(e))
    
    def _on_training_update(self, loss: float, samples: int):
        """Handle training updates."""
        self.samples_processed = samples
        
        # Call training callbacks
        for callback in self.training_callbacks:
            try:
                callback(loss, samples)
            except Exception as e:
                log_step("Training callback error", error=str(e))
    
    def _on_metrics_update(self, metrics: TrainingMetrics):
        """Handle metrics updates."""
        # Call metrics callbacks
        for callback in self.metrics_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                log_step("Metrics callback error", error=str(e))
    
    def _on_performance_update(self, metrics):
        """Handle performance updates."""
        # Log performance warnings
        if metrics.step_time > 0.1:  # 100ms threshold
            log_step("Performance warning", 
                    step_time=metrics.step_time,
                    throughput=metrics.throughput)
    
    def add_training_callback(self, callback: Callable):
        """Add callback for training events."""
        self.training_callbacks.append(callback)
    
    def add_metrics_callback(self, callback: Callable):
        """Add callback for metrics updates."""
        self.metrics_callbacks.append(callback)
    
    def add_ui_callback(self, callback: Callable):
        """Add callback for UI updates."""
        self.ui_callbacks.append(callback)
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        elapsed_time = time.time() - self.start_time
        
        return {
            "training_active": self.training_active,
            "model_type": self.model_type,
            "learning_rate": self.learning_rate,
            "samples_processed": self.samples_processed,
            "elapsed_time": elapsed_time,
            "audio_streams": len(self.audio_streams),
            "visual_streams": len(self.visual_streams),
            "custom_streams": len(self.custom_streams),
            "total_streams": len(self.audio_streams) + len(self.visual_streams) + len(self.custom_streams)
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if self.performance_optimizer:
            return self.performance_optimizer.get_performance_stats()
        return {"status": "no_data"}
    
    def set_learning_rate(self, lr: float):
        """Update learning rate."""
        self.learning_rate = lr
        if self.training_engine:
            self.training_engine.set_learning_rate(lr)
        log_step("Learning rate updated", new_lr=lr)
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_training()
        
        if self.training_engine:
            self.training_engine.cleanup()
        
        log_step("LiveTrainingInterface cleaned up")

# Example usage and testing
if __name__ == "__main__":
    print("🚀 Testing Live Training Interface...")
    
    # Create interface
    interface = LiveTrainingInterface(model_type="autoencoder", learning_rate=0.001)
    
    # Add callbacks
    def training_callback(loss, samples):
        print(f"Training: Loss={loss:.4f}, Samples={samples}")
    
    def metrics_callback(metrics):
        print(f"Metrics: Speed={metrics.training_speed:.1f} samples/sec")
    
    def ui_callback(event, data):
        print(f"UI Event: {event}, Data: {data}")
    
    interface.add_training_callback(training_callback)
    interface.add_metrics_callback(metrics_callback)
    interface.add_ui_callback(ui_callback)
    
    # Add test streams
    def test_data_generator():
        return np.random.randn(1024).astype(np.float32)
    
    interface.add_custom_stream("test_stream", test_data_generator, sample_rate=10)
    
    # Start training
    interface.start_training()
    
    try:
        # Run for 10 seconds
        time.sleep(10)
        
        # Get status
        status = interface.get_training_status()
        print(f"Training Status: {status}")
        
        # Get performance stats
        perf_stats = interface.get_performance_stats()
        print(f"Performance Stats: {perf_stats}")
        
    except KeyboardInterrupt:
        pass
    finally:
        interface.stop_training()
        interface.cleanup()
    
    print("✅ Live Training Interface test completed!")
