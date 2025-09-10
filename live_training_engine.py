"""
live_training_engine.py

Real-time neural network training from multiple live data sources.
Supports audio streams, visual data, and other real-time inputs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import threading
import time
import queue
import logging
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass
from collections import deque
import pyaudio
import wave
import struct
from logging_utils import log_step, log_runtime

@dataclass
class DataStream:
    """Configuration for a live data stream."""
    name: str
    source_type: str  # 'audio', 'visual', 'sensor', 'custom'
    sample_rate: int
    buffer_size: int
    channels: int = 1
    callback: Optional[Callable] = None
    preprocessing: Optional[Callable] = None
    active: bool = True

@dataclass
class TrainingMetrics:
    """Real-time training metrics."""
    loss: float
    accuracy: float
    learning_rate: float
    batch_size: int
    samples_processed: int
    training_speed: float  # samples per second
    timestamp: float

class LiveTrainingEngine:
    """
    Real-time neural network training from multiple live data sources.
    Supports streaming audio, visual data, and custom data sources.
    """
    
    def __init__(self, model: nn.Module, learning_rate: float = 0.001):
        """
        Initialize the live training engine.
        
        Args:
            model: PyTorch neural network model
            learning_rate: Learning rate for training
        """
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Data streams
        self.data_streams: Dict[str, DataStream] = {}
        self.data_queues: Dict[str, queue.Queue] = {}
        self.stream_threads: Dict[str, threading.Thread] = {}
        
        # Training state
        self.training_active = False
        self.training_thread = None
        self.metrics_history = deque(maxlen=1000)
        
        # Audio processing
        self.audio_stream = None
        self.pyaudio_instance = None
        
        # Performance tracking
        self.samples_processed = 0
        self.start_time = time.time()
        self.last_metrics_time = time.time()
        
        # Callbacks
        self.training_callbacks: List[Callable] = []
        self.metrics_callbacks: List[Callable] = []
        
        log_step("LiveTrainingEngine initialized", 
                model_params=sum(p.numel() for p in model.parameters()),
                learning_rate=learning_rate)
    
    def add_audio_stream(self, name: str, sample_rate: int = 44100, 
                        buffer_size: int = 1024, channels: int = 1,
                        preprocessing: Optional[Callable] = None):
        """
        Add a real-time audio stream for training.
        
        Args:
            name: Unique name for the stream
            sample_rate: Audio sample rate (Hz)
            buffer_size: Buffer size for audio processing
            channels: Number of audio channels
            preprocessing: Optional preprocessing function
        """
        try:
            # Initialize PyAudio
            if self.pyaudio_instance is None:
                self.pyaudio_instance = pyaudio.PyAudio()
            
            # Create audio stream
            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=buffer_size,
                stream_callback=self._audio_callback
            )
            
            # Create data stream configuration
            stream = DataStream(
                name=name,
                source_type='audio',
                sample_rate=sample_rate,
                buffer_size=buffer_size,
                channels=channels,
                preprocessing=preprocessing,
                active=True
            )
            
            self.data_streams[name] = stream
            self.data_queues[name] = queue.Queue(maxsize=100)
            
            log_step("Audio stream added", 
                    name=name, 
                    sample_rate=sample_rate,
                    buffer_size=buffer_size)
            
        except Exception as e:
            log_step("Failed to add audio stream", error=str(e), name=name)
            raise
    
    def add_visual_stream(self, name: str, width: int = 320, height: int = 240,
                         fps: int = 30, preprocessing: Optional[Callable] = None):
        """
        Add a real-time visual stream for training.
        
        Args:
            name: Unique name for the stream
            width: Frame width
            height: Frame height
            fps: Frames per second
            preprocessing: Optional preprocessing function
        """
        stream = DataStream(
            name=name,
            source_type='visual',
            sample_rate=fps,
            buffer_size=width * height * 3,  # RGB
            preprocessing=preprocessing,
            active=True
        )
        
        self.data_streams[name] = stream
        self.data_queues[name] = queue.Queue(maxsize=50)  # Smaller queue for visual data
        
        log_step("Visual stream added", 
                name=name, 
                width=width, 
                height=height, 
                fps=fps)
    
    def add_custom_stream(self, name: str, callback: Callable, 
                         sample_rate: int = 1, buffer_size: int = 1000,
                         preprocessing: Optional[Callable] = None):
        """
        Add a custom data stream with user-defined callback.
        
        Args:
            name: Unique name for the stream
            callback: Function to generate data
            sample_rate: Data generation rate
            buffer_size: Buffer size for data
            preprocessing: Optional preprocessing function
        """
        stream = DataStream(
            name=name,
            source_type='custom',
            sample_rate=sample_rate,
            buffer_size=buffer_size,
            callback=callback,
            preprocessing=preprocessing,
            active=True
        )
        
        self.data_streams[name] = stream
        self.data_queues[name] = queue.Queue(maxsize=1000)
        
        # Start custom stream thread
        thread = threading.Thread(
            target=self._custom_stream_worker,
            args=(name, callback),
            daemon=True
        )
        thread.start()
        self.stream_threads[name] = thread
        
        log_step("Custom stream added", name=name, sample_rate=sample_rate)
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for real-time audio processing."""
        try:
            # Convert audio data to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Process audio data
            if len(audio_data) > 0:
                # Apply preprocessing if available
                for stream_name, stream in self.data_streams.items():
                    if stream.source_type == 'audio' and stream.active:
                        processed_data = audio_data
                        if stream.preprocessing:
                            processed_data = stream.preprocessing(audio_data)
                        
                        # Add to queue (non-blocking)
                        try:
                            self.data_queues[stream_name].put_nowait(processed_data)
                        except queue.Full:
                            # Drop oldest data if queue is full
                            try:
                                self.data_queues[stream_name].get_nowait()
                                self.data_queues[stream_name].put_nowait(processed_data)
                            except queue.Empty:
                                pass
            
            return (in_data, pyaudio.paContinue)
            
        except Exception as e:
            log_step("Audio callback error", error=str(e))
            return (in_data, pyaudio.paContinue)
    
    def _custom_stream_worker(self, stream_name: str, callback: Callable):
        """Worker thread for custom data streams."""
        try:
            while self.training_active and stream_name in self.data_streams:
                stream = self.data_streams[stream_name]
                if not stream.active:
                    time.sleep(0.1)
                    continue
                
                # Generate data using callback
                data = callback()
                if data is not None:
                    # Apply preprocessing
                    processed_data = data
                    if stream.preprocessing:
                        processed_data = stream.preprocessing(data)
                    
                    # Add to queue
                    try:
                        self.data_queues[stream_name].put_nowait(processed_data)
                    except queue.Full:
                        # Drop oldest data
                        try:
                            self.data_queues[stream_name].get_nowait()
                            self.data_queues[stream_name].put_nowait(processed_data)
                        except queue.Empty:
                            pass
                
                # Sleep based on sample rate
                sleep_time = 1.0 / stream.sample_rate
                time.sleep(sleep_time)
                
        except Exception as e:
            log_step("Custom stream worker error", 
                    stream_name=stream_name, 
                    error=str(e))
    
    def start_training(self):
        """Start real-time training from live data streams."""
        if self.training_active:
            log_step("Training already active")
            return
        
        self.training_active = True
        self.start_time = time.time()
        self.samples_processed = 0
        
        # Start training thread
        self.training_thread = threading.Thread(
            target=self._training_loop,
            daemon=True
        )
        self.training_thread.start()
        
        # Start audio stream if available
        if self.audio_stream and not self.audio_stream.is_active():
            self.audio_stream.start_stream()
        
        log_step("Live training started", 
                streams=len(self.data_streams),
                active_streams=sum(1 for s in self.data_streams.values() if s.active))
    
    def stop_training(self):
        """Stop real-time training."""
        self.training_active = False
        
        # Stop audio stream
        if self.audio_stream and self.audio_stream.is_active():
            self.audio_stream.stop_stream()
        
        # Wait for training thread to finish
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=2.0)
        
        log_step("Live training stopped")
    
    def _training_loop(self):
        """Main training loop for real-time learning."""
        log_step("Training loop started")
        
        while self.training_active:
            try:
                # Collect data from all active streams
                batch_data = {}
                batch_targets = {}
                
                for stream_name, stream in self.data_streams.items():
                    if not stream.active:
                        continue
                    
                    # Get data from queue
                    try:
                        data = self.data_queues[stream_name].get_nowait()
                        batch_data[stream_name] = data
                        
                        # Create targets (for now, use data as target for autoencoder-style learning)
                        batch_targets[stream_name] = data
                        
                    except queue.Empty:
                        continue
                
                # Train if we have data
                if batch_data:
                    self._train_step(batch_data, batch_targets)
                    self.samples_processed += sum(len(data) for data in batch_data.values())
                
                # Update metrics periodically
                current_time = time.time()
                if current_time - self.last_metrics_time > 1.0:  # Every second
                    self._update_metrics()
                    self.last_metrics_time = current_time
                
                # Small sleep to prevent busy waiting
                time.sleep(0.001)
                
            except Exception as e:
                log_step("Training loop error", error=str(e))
                time.sleep(0.1)
        
        log_step("Training loop ended")
    
    def _train_step(self, batch_data: Dict[str, np.ndarray], 
                   batch_targets: Dict[str, np.ndarray]):
        """Perform a single training step."""
        try:
            self.optimizer.zero_grad()
            
            total_loss = 0.0
            num_streams = 0
            
            for stream_name, data in batch_data.items():
                # Convert to tensor
                input_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
                target_tensor = torch.tensor(batch_targets[stream_name], dtype=torch.float32).unsqueeze(0)
                
                # Forward pass
                output = self.model(input_tensor)
                
                # Calculate loss
                loss = self.criterion(output, target_tensor)
                total_loss += loss.item()
                num_streams += 1
                
                # Backward pass
                loss.backward()
            
            # Update weights
            if num_streams > 0:
                self.optimizer.step()
                
                # Call training callbacks
                for callback in self.training_callbacks:
                    try:
                        callback(total_loss / num_streams, self.samples_processed)
                    except Exception as e:
                        log_step("Training callback error", error=str(e))
            
        except Exception as e:
            log_step("Train step error", error=str(e))
    
    def _update_metrics(self):
        """Update and broadcast training metrics."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time > 0:
            training_speed = self.samples_processed / elapsed_time
            
            metrics = TrainingMetrics(
                loss=0.0,  # Will be updated by training callbacks
                accuracy=0.0,  # Placeholder
                learning_rate=self.optimizer.param_groups[0]['lr'],
                batch_size=1,  # Real-time processing
                samples_processed=self.samples_processed,
                training_speed=training_speed,
                timestamp=current_time
            )
            
            self.metrics_history.append(metrics)
            
            # Call metrics callbacks
            for callback in self.metrics_callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    log_step("Metrics callback error", error=str(e))
    
    def add_training_callback(self, callback: Callable):
        """Add a callback for training events."""
        self.training_callbacks.append(callback)
    
    def add_metrics_callback(self, callback: Callable):
        """Add a callback for metrics updates."""
        self.metrics_callbacks.append(callback)
    
    def get_metrics(self) -> List[TrainingMetrics]:
        """Get recent training metrics."""
        return list(self.metrics_history)
    
    def set_learning_rate(self, lr: float):
        """Update the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        log_step("Learning rate updated", new_lr=lr)
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_training()
        
        if self.audio_stream:
            self.audio_stream.close()
        
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
        
        log_step("LiveTrainingEngine cleaned up")


# Example usage and testing
if __name__ == "__main__":
    # Create a simple neural network for testing
    class SimpleNet(nn.Module):
        def __init__(self, input_size=1024, hidden_size=512, output_size=1024):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
        
        def forward(self, x):
            return self.network(x)
    
    # Test the live training engine
    print("🚀 Testing Live Training Engine...")
    
    model = SimpleNet()
    engine = LiveTrainingEngine(model, learning_rate=0.001)
    
    # Add a custom test stream
    def test_data_generator():
        return np.random.randn(1024).astype(np.float32)
    
    engine.add_custom_stream("test_stream", test_data_generator, sample_rate=10)
    
    # Add callbacks
    def training_callback(loss, samples):
        print(f"Training: Loss={loss:.4f}, Samples={samples}")
    
    def metrics_callback(metrics):
        print(f"Metrics: Speed={metrics.training_speed:.1f} samples/sec, "
              f"Total={metrics.samples_processed}")
    
    engine.add_training_callback(training_callback)
    engine.add_metrics_callback(metrics_callback)
    
    # Start training
    engine.start_training()
    
    try:
        # Run for 10 seconds
        time.sleep(10)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop_training()
        engine.cleanup()
    
    print("✅ Live Training Engine test completed!")
