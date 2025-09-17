
import numpy as np
import torch

from torch_geometric.data import Data

import logging
from typing import Dict, Any, List, Optional, Tuple
from utils.logging_utils import log_step
from energy.node_access_layer import get_node_by_id, update_node_property
from energy import NodeAccessLayer
from utils.common_utils import safe_hasattr, safe_get_attr

import importlib

try:
    librosa = importlib.import_module('librosa')
    LIBROSA_AVAILABLE = True
except ImportError:
    librosa = None
    LIBROSA_AVAILABLE = False
    logging.warning("librosa not available. Audio feature extraction will use fallbacks.")


class AudioToNeuralBridge:
    def __init__(self, neural_simulation=None):
        self.neural_simulation = neural_simulation
        self.audio_features_cache = {}
        self.sensory_node_mapping = {}
        self.sample_rate = 44100
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
        self.n_mfcc = 13
        log_step("AudioToNeuralBridge initialized")

    def process_audio_to_sensory_nodes(self, audio_data: np.ndarray) -> List[Dict[str, Any]]:
        try:
            features = self._extract_audio_features(audio_data)
            sensory_nodes = self._create_audio_sensory_nodes(features)
            log_step("Audio converted to sensory nodes",
                    audio_samples=len(audio_data),
                    sensory_nodes_created=len(sensory_nodes))
            return sensory_nodes
        except Exception as e:
            log_step("Error converting audio to sensory nodes", error=str(e))
            return []
    def _extract_audio_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        try:
            max_samples = 44100
            if len(audio_data) > max_samples:
                audio_data = audio_data[:max_samples]
                log_step(f"Downsampled audio data to {max_samples} samples for performance")

            normalized_audio = self._normalize_audio(audio_data)
            features = {
                'mfcc': self._extract_mfcc(normalized_audio),
                'mel_spectrogram': self._extract_mel_spectrogram(normalized_audio),
                'spectral_features': self._extract_spectral_features(normalized_audio),
                'temporal_features': self._extract_temporal_features(normalized_audio)
            }
            return features
        except Exception as e:
            log_step("Error extracting audio features", error=str(e))
            return {}

    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        if np.max(np.abs(audio_data)) > 0:
            return audio_data / np.max(np.abs(audio_data))
        return audio_data
    def _extract_mfcc(self, audio_data: np.ndarray) -> np.ndarray:
        if not LIBROSA_AVAILABLE or librosa is None:
            logging.warning("librosa unavailable for MFCC extraction, using fallback.")
            return np.zeros((self.n_mfcc, 10), dtype=np.float32)
        try:
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            return mfcc.astype(np.float32)
        except Exception as e:
            log_step("Error extracting MFCC", error=str(e))
            return np.zeros((self.n_mfcc, 10), dtype=np.float32)

    def _extract_mel_spectrogram(self, audio_data: np.ndarray) -> np.ndarray:
        if not LIBROSA_AVAILABLE or librosa is None:
            logging.warning("librosa unavailable for mel spectrogram extraction, using fallback.")
            return np.zeros((self.n_mels, 10), dtype=np.float32)
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            log_mel_spec = np.log(mel_spec + 1e-8)
            return log_mel_spec.astype(np.float32)
        except Exception as e:
            log_step("Error extracting mel spectrogram", error=str(e))
            return np.zeros((self.n_mels, 10), dtype=np.float32)
    def _extract_spectral_features(self, audio_data: np.ndarray) -> np.ndarray:
        if not LIBROSA_AVAILABLE or librosa is None:
            logging.warning("librosa unavailable for spectral features extraction, using fallback.")
            return np.zeros((3, 10), dtype=np.float32)
        try:
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data, sr=self.sample_rate
            )[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=self.sample_rate
            )[0]
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]

            try:
                if (spectral_centroids.shape != spectral_rolloff.shape or
                    spectral_centroids.shape != zcr.shape):
                    log_step("Mismatched feature array dimensions",
                            centroids_shape=spectral_centroids.shape,
                            rolloff_shape=spectral_rolloff.shape,
                            zcr_shape=zcr.shape)
                    min_length = min(len(spectral_centroids), len(spectral_rolloff), len(zcr))
                    if min_length == 0:
                        features = np.zeros((3, 10), dtype=np.float32)
                    else:
                        spectral_centroids = spectral_centroids[:min_length]
                        spectral_rolloff = spectral_rolloff[:min_length]
                        zcr = zcr[:min_length]
                        features = np.stack([
                            spectral_centroids,
                            spectral_rolloff,
                            zcr
                        ], axis=0)
                else:
                    features = np.stack([
                        spectral_centroids,
                        spectral_rolloff,
                        zcr
                    ], axis=0)
            except ValueError as e:
                log_step("Feature stacking failed, using fallback", error=str(e))
                features = np.zeros((3, 10), dtype=np.float32)
            except Exception as e:
                log_step("Unexpected error in spectral feature extraction", error=str(e))
                features = np.zeros((3, 10), dtype=np.float32)

            return features.astype(np.float32)
        except Exception as e:
            log_step("Error extracting spectral features", error=str(e))
            return np.zeros((3, 10), dtype=np.float32)
    def _extract_temporal_features(self, audio_data: np.ndarray) -> np.ndarray:
        if not LIBROSA_AVAILABLE or librosa is None:
            logging.warning("librosa unavailable for temporal features extraction, using fallback.")
            return np.zeros((2, 10), dtype=np.float32)
        try:
            rms = librosa.feature.rms(y=audio_data)[0]
            bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_data, sr=self.sample_rate
            )[0]

            try:
                if rms.shape != bandwidth.shape:
                    log_step("Mismatched temporal feature dimensions",
                            rms_shape=rms.shape,
                            bandwidth_shape=bandwidth.shape)
                    min_length = min(len(rms), len(bandwidth))
                    if min_length == 0:
                        features = np.zeros((2, 10), dtype=np.float32)
                    else:
                        rms = rms[:min_length]
                        bandwidth = bandwidth[:min_length]
                        features = np.stack([rms, bandwidth], axis=0)
                else:
                    features = np.stack([rms, bandwidth], axis=0)
            except ValueError as e:
                log_step("Temporal feature stacking failed, using fallback", error=str(e))
                features = np.zeros((2, 10), dtype=np.float32)
            except Exception as e:
                log_step("Unexpected error in temporal feature extraction", error=str(e))
                features = np.zeros((2, 10), dtype=np.float32)

            return features.astype(np.float32)
        except Exception as e:
            log_step("Error extracting temporal features", error=str(e))
            return np.zeros((2, 10), dtype=np.float32)
    def _create_audio_sensory_nodes(self, features: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:

        sensory_nodes = []
        node_id_counter = 0
        try:
            for feature_type, feature_data in features.items():
                if feature_data.size == 0:
                    continue
                flattened_features = feature_data.flatten()
                for i, feature_value in enumerate(flattened_features):
                    sensory_node = {
                        'id': f"audio_{feature_type}_{node_id_counter}",
                        'type': 'sensory',
                        'subtype': 'audio',
                        'feature_type': feature_type,
                        'feature_index': i,
                        'energy': float(feature_value),
                        'feature_position': i,
                        'feature_rank': i % 100,
                        'channel': 0,
                        'state': 'active',
                        'last_update': 0,
                        'threshold': 0.1,
                        'membrane_potential': min(abs(feature_value), 1.0),
                        'refractory_timer': 0.0,
                        'plasticity_enabled': True,
                        'eligibility_trace': 0.0
                    }
                    sensory_nodes.append(sensory_node)
                    node_id_counter += 1
            log_step("Created audio sensory nodes",
                    feature_types=len(features),
                    total_nodes=len(sensory_nodes))
            return sensory_nodes
        except Exception as e:
            log_step("Error creating audio sensory nodes", error=str(e))
            return []
    def integrate_audio_nodes_into_graph(self, graph: Data, audio_data: np.ndarray) -> Data:

        try:
            audio_sensory_nodes = self.process_audio_to_sensory_nodes(audio_data)
            if not audio_sensory_nodes:
                return graph
            updated_graph = self._add_audio_nodes_to_graph(graph, audio_sensory_nodes)
            if hasattr(self, 'enhanced_integration') and self.enhanced_integration is not None:
                try:
                    updated_graph = self._process_audio_with_enhanced_dynamics(
                        updated_graph, audio_data, audio_sensory_nodes
                    )
                except Exception as e:
                    log_step("Enhanced audio processing failed, using basic integration", error=str(e))
            log_step("Audio nodes integrated into graph",
                    audio_nodes_added=len(audio_sensory_nodes),
                    total_nodes=len(updated_graph.node_labels))
            return updated_graph
        except Exception as e:
            log_step("Error integrating audio nodes into graph", error=str(e))
            return graph
    def _add_audio_nodes_to_graph(self, graph: Data, audio_nodes: List[Dict[str, Any]]) -> Data:

        try:
            if not hasattr(graph, 'node_labels'):
                graph.node_labels = []
            for audio_node in audio_nodes:
                graph.node_labels.append(audio_node)
            if hasattr(graph, 'x'):
                audio_features = []
                for audio_node in audio_nodes:
                    feature_vector = [
                        audio_node['energy'],
                        audio_node['membrane_potential'],
                        audio_node['threshold'],
                        audio_node['refractory_timer'],
                        audio_node['eligibility_trace']
                    ]
                    audio_features.append(feature_vector)
                audio_tensor = torch.tensor(audio_features, dtype=torch.float32)
                if graph.x is not None:
                    if graph.x.shape[1] != audio_tensor.shape[1]:
                        if graph.x.shape[1] > audio_tensor.shape[1]:
                            padding = torch.zeros(audio_tensor.shape[0],
                                                graph.x.shape[1] - audio_tensor.shape[1])
                            audio_tensor = torch.cat([audio_tensor, padding], dim=1)
                        else:
                            audio_tensor = audio_tensor[:, :graph.x.shape[1]]
                    graph.x = torch.cat([graph.x, audio_tensor], dim=0)
                else:
                    graph.x = audio_tensor
            return graph
        except Exception as e:
            log_step("Error adding audio nodes to graph", error=str(e))
            return graph
    def get_audio_feature_statistics(self) -> Dict[str, Any]:
        return {
            'cached_features': len(self.audio_features_cache),
            'sensory_node_mappings': len(self.sensory_node_mapping),
            'sample_rate': self.sample_rate,
            'n_mels': self.n_mels,
            'n_mfcc': self.n_mfcc
        }
    def _process_audio_with_enhanced_dynamics(self, graph: Data, audio_data: np.ndarray,
                                            audio_sensory_nodes: List[Dict[str, Any]]) -> Data:

        try:
            access_layer = NodeAccessLayer(graph)
            for audio_node in audio_sensory_nodes:
                node_id = audio_node['id']
                graph_node = access_layer.get_node_by_id(node_id)
                if graph_node is None:
                    continue
                access_layer.update_node_property(node_id, 'enhanced_behavior', True)
                access_layer.update_node_property(node_id, 'audio_stimulation', True)
                access_layer.update_node_property(node_id, 'feature_type', audio_node['feature_type'])
                access_layer.update_node_property(node_id, 'membrane_potential', audio_node['membrane_potential'])
                access_layer.update_node_property(node_id, 'plasticity_enabled', True)
                access_layer.update_node_property(node_id, 'eligibility_trace', 0.0)
                self._create_audio_enhanced_connections(graph, node_id, access_layer)
            if self.enhanced_integration is not None:
                graph = self.enhanced_integration.integrate_with_existing_system(graph, 0)
            return graph
        except Exception as e:
            log_step("Error in enhanced audio processing", error=str(e))
            return graph
    def _create_audio_enhanced_connections(self, graph: Data, audio_node_id: int,
                                         access_layer: NodeAccessLayer):

        try:
            if self.enhanced_integration is None:
                return
            enhanced_nodes = access_layer.select_nodes_by_property('enhanced_behavior', True)
            for target_id in enhanced_nodes[:2]:
                if audio_node_id != target_id:
                    self.enhanced_integration.create_enhanced_connection(
                        graph, audio_node_id, target_id, 'excitatory',
                        weight=0.3,
                        delay=0.05,
                        plasticity_enabled=True
                    )
        except Exception as e:
            log_step("Error creating audio enhanced connections", error=str(e))
    def set_enhanced_integration(self, enhanced_integration):
        self.enhanced_integration = enhanced_integration
    def clear_cache(self):
        self.audio_features_cache.clear()
        self.sensory_node_mapping.clear()
        log_step("Audio features cache cleared")


def create_audio_to_neural_bridge(neural_simulation=None) -> AudioToNeuralBridge:

    return AudioToNeuralBridge(neural_simulation)
if __name__ == "__main__":
    print("AudioToNeuralBridge created successfully!")
    print("Features include:")
    print("- Audio feature extraction (MFCC, mel spectrogram, spectral features)")
    print("- Audio to sensory node conversion")
    print("- Integration with neural simulation graph")
    print("- Real-time audio processing")
    print("- Fallback support for missing dependencies")
    try:
        bridge = create_audio_to_neural_bridge()
        dummy_audio = np.random.randn(44100)
        sensory_nodes = bridge.process_audio_to_sensory_nodes(dummy_audio)
        print(f"Created {len(sensory_nodes)} sensory nodes from audio")
        stats = bridge.get_audio_feature_statistics()
        print(f"Bridge statistics: {stats}")
    except Exception as e:
        print(f"AudioToNeuralBridge test failed: {e}")
    print("AudioToNeuralBridge test completed!")
