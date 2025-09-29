
import json
import logging
import os
import time
from typing import Any, Dict, Optional

import torch
from torch_geometric.data import Data

from src.utils.logging_utils import log_step
from src.utils.print_utils import print_invalid_slot


class NeuralMapPersistence:

    def __init__(self, save_directory: str = "data/neural_maps"):

        self.save_directory = save_directory
        self.max_slots = 10
        self.current_slot = 0
        self.slot_metadata = {}
        os.makedirs(self.save_directory, exist_ok=True)
        self._load_slot_metadata()
        log_step("NeuralMapPersistence initialized",
                save_directory=save_directory,
                max_slots=self.max_slots)
    def save_neural_map(self, graph: Data, slot_number: Optional[int] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> bool:

        try:
            if slot_number is None:
                slot_number = self.current_slot
            if not (0 <= slot_number < self.max_slots):
                print_invalid_slot(slot_number)
                log_step("Invalid slot number", slot_number=slot_number)
                return False
            map_data = self._serialize_graph(graph)
            if metadata is None:
                metadata = {}
            metadata.update({
                'save_time': time.time(),
                'save_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'slot_number': slot_number,
                'node_count': len(graph.node_labels) if hasattr(graph, 'node_labels') else 0,
                'edge_count': graph.edge_index.shape[1] if hasattr(graph, 'edge_index') and graph.edge_index is not None else 0
            })
            # Truncate large metadata values to prevent JSON serialization issues
            truncated_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, str) and len(value) > 10000:  # 10KB limit
                    truncated_metadata[key] = value[:10000] + "...[truncated]"
                else:
                    truncated_metadata[key] = value
            map_data['metadata'] = truncated_metadata
            filename = f"neural_map_slot_{slot_number}.json"
            filepath = os.path.join(self.save_directory, filename)
            with open(filepath, 'w') as f:
                json.dump(map_data, f, indent=2)
            self.slot_metadata[slot_number] = metadata
            log_step("Set slot metadata", slot_number=slot_number, key_type=str(type(slot_number)))
            self._save_slot_metadata()
            log_step("Neural map saved",
                    slot_number=slot_number,
                    node_count=metadata['node_count'],
                    edge_count=metadata['edge_count'])
            return True
        except Exception as e:
            logging.error(f"Error saving neural map: {e}", exc_info=True)
            log_step("Error saving neural map", error=str(e), slot_number=slot_number)
            return False
    def load_neural_map(self, slot_number: int) -> Optional[Data]:

        try:
            if not (0 <= slot_number < self.max_slots):
                print_invalid_slot(slot_number)
                log_step("Invalid slot number", slot_number=slot_number)
                return None
            filename = f"neural_map_slot_{slot_number}.json"
            filepath = os.path.join(self.save_directory, filename)
            if not os.path.exists(filepath):
                log_step("Neural map file not found", filepath=filepath)
                return None
            with open(filepath, 'r') as f:
                map_data = json.load(f)
            graph = self._deserialize_graph(map_data)
            if graph is not None:
                metadata = map_data.get('metadata', {})
                log_step("Neural map loaded",
                        slot_number=slot_number,
                        node_count=metadata.get('node_count', 0),
                        edge_count=metadata.get('edge_count', 0))
            return graph
        except Exception as e:
            log_step("Error loading neural map", error=str(e), slot_number=slot_number)
            return None
    def delete_neural_map(self, slot_number: int) -> bool:

        try:
            if not (0 <= slot_number < self.max_slots):
                print_invalid_slot(slot_number)
                log_step("Invalid slot number", slot_number=slot_number)
                return False
            filename = f"neural_map_slot_{slot_number}.json"
            filepath = os.path.join(self.save_directory, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                if slot_number in self.slot_metadata:
                    del self.slot_metadata[slot_number]
                self._save_slot_metadata()
                log_step("Neural map deleted", slot_number=slot_number)
                return True
            else:
                log_step("Neural map file not found for deletion", filepath=filepath)
                return False
        except Exception as e:
            log_step("Error deleting neural map", error=str(e), slot_number=slot_number)
            return False
    def list_available_slots(self) -> Dict[int, Dict[str, Any]]:

        return self.slot_metadata.copy()
    def get_slot_info(self, slot_number: int) -> Optional[Dict[str, Any]]:

        return self.slot_metadata.get(slot_number)
    def set_current_slot(self, slot_number: int) -> bool:

        if 0 <= slot_number < self.max_slots:
            self.current_slot = slot_number
            log_step("Current slot set", slot_number=slot_number)
            return True
        else:
            log_step("Invalid slot number", slot_number=slot_number)
            return False
    def get_current_slot(self) -> int:
        return self.current_slot
    def _serialize_graph(self, graph: Data) -> Dict[str, Any]:

        try:
            serialized_data = {}
            if hasattr(graph, 'node_labels') and graph.node_labels is not None:
                serialized_data['node_labels'] = graph.node_labels
            if hasattr(graph, 'edge_index') and graph.edge_index is not None:
                serialized_data['edge_index'] = graph.edge_index.tolist()
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                serialized_data['edge_attr'] = graph.edge_attr.tolist()
            if hasattr(graph, 'x') and graph.x is not None:
                serialized_data['x'] = graph.x.tolist()
            for attr_name in ['y', 'pos', 'batch']:
                if hasattr(graph, attr_name):
                    attr_value = getattr(graph, attr_name)
                    if attr_value is not None:
                        if isinstance(attr_value, torch.Tensor):
                            serialized_data[attr_name] = attr_value.tolist()
                        else:
                            serialized_data[attr_name] = attr_value
            return serialized_data
        except Exception as e:
            log_step("Error serializing graph", error=str(e))
            return {}
    def _deserialize_graph(self, map_data: Dict[str, Any]) -> Optional[Data]:

        try:
            graph = Data()
            if 'node_labels' in map_data:
                graph.node_labels = map_data['node_labels']
            if 'edge_index' in map_data:
                graph.edge_index = torch.tensor(map_data['edge_index'], dtype=torch.long)
            if 'edge_attr' in map_data:
                graph.edge_attr = torch.tensor(map_data['edge_attr'], dtype=torch.float32)
            if 'x' in map_data:
                graph.x = torch.tensor(map_data['x'], dtype=torch.float32)
            for attr_name in ['y', 'pos', 'batch']:
                if attr_name in map_data:
                    attr_value = map_data[attr_name]
                    if isinstance(attr_value, list):
                        setattr(graph, attr_name, torch.tensor(attr_value))
                    else:
                        setattr(graph, attr_name, attr_value)
            return graph
        except Exception as e:
            log_step("Error deserializing graph", error=str(e))
            return None
    def _load_slot_metadata(self):
        try:
            metadata_file = os.path.join(self.save_directory, "slot_metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    loaded_metadata = json.load(f)
                # Convert string keys back to integers
                self.slot_metadata = {int(k): v for k, v in loaded_metadata.items()}
                log_step("Loaded slot metadata", keys=list(self.slot_metadata.keys()), key_types=[str(type(k)) for k in self.slot_metadata.keys()])
            else:
                self.slot_metadata = {}
        except Exception as e:
            log_step("Error loading slot metadata", error=str(e))
            self.slot_metadata = {}
    def _save_slot_metadata(self):
        try:
            metadata_file = os.path.join(self.save_directory, "slot_metadata.json")
            # Create a copy and truncate large metadata to prevent JSON serialization issues
            truncated_metadata = {}
            for slot, meta in self.slot_metadata.items():
                truncated_meta = meta.copy()
                # Truncate large string values in metadata
                for key, value in truncated_meta.items():
                    if isinstance(value, str) and len(value) > 10000:  # 10KB limit
                        truncated_meta[key] = value[:10000] + "...[truncated]"
                truncated_metadata[slot] = truncated_meta
            with open(metadata_file, 'w') as f:
                json.dump(truncated_metadata, f, indent=2)
        except Exception as e:
            log_step("Error saving slot metadata", error=str(e))
    def get_persistence_statistics(self) -> Dict[str, Any]:
        return {
            'total_slots': self.max_slots,
            'used_slots': len(self.slot_metadata),
            'available_slots': self.max_slots - len(self.slot_metadata),
            'current_slot': self.current_slot,
            'save_directory': self.save_directory
        }
    def cleanup(self):
        self.slot_metadata.clear()
        log_step("NeuralMapPersistence cleanup completed")


def create_neural_map_persistence(save_directory: str = "data/neural_maps") -> NeuralMapPersistence:

    return NeuralMapPersistence(save_directory)
if __name__ == "__main__":
    print("NeuralMapPersistence created successfully!")
    print("Features include:")
    print("- 10-slot save/load system")
    print("- Neural map serialization")
    print("- Metadata tracking")
    print("- Session persistence")
    print("- Slot management")
    try:
        persistence = create_neural_map_persistence()
        stats = persistence.get_persistence_statistics()
        print(f"Persistence statistics: {stats}")
        slots = persistence.list_available_slots()
        print(f"Available slots: {len(slots)}")
    except Exception as e:
        print(f"NeuralMapPersistence test failed: {e}")
    print("NeuralMapPersistence test completed!")







