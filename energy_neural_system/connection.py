import time
import numpy as np
from utils import logger

class Connection:
    def __init__(self, source, destination, weight=0.1, energy_transfer_capacity=0.3):
        self.source = source
        self.destination = destination
        self.weight = weight
        self.activity = 0
        self.formation_time = time.time()
        self.activity_history = []  # (timestamp, activity)
        self.energy_transfer_capacity = energy_transfer_capacity
        self.last_activity = 0.0
        self.last_used_tick = 0
        self.created_tick = int(time.time())

    def update(self, maintenance_cost=0.01):
        """
        Update connection activity and transfer energy if information is flowing.
        Activity is based on the difference in 'value' channels.
        """
        src_val = self.source.channels.get('value', 0)
        dst_val = self.destination.channels.get('value', 0)
        # Ensure src_val and dst_val are scalars
        if isinstance(src_val, (np.ndarray, list, tuple)):
            src_val = float(np.mean(src_val))
        if isinstance(dst_val, (np.ndarray, list, tuple)):
            dst_val = float(np.mean(dst_val))
        activity = abs(src_val - dst_val) * abs(self.weight)
        self.activity = activity
        self.last_activity = activity
        self.activity_history.append((time.time(), activity))
        if len(self.activity_history) > 100:
            self.activity_history = self.activity_history[-100:]
        # Transfer energy if activity is high
        if activity > 0.1:
            # Add ±5% randomizer to the transfer amount
            base_transfer = self.energy_transfer_capacity * activity
            max_transfer = min(base_transfer, self.source.energy)
            # Randomize within ±5%
            random_factor = 1.0 + np.random.uniform(-0.05, 0.05)
            transfer = max_transfer * random_factor
            transfer = max(0.0, min(transfer, self.source.energy))
            print(f"[DEBUG] Connection {id(self)}: src_val={src_val}, dst_val={dst_val}, activity={activity}, transfer={transfer}")
            self.source.energy -= transfer
            self.destination.energy += transfer
        # Decay unused connections
        current_tick = int(time.time())
        if self.last_used_tick < current_tick - 100:
            self.weight *= 0.99
        # Adapt weight (example: Hebbian or reward-based)
        # For demonstration, nudge toward 0.5 if used
        learning_rate = 0.05
        if hasattr(self, 'used_this_tick') and self.used_this_tick:
            target = 0.5
            self.weight += learning_rate * (target - self.weight)
        # Apply maintenance cost
        self.source.energy -= maintenance_cost
        self.destination.energy -= maintenance_cost
        self.last_used_tick = current_tick 