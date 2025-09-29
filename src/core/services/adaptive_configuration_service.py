"""
AdaptiveConfigurationService implementation - Adaptive configuration for neural simulation.

This module provides the concrete implementation of IAdaptiveConfiguration,
enabling self-tuning of system parameters based on performance metrics and workload characteristics.
"""

import time
from typing import Any, Dict, List, Optional

from ..interfaces.adaptive_configuration import (AdaptationRule,
                                                 ConfigurationParameter,
                                                 IAdaptiveConfiguration)
from ..interfaces.configuration_service import (ConfigurationScope,
                                                IConfigurationService)
from ..interfaces.event_coordinator import IEventCoordinator
from ..interfaces.real_time_analytics import IRealTimeAnalytics


class AdaptiveConfigurationService(IAdaptiveConfiguration):
    """
    Concrete implementation of IAdaptiveConfiguration.

    This service provides dynamic parameter adjustment based on system performance,
    workload characteristics, and environmental factors for optimal neural simulation.
    """

    def __init__(self,
                 configuration_service: IConfigurationService,
                 event_coordinator: IEventCoordinator,
                 analytics_service: IRealTimeAnalytics):
        """
        Initialize the AdaptiveConfigurationService.

        Args:
            configuration_service: Service for configuration management
            event_coordinator: Service for event publishing
            analytics_service: Service for real-time analytics
        """
        self.configuration_service = configuration_service
        self.event_coordinator = event_coordinator
        self.analytics_service = analytics_service

        # Parameter management
        self.parameters: Dict[str, ConfigurationParameter] = {}
        self.adaptation_rules: List[AdaptationRule] = []
        self.configuration_profiles: Dict[str, Dict[str, Any]] = {}

        # Adaptation state
        self.last_adaptation_time = 0.0
        self.adaptation_interval = 30.0  # seconds
        self.max_adaptation_rate = 0.1  # Maximum 10% change per adaptation

        # Performance tracking
        self.adaptation_history: List[Dict[str, Any]] = []

    def register_parameter(self, parameter: ConfigurationParameter) -> bool:
        """
        Register a parameter for adaptive configuration.

        Args:
            parameter: The parameter to register

        Returns:
            bool: True if registration successful
        """
        try:
            if parameter.name in self.parameters:
                print(f"Parameter {parameter.name} already registered")
                return False

            self.parameters[parameter.name] = parameter

            # Publish parameter registration event
            self.event_coordinator.publish("parameter_registered", {
                "parameter_name": parameter.name,
                "current_value": parameter.current_value,
                "adaptation_enabled": parameter.adaptation_enabled,
                "timestamp": time.time()
            })

            return True

        except RuntimeError as e:
            print(f"Error registering parameter: {e}")
            return False

    def add_adaptation_rule(self, rule: AdaptationRule) -> bool:
        """
        Add an adaptation rule for parameter adjustment.

        Args:
            rule: The adaptation rule to add

        Returns:
            bool: True if rule added successfully
        """
        try:
            # Validate rule
            if not self._validate_rule(rule):
                return False

            self.adaptation_rules.append(rule)

            # Sort rules by priority (highest first)
            self.adaptation_rules.sort(key=lambda r: r.priority, reverse=True)

            # Publish rule addition event
            self.event_coordinator.publish("adaptation_rule_added", {
                "parameter_name": rule.parameter_name,
                "condition": rule.condition,
                "action": rule.action,
                "priority": rule.priority,
                "timestamp": time.time()
            })

            return True

        except RuntimeError as e:
            print(f"Error adding adaptation rule: {e}")
            return False

    def evaluate_adaptation_rules(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluate adaptation rules against current metrics.

        Args:
            metrics: Current system metrics

        Returns:
            List[Dict[str, Any]]: Adaptation actions to take
        """
        try:
            actions = []

            for rule in self.adaptation_rules:
                if not rule.enabled:
                    continue

                # Check if condition is met
                if self._evaluate_condition(rule.condition, metrics):
                    # Parse and execute action
                    action_result = self._parse_action(rule.action, metrics)

                    if action_result:
                        actions.append({
                            "rule": rule.to_dict(),
                            "action": action_result,
                            "timestamp": time.time()
                        })

                        # Update rule statistics
                        rule.activation_count += 1
                        rule.last_activated = time.time()

            return actions

        except RuntimeError as e:
            print(f"Error evaluating adaptation rules: {e}")
            return []

    def apply_adaptation(self, parameter_name: str, new_value: Any) -> bool:
        """
        Apply an adaptation to a parameter.

        Args:
            parameter_name: Name of the parameter to adapt
            new_value: New value for the parameter

        Returns:
            bool: True if adaptation applied successfully
        """
        try:
            if parameter_name not in self.parameters:
                print(f"Parameter {parameter_name} not registered")
                return False

            parameter = self.parameters[parameter_name]

            # Validate new value
            if not self._validate_parameter_value(parameter, new_value):
                return False

            # Store old value for history
            old_value = parameter.current_value

            # Apply adaptation with rate limiting
            if parameter.value_type in ["int", "float"]:
                # Smooth adaptation for numeric values
                current_val = float(parameter.current_value)
                target_val = float(new_value)
                max_change = abs(current_val) * self.max_adaptation_rate
                actual_change = target_val - current_val

                if abs(actual_change) > max_change:
                    # Limit the change
                    direction = 1 if actual_change > 0 else -1
                    new_value = current_val + direction * max_change

            # Update parameter
            parameter.current_value = new_value
            parameter.last_adapted = time.time()
            parameter.adaptation_history.append((time.time(), new_value))

            # Update configuration service
            self.configuration_service.set_parameter(
                parameter_name, new_value, ConfigurationScope.GLOBAL
            )

            # Record adaptation
            adaptation_record = {
                "parameter_name": parameter_name,
                "old_value": old_value,
                "new_value": new_value,
                "timestamp": time.time(),
                "reason": "adaptive_configuration"
            }
            self.adaptation_history.append(adaptation_record)

            # Publish adaptation event
            self.event_coordinator.publish("parameter_adapted", adaptation_record)

            return True

        except RuntimeError as e:
            print(f"Error applying adaptation: {e}")
            return False

    def get_optimal_configuration(self, workload_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get optimal configuration for a given workload profile.

        Args:
            workload_profile: Description of the expected workload

        Returns:
            Dict[str, Any]: Optimal configuration parameters
        """
        try:
            optimal_config = {}

            # Analyze workload characteristics
            workload_type = workload_profile.get("type", "balanced")
            expected_load = workload_profile.get("expected_load", 0.5)
            time_constraints = workload_profile.get("time_constraints", "medium")

            # Generate optimal parameters based on workload
            if workload_type == "high_performance":
                optimal_config.update({
                    "batch_size": min(1024, int(512 * (1 + expected_load))),
                    "learning_rate": 0.01 * (1 - expected_load * 0.5),
                    "energy_decay_rate": 0.98 + expected_load * 0.01
                })
            elif workload_type == "energy_efficient":
                optimal_config.update({
                    "batch_size": max(32, int(256 * (1 - expected_load))),
                    "learning_rate": 0.005 * (1 + expected_load * 0.5),
                    "energy_decay_rate": 0.995 - expected_load * 0.005
                })
            elif workload_type == "balanced":
                optimal_config.update({
                    "batch_size": int(256 * (1 + expected_load * 0.5)),
                    "learning_rate": 0.0075,
                    "energy_decay_rate": 0.99
                })

            # Adjust for time constraints
            if time_constraints == "strict":
                optimal_config["batch_size"] = max(32, optimal_config["batch_size"] // 2)
            elif time_constraints == "relaxed":
                optimal_config["batch_size"] = min(2048, optimal_config["batch_size"] * 2)

            return {
                "workload_profile": workload_profile,
                "optimal_configuration": optimal_config,
                "expected_performance_impact": self._estimate_performance_impact(optimal_config),
                "timestamp": time.time()
            }

        except RuntimeError as e:
            print(f"Error getting optimal configuration: {e}")
            return {"error": str(e)}

    def analyze_configuration_impact(self, parameter_changes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the impact of proposed parameter changes.

        Args:
            parameter_changes: Proposed parameter changes

        Returns:
            Dict[str, Any]: Impact analysis results
        """
        try:
            impact_analysis = {
                "parameter_changes": parameter_changes,
                "estimated_impacts": {},
                "risk_assessment": {},
                "recommendations": []
            }

            for param_name, new_value in parameter_changes.items():
                if param_name not in self.parameters:
                    impact_analysis["estimated_impacts"][param_name] = "Unknown parameter"
                    continue

                current_param = self.parameters[param_name]

                # Estimate performance impact
                performance_impact = self._estimate_parameter_impact(
                    param_name, current_param.current_value, new_value
                )
                impact_analysis["estimated_impacts"][param_name] = performance_impact

                # Assess risk
                risk_level = self._assess_adaptation_risk(
                    param_name, current_param.current_value, new_value
                )
                impact_analysis["risk_assessment"][param_name] = risk_level

                # Generate recommendations
                if risk_level == "high":
                    impact_analysis["recommendations"].append({
                        "parameter": param_name,
                        "recommendation": "Consider gradual adaptation or A/B testing",
                        "reason": "High risk of performance degradation"
                    })

            return impact_analysis

        except RuntimeError as e:
            print(f"Error analyzing configuration impact: {e}")
            return {"error": str(e)}

    def create_configuration_profile(self, profile_name: str, parameters: Dict[str, Any]) -> bool:
        """
        Create a named configuration profile.

        Args:
            profile_name: Name of the configuration profile
            parameters: Parameter values for the profile

        Returns:
            bool: True if profile created successfully
        """
        try:
            if profile_name in self.configuration_profiles:
                print(f"Profile {profile_name} already exists")
                return False

            # Validate parameters
            for param_name, value in parameters.items():
                if param_name in self.parameters:
                    param = self.parameters[param_name]
                    if not self._validate_parameter_value(param, value):
                        print(f"Invalid value for parameter {param_name}: {value}")
                        return False

            self.configuration_profiles[profile_name] = {
                "parameters": parameters.copy(),
                "created_at": time.time(),
                "usage_count": 0
            }

            # Publish profile creation event
            self.event_coordinator.publish("configuration_profile_created", {
                "profile_name": profile_name,
                "parameter_count": len(parameters),
                "timestamp": time.time()
            })

            return True

        except RuntimeError as e:
            print(f"Error creating configuration profile: {e}")
            return False

    def load_configuration_profile(self, profile_name: str) -> bool:
        """
        Load a named configuration profile.

        Args:
            profile_name: Name of the configuration profile to load

        Returns:
            bool: True if profile loaded successfully
        """
        try:
            if profile_name not in self.configuration_profiles:
                print(f"Profile {profile_name} not found")
                return False

            profile = self.configuration_profiles[profile_name]

            # Apply all parameters in the profile
            success_count = 0
            for param_name, value in profile["parameters"].items():
                if self.apply_adaptation(param_name, value):
                    success_count += 1

            # Update usage statistics
            profile["usage_count"] += 1
            profile["last_used"] = time.time()

            # Publish profile loading event
            self.event_coordinator.publish("configuration_profile_loaded", {
                "profile_name": profile_name,
                "parameters_applied": success_count,
                "total_parameters": len(profile["parameters"]),
                "timestamp": time.time()
            })

            return success_count == len(profile["parameters"])

        except RuntimeError as e:
            print(f"Error loading configuration profile: {e}")
            return False

    def _validate_rule(self, rule: AdaptationRule) -> bool:
        """Validate an adaptation rule."""
        try:
            # Check if parameter exists
            if rule.parameter_name not in self.parameters:
                print(f"Parameter {rule.parameter_name} not registered")
                return False

            # Validate condition syntax (basic check)
            if not rule.condition:
                print(f"Empty condition: {rule.condition}")
                return False

            # Allow various condition formats
            valid_operators = [">", "<", ">=", "<=", "==", "="]
            has_valid_operator = any(op in rule.condition for op in valid_operators)

            if not has_valid_operator:
                print(f"Invalid condition syntax (no valid operator): {rule.condition}")
                return False

            # Validate action syntax (basic check)
            if not rule.action or "by" not in rule.action:
                if "to" not in rule.action and "set" not in rule.action:
                    print(f"Invalid action syntax: {rule.action}")
                    return False

            return True

        except ValueError as e:
            print(f"Error validating rule: {e}")
            return False

    def _evaluate_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """Evaluate a condition against current metrics."""
        try:
            # Simple condition evaluation (supports basic comparisons)
            if ">" in condition:
                parts = condition.split(">")
                if len(parts) == 2:
                    metric_name = parts[0].strip()
                    threshold = float(parts[1].strip())
                    metric_value = metrics.get(metric_name, 0)
                    return metric_value > threshold

            elif "<" in condition:
                parts = condition.split("<")
                if len(parts) == 2:
                    metric_name = parts[0].strip()
                    threshold = float(parts[1].strip())
                    metric_value = metrics.get(metric_name, 0)
                    return metric_value < threshold

            elif "==" in condition or "=" in condition:
                parts = condition.split("==") if "==" in condition else condition.split("=")
                if len(parts) == 2:
                    metric_name = parts[0].strip()
                    threshold = float(parts[1].strip())
                    metric_value = metrics.get(metric_name, 0)
                    return abs(metric_value - threshold) < 0.01

            return False

        except ValueError as e:
            print(f"Error evaluating condition: {e}")
            return False

    def _parse_action(self, action: str, _metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse and execute an adaptation action."""
        try:
            # Simple action parsing (supports basic operations)
            if "increase" in action and "by" in action:
                parts = action.split("by")
                if len(parts) == 2:
                    param_name = parts[0].replace("increase", "").strip()
                    factor = float(parts[1].strip())
                    if param_name in self.parameters:
                        current_value = self.parameters[param_name].current_value
                        new_value = current_value * (1 + factor)
                        return {
                            "type": "increase",
                            "parameter": param_name,
                            "factor": factor,
                            "new_value": new_value
                        }

            elif "decrease" in action and "by" in action:
                parts = action.split("by")
                if len(parts) == 2:
                    param_name = parts[0].replace("decrease", "").strip()
                    factor = float(parts[1].strip())
                    if param_name in self.parameters:
                        current_value = self.parameters[param_name].current_value
                        new_value = current_value * (1 - factor)
                        return {
                            "type": "decrease",
                            "parameter": param_name,
                            "factor": factor,
                            "new_value": new_value
                        }

            elif "set" in action and "to" in action:
                parts = action.split("to")
                if len(parts) == 2:
                    param_name = parts[0].replace("set", "").strip()
                    new_value = float(parts[1].strip())
                    return {
                        "type": "set",
                        "parameter": param_name,
                        "new_value": new_value
                    }

            return None

        except ValueError as e:
            print(f"Error parsing action: {e}")
            return None

    def _validate_parameter_value(self, parameter: ConfigurationParameter, value: Any) -> bool:
        """Validate a parameter value against constraints."""
        try:
            # Type validation
            if parameter.value_type == "int":
                value = int(value)
            elif parameter.value_type == "float":
                value = float(value)
            elif parameter.value_type == "bool":
                value = bool(value)

            # Range validation
            if parameter.min_value is not None and value < parameter.min_value:
                return False
            if parameter.max_value is not None and value > parameter.max_value:
                return False

            return True

        except (ValueError, TypeError):
            return False

    def _estimate_parameter_impact(
        self, param_name: str, old_value: Any, new_value: Any
    ) -> Dict[str, Any]:
        """Estimate the performance impact of a parameter change."""
        try:
            impact = {"performance_change": 0.0, "energy_change": 0.0, "stability_change": 0.0}

            if param_name == "batch_size":
                # Larger batch sizes generally improve performance but use more memory
                ratio = new_value / old_value if old_value != 0 else 1
                impact["performance_change"] = (ratio - 1) * 0.1  # 10% improvement per 2x increase
                impact["energy_change"] = (ratio - 1) * 0.05  # 5% energy increase per 2x increase

            elif param_name == "learning_rate":
                # Learning rate affects convergence speed and stability
                ratio = new_value / old_value if old_value != 0 else 1
                impact["performance_change"] = (ratio - 1) * 0.2  # Learning speed change
                impact["stability_change"] = (1 - ratio) * 0.3  # Stability inversely related

            elif param_name == "energy_decay_rate":
                # Energy decay affects energy efficiency
                impact["energy_change"] = (new_value - old_value) * 10  # Direct impact on energy

            return impact

        except RuntimeError as e:
            print(f"Error estimating parameter impact: {e}")
            return {"error": str(e)}

    def _estimate_performance_impact(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate overall performance impact of a configuration."""
        try:
            total_performance = 0.0
            total_energy = 0.0
            total_stability = 0.0

            for param_name, value in config.items():
                if param_name in self.parameters:
                    current_value = self.parameters[param_name].current_value
                    impact = self._estimate_parameter_impact(param_name, current_value, value)
                    total_performance += impact.get("performance_change", 0)
                    total_energy += impact.get("energy_change", 0)
                    total_stability += impact.get("stability_change", 0)

            return {
                "performance_impact": total_performance,
                "energy_impact": total_energy,
                "stability_impact": total_stability,
                "overall_score": total_performance - abs(total_energy) - abs(total_stability)
            }

        except RuntimeError as e:
            print(f"Error estimating performance impact: {e}")
            return {"error": str(e)}

    def _assess_adaptation_risk(self, param_name: str, old_value: Any, new_value: Any) -> str:
        """Assess the risk level of a parameter adaptation."""
        try:
            if param_name in ["learning_rate", "energy_decay_rate"]:
                # High-risk parameters
                change_ratio = abs(new_value - old_value) / abs(old_value) if old_value != 0 else 1
                if change_ratio > 0.5:
                    return "high"
                elif change_ratio > 0.2:
                    return "medium"
                else:
                    return "low"

            elif param_name == "batch_size":
                # Medium-risk parameter
                change_ratio = abs(new_value - old_value) / abs(old_value) if old_value != 0 else 1
                if change_ratio > 1.0:
                    return "medium"
                else:
                    return "low"

            else:
                return "low"

        except ValueError:
            return "medium"

    def cleanup(self) -> None:
        """Clean up resources."""
        self.parameters.clear()
        self.adaptation_rules.clear()
        self.configuration_profiles.clear()
        self.adaptation_history.clear()


