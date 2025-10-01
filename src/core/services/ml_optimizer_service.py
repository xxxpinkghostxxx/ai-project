"""
MLOptimizerService implementation - Machine Learning-based optimization for neural simulation.

This module provides the concrete implementation of IMLOptimizer,
using machine learning algorithms to optimize neural simulation performance,
predict optimal configurations, and automate parameter tuning.
"""

import random
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from ..interfaces.configuration_service import IConfigurationService
from ..interfaces.event_coordinator import IEventCoordinator
from ..interfaces.ml_optimizer import (IMLOptimizer, OptimizationExperiment,
                                       OptimizationModel)
from ..interfaces.real_time_analytics import IRealTimeAnalytics


class SimpleRegressionModel:
    """Simple regression model for optimization predictions."""

    def __init__(self):
        self.coefficients: Dict[str, float] = {}
        self.intercept = 0.0
        self.feature_names: List[str] = []
        self.trained = False

    def train(self, features: List[List[float]], y: List[float]) -> bool:
        """Train the regression model using simple linear regression."""
        try:
            if len(features) == 0 or len(y) == 0:
                return False

            # Get feature names from first sample
            if isinstance(features[0], dict):
                self.feature_names = list(features[0].keys())
                # Convert dict features to lists
                features_numeric = []
                for sample in features:
                    features_numeric.append([sample.get(feature, 0.0) for feature in self.feature_names])
                features = features_numeric

            n_features = len(features[0])
            n_samples = len(features)

            # Simple linear regression implementation
            # Calculate means
            y_mean = sum(y) / len(y)
            x_means = []
            for i in range(n_features):
                x_means.append(sum(row[i] for row in features) / len(features))

            # Calculate coefficients
            for i in range(n_features):
                numerator = sum((features[j][i] - x_means[i]) * (y[j] - y_mean) for j in range(n_samples))
                denominator = sum((features[j][i] - x_means[i]) ** 2 for j in range(n_samples))

                if denominator != 0:
                    self.coefficients[self.feature_names[i]] = numerator / denominator
                else:
                    self.coefficients[self.feature_names[i]] = 0.0

            # Calculate intercept
            self.intercept = y_mean
            for i, feature in enumerate(self.feature_names):
                self.intercept -= self.coefficients[feature] * x_means[i]

            self.trained = True
            return True

        except (ValueError, TypeError, ZeroDivisionError, IndexError) as e:
            print(f"Error training regression model: {e}")
            return False

    def predict(self, features: Dict[str, float]) -> float:
        """Make prediction using trained model."""
        if not self.trained:
            return 0.0

        prediction = self.intercept
        for feature, value in features.items():
            prediction += self.coefficients.get(feature, 0.0) * value

        return prediction


class BayesianOptimizer:
    """Bayesian optimization for parameter tuning."""

    def __init__(self):
        self.observations: List[Tuple[Dict[str, float], float]] = []
        self.best_observation: Optional[Tuple[Dict[str, float], float]] = None

    def add_observation(self, parameters: Dict[str, float], score: float):
        """Add an observation to the optimization history."""
        self.observations.append((parameters.copy(), score))

        if self.best_observation is None or score > self.best_observation[1]:
            self.best_observation = (parameters.copy(), score)

    def suggest_next_parameters(self, parameter_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Suggest next parameters to try using simple exploration strategy."""
        if not self.observations:
            # Random initial suggestion
            return {
                param: bounds[0] + random.random() * (bounds[1] - bounds[0])
                for param, bounds in parameter_bounds.items()
            }

        # Simple exploration: perturb best parameters
        best_params = self.best_observation[0]
        next_params = {}

        for param, bounds in parameter_bounds.items():
            current_value = best_params.get(param, (bounds[0] + bounds[1]) / 2)
            # Add some noise for exploration
            noise = random.gauss(0, 0.1) * (bounds[1] - bounds[0])
            next_params[param] = max(bounds[0], min(bounds[1], current_value + noise))

        return next_params

    def get_best_parameters(self) -> Optional[Dict[str, float]]:
        """Get the best parameters found so far."""
        return self.best_observation[0] if self.best_observation else None


class MLOptimizerService(IMLOptimizer):
    """
    Concrete implementation of IMLOptimizer.

    This service uses machine learning algorithms to optimize neural simulation
    performance, predict optimal configurations, and automate parameter tuning.
    """

    def __init__(self,
                 configuration_service: IConfigurationService,
                 event_coordinator: IEventCoordinator,
                 analytics_service: IRealTimeAnalytics):
        """
        Initialize the MLOptimizerService.

        Args:
            configuration_service: Service for configuration management
            event_coordinator: Service for event publishing
            analytics_service: Service for real-time analytics
        """
        self.configuration_service = configuration_service
        self.event_coordinator = event_coordinator
        self.analytics_service = analytics_service

        # ML Models
        self.optimization_models: Dict[str, OptimizationModel] = {}
        self.regression_models: Dict[str, SimpleRegressionModel] = {}
        self.bayesian_optimizers: Dict[str, BayesianOptimizer] = {}

        # Experiments
        self.experiments: Dict[str, OptimizationExperiment] = {}
        self.experiment_counter = 0

        # Historical data
        self.performance_history: deque = deque(maxlen=10000)
        self.optimization_history: List[Dict[str, Any]] = []

        # Configuration
        self.min_training_samples = 10
        self.max_optimization_iterations = 50
        self.confidence_threshold = 0.8

    def train_optimization_model(self, historical_data: List[Dict[str, Any]],
                               target_metric: str) -> bool:
        """
        Train an ML model for optimization using historical performance data.

        Args:
            historical_data: Historical performance data for training
            target_metric: The metric to optimize

        Returns:
            bool: True if model training successful
        """
        try:
            if len(historical_data) < self.min_training_samples:
                print(f"Insufficient training data: {len(historical_data)} < {self.min_training_samples}")
                return False

            # Prepare training data
            features_list = []
            y = []

            for data_point in historical_data:
                if target_metric in data_point:
                    features = {k: v for k, v in data_point.items() if k != target_metric and isinstance(v, (int, float))}
                    if features:
                        features_list.append(features)
                        y.append(data_point[target_metric])

            if len(features_list) < self.min_training_samples:
                print(f"Insufficient valid training samples: {len(features_list)}")
                return False

            # Train regression model
            model = SimpleRegressionModel()
            success = model.train(features_list, y)

            if success:
                # Store the trained model
                model_key = f"{target_metric}_optimizer"
                self.regression_models[model_key] = model

                # Create optimization model metadata
                opt_model = OptimizationModel("regression", target_metric)
                opt_model.accuracy = self._calculate_model_accuracy(model, features_list[-10:], y[-10:])  # Test on last 10 samples
                opt_model.training_data_size = len(features_list)
                opt_model.last_trained = time.time()
                opt_model.parameters = {"features": list(features_list[0].keys())}
                opt_model.feature_importance = self._calculate_feature_importance(model)

                self.optimization_models[model_key] = opt_model

                # Publish training completion event
                self.event_coordinator.publish("ml_model_trained", {
                    "model_key": model_key,
                    "target_metric": target_metric,
                    "training_samples": len(features_list),
                    "accuracy": opt_model.accuracy,
                    "timestamp": time.time()
                })

                return True

            return False

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            print(f"Error training optimization model: {e}")
            return False

    def predict_optimal_configuration(self, current_state: Dict[str, Any],
                                    constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Predict the optimal configuration for current system state.

        Args:
            current_state: Current system state and metrics
            constraints: Optional constraints for the optimization

        Returns:
            Dict[str, Any]: Predicted optimal configuration
        """
        try:
            predictions = {}

            # Get numeric features from current state
            features = {k: v for k, v in current_state.items() if isinstance(v, (int, float))}

            # Make predictions for each trained model
            for model_key, model in self.regression_models.items():
                if model.trained:
                    target_metric = model_key.replace("_optimizer", "")
                    predicted_value = model.predict(features)

                    # Apply constraints if provided
                    if constraints and target_metric in constraints:
                        constraint = constraints[target_metric]
                        if "min" in constraint:
                            predicted_value = max(predicted_value, constraint["min"])
                        if "max" in constraint:
                            predicted_value = min(predicted_value, constraint["max"])

                    predictions[target_metric] = {
                        "predicted_value": predicted_value,
                        "confidence": self._estimate_prediction_confidence(features),
                        "model_accuracy": self.optimization_models.get(model_key, OptimizationModel("", "")).accuracy
                    }

            return {
                "current_state": current_state,
                "predictions": predictions,
                "constraints_applied": constraints is not None,
                "timestamp": time.time()
            }

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            print(f"Error predicting optimal configuration: {e}")
            return {"error": str(e)}

    def run_optimization_experiment(self, experiment_config: Dict[str, Any]) -> str:
        """
        Run an optimization experiment to find best parameters.

        Args:
            experiment_config: Configuration for the optimization experiment

        Returns:
            str: Experiment ID for tracking
        """
        try:
            self.experiment_counter += 1
            experiment_id = f"opt_exp_{self.experiment_counter}"

            # Create experiment
            experiment = OptimizationExperiment(
                experiment_id,
                experiment_config.get("optimization_target", "performance")
            )

            self.experiments[experiment_id] = experiment

            # Get parameter bounds
            parameter_bounds = experiment_config.get("parameter_bounds", {})
            if not parameter_bounds:
                # Default bounds for common parameters
                parameter_bounds = {
                    "learning_rate": (0.001, 0.1),
                    "batch_size": (16, 512),
                    "energy_decay_rate": (0.95, 0.999)
                }

            # Initialize Bayesian optimizer
            optimizer_key = f"{experiment_id}_optimizer"
            self.bayesian_optimizers[optimizer_key] = BayesianOptimizer()

            # Run optimization iterations
            max_iterations = min(experiment_config.get("max_iterations", 20), self.max_optimization_iterations)

            for iteration in range(max_iterations):
                # Suggest next parameters
                next_params = self.bayesian_optimizers[optimizer_key].suggest_next_parameters(parameter_bounds)

                # Evaluate parameters (simulate evaluation)
                score = self._evaluate_parameters(next_params, experiment.optimization_target)

                # Add observation
                self.bayesian_optimizers[optimizer_key].add_observation(next_params, score)

                # Record in experiment
                experiment.parameters_tested.append(next_params)
                experiment.results.append({
                    "iteration": iteration,
                    "parameters": next_params,
                    "score": score,
                    "timestamp": time.time()
                })

            # Complete experiment
            experiment.end_time = time.time()
            experiment.status = "completed"
            experiment.best_configuration = self.bayesian_optimizers[optimizer_key].get_best_parameters()

            if experiment.best_configuration:
                # Calculate improvement
                baseline_score = experiment.results[0]["score"] if experiment.results else 0
                best_score = max(result["score"] for result in experiment.results)
                experiment.improvement_percentage = ((best_score - baseline_score) / abs(baseline_score)) * 100 if baseline_score != 0 else 0

            # Publish experiment completion event
            self.event_coordinator.publish("optimization_experiment_completed", {
                "experiment_id": experiment_id,
                "optimization_target": experiment.optimization_target,
                "iterations_completed": len(experiment.results),
                "best_score": max((r["score"] for r in experiment.results), default=0),
                "improvement_percentage": experiment.improvement_percentage,
                "timestamp": time.time()
            })

            return experiment_id

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            print(f"Error running optimization experiment: {e}")
            return ""

    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get the status and results of an optimization experiment.

        Args:
            experiment_id: ID of the experiment

        Returns:
            Dict[str, Any]: Experiment status and results
        """
        try:
            if experiment_id not in self.experiments:
                return {"error": f"Experiment {experiment_id} not found"}

            experiment = self.experiments[experiment_id]
            return experiment.to_dict()

        except (KeyError, AttributeError, TypeError) as e:
            print(f"Error getting experiment status: {e}")
            return {"error": str(e)}

    def apply_ml_optimization(self, optimization_type: str,
                            parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply ML-based optimization to specific system parameters.

        Args:
            optimization_type: Type of optimization ("performance", "energy", "stability")
            parameters: Current parameter values

        Returns:
            Dict[str, Any]: Optimization results and recommendations
        """
        try:
            optimization_results = {
                "optimization_type": optimization_type,
                "original_parameters": parameters.copy(),
                "optimized_parameters": {},
                "expected_improvement": {},
                "confidence": 0.0
            }

            # Get current system state
            current_metrics = self.analytics_service.collect_system_metrics()
            current_state = {metric.name: metric.value for metric in current_metrics}

            # Get target metrics for optimization type
            target_metrics = self._get_target_metrics(optimization_type)

            # Get predictions for target metrics
            predictions = self.predict_optimal_configuration(current_state)

            # Generate optimized parameters
            self._generate_optimized_parameters(
                target_metrics, predictions, parameters, current_state, optimization_results
            )

            return optimization_results

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            print(f"Error applying ML optimization: {e}")
            return {"error": str(e)}

    def _get_target_metrics(self, optimization_type: str) -> List[str]:
        """Get target metrics for optimization type."""
        metric_map = {
            "performance": ["step_time", "cpu_usage"],
            "energy": ["energy_consumption_rate", "energy_efficiency"],
            "stability": ["memory_usage", "connection_stability"]
        }
        return metric_map.get(optimization_type, ["step_time"])

    def _generate_optimized_parameters(self, target_metrics: List[str],
                                     predictions: Dict[str, Any], parameters: Dict[str, Any],
                                     current_state: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Generate optimized parameters based on predictions."""
        for target_metric in target_metrics:
            if target_metric in predictions["predictions"]:
                pred_data = predictions["predictions"][target_metric]

                # Apply parameter adjustments based on metric
                self._adjust_parameters_for_metric(target_metric, pred_data, parameters, current_state, results)

                results["expected_improvement"][target_metric] = pred_data["predicted_value"]
                results["confidence"] = max(results["confidence"], pred_data["confidence"])

    def _adjust_parameters_for_metric(self, target_metric: str, pred_data: Dict[str, Any],
                                    parameters: Dict[str, Any], current_state: Dict[str, Any],
                                    results: Dict[str, Any]) -> None:
        """Adjust parameters based on specific metric predictions."""
        if target_metric == "step_time" and "batch_size" in parameters:
            self._adjust_batch_size(pred_data, parameters, current_state, results)
        elif target_metric == "energy_efficiency" and "energy_decay_rate" in parameters:
            self._adjust_energy_decay_rate(pred_data, results)

    def _adjust_batch_size(self, pred_data: Dict[str, Any], parameters: Dict[str, Any],
                         current_state: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Adjust batch size based on step time prediction."""
        current_batch = parameters["batch_size"]
        if pred_data["predicted_value"] < current_state.get("step_time", 0.1):
            results["optimized_parameters"]["batch_size"] = min(512, int(current_batch * 1.2))
        else:
            results["optimized_parameters"]["batch_size"] = max(16, int(current_batch * 0.8))

    def _adjust_energy_decay_rate(self, pred_data: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Adjust energy decay rate based on efficiency prediction."""
        efficiency = pred_data["predicted_value"]
        if efficiency > 0.8:
            results["optimized_parameters"]["energy_decay_rate"] = 0.995
        elif efficiency < 0.5:
            results["optimized_parameters"]["energy_decay_rate"] = 0.98

    def analyze_optimization_impact(self, before_metrics: Dict[str, Any],
                                  after_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the impact of optimization changes.

        Args:
            before_metrics: Metrics before optimization
            after_metrics: Metrics after optimization

        Returns:
            Dict[str, Any]: Impact analysis results
        """
        try:
            impact_analysis = {
                "before_metrics": before_metrics,
                "after_metrics": after_metrics,
                "improvements": {},
                "degradations": {},
                "overall_impact": "neutral"
            }

            # Calculate changes for each metric
            for metric_name in set(before_metrics.keys()) | set(after_metrics.keys()):
                before_value = before_metrics.get(metric_name, 0)
                after_value = after_metrics.get(metric_name, 0)

                change_percentage = self._calculate_change_percentage(before_value, after_value)

                # Categorize impact if significant
                if abs(change_percentage) > 5:
                    self._categorize_metric_impact(
                        metric_name, change_percentage, before_value, after_value, impact_analysis
                    )

            # Determine overall impact
            self._determine_overall_impact(impact_analysis)

            return impact_analysis

        except (ValueError, TypeError, KeyError, ZeroDivisionError) as e:
            print(f"Error analyzing optimization impact: {e}")
            return {"error": str(e)}

    def _calculate_change_percentage(self, before_value: float, after_value: float) -> float:
        """Calculate percentage change between values."""
        if before_value != 0:
            return ((after_value - before_value) / abs(before_value)) * 100
        return 0

    def _categorize_metric_impact(self, metric_name: str, change_percentage: float,
                                before_value: float, after_value: float,
                                impact_analysis: Dict[str, Any]) -> None:
        """Categorize the impact of a metric change."""
        # Define metric preferences (lower/higher is better)
        lower_better = ["step_time", "cpu_usage", "memory_usage"]
        higher_better = ["energy_efficiency", "learning_progress"]

        if metric_name in lower_better:
            self._assess_lower_better_metric(
                metric_name, change_percentage, before_value, after_value, impact_analysis
            )
        elif metric_name in higher_better:
            self._assess_higher_better_metric(
                metric_name, change_percentage, before_value, after_value, impact_analysis
            )

    def _assess_lower_better_metric(self, metric_name: str, change_percentage: float,
                                  before_value: float, after_value: float,
                                  impact_analysis: Dict[str, Any]) -> None:
        """Assess metrics where lower values are better."""
        if change_percentage < 0:
            impact_analysis["improvements"][metric_name] = {
                "change_percentage": change_percentage,
                "before": before_value,
                "after": after_value
            }
        else:
            impact_analysis["degradations"][metric_name] = {
                "change_percentage": change_percentage,
                "before": before_value,
                "after": after_value
            }

    def _assess_higher_better_metric(self, metric_name: str, change_percentage: float,
                                   before_value: float, after_value: float,
                                   impact_analysis: Dict[str, Any]) -> None:
        """Assess metrics where higher values are better."""
        if change_percentage > 0:
            impact_analysis["improvements"][metric_name] = {
                "change_percentage": change_percentage,
                "before": before_value,
                "after": after_value
            }
        else:
            impact_analysis["degradations"][metric_name] = {
                "change_percentage": change_percentage,
                "before": before_value,
                "after": after_value
            }

    def _determine_overall_impact(self, impact_analysis: Dict[str, Any]) -> None:
        """Determine the overall impact based on improvements vs degradations."""
        num_improvements = len(impact_analysis["improvements"])
        num_degradations = len(impact_analysis["degradations"])

        if num_improvements > num_degradations:
            impact_analysis["overall_impact"] = "positive"
        elif num_degradations > num_improvements:
            impact_analysis["overall_impact"] = "negative"

    def get_optimization_recommendations(self, current_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get ML-driven optimization recommendations.

        Args:
            current_metrics: Current system metrics

        Returns:
            List[Dict[str, Any]]: Optimization recommendations
        """
        try:
            recommendations = []

            # Analyze current performance
            step_time = current_metrics.get("step_time", 0.1)
            cpu_usage = current_metrics.get("cpu_usage", 50)
            memory_usage = current_metrics.get("memory_usage", 0.5)
            energy_efficiency = current_metrics.get("energy_efficiency", 0.7)

            # Generate recommendations based on metrics
            if step_time > 0.05:  # Slow performance
                recommendations.append({
                    "type": "performance_optimization",
                    "priority": "high",
                    "issue": f"High step time: {step_time:.3f}s",
                    "recommendation": "Consider increasing batch size or enabling GPU acceleration",
                    "expected_impact": "20-30% performance improvement",
                    "confidence": 0.85
                })

            if cpu_usage > 80:  # High CPU usage
                recommendations.append({
                    "type": "resource_optimization",
                    "priority": "medium",
                    "issue": f"High CPU usage: {cpu_usage:.1f}%",
                    "recommendation": "Consider reducing neural network complexity or enabling parallel processing",
                    "expected_impact": "15-25% CPU reduction",
                    "confidence": 0.75
                })

            if memory_usage > 0.8:  # High memory usage
                recommendations.append({
                    "type": "memory_optimization",
                    "priority": "high",
                    "issue": f"High memory usage: {memory_usage:.1%}",
                    "recommendation": "Implement memory pooling or reduce batch sizes",
                    "expected_impact": "20-30% memory reduction",
                    "confidence": 0.90
                })

            if energy_efficiency < 0.6:  # Low energy efficiency
                recommendations.append({
                    "type": "energy_optimization",
                    "priority": "medium",
                    "issue": f"Low energy efficiency: {energy_efficiency:.2f}",
                    "recommendation": "Optimize energy distribution and reduce unnecessary computations",
                    "expected_impact": "15-25% energy efficiency improvement",
                    "confidence": 0.80
                })

            # Sort by priority
            priority_order = {"high": 0, "medium": 1, "low": 2}
            recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))

            return recommendations

        except (ValueError, TypeError, KeyError) as e:
            print(f"Error getting optimization recommendations: {e}")
            return []

    def validate_optimization_model(self, validation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate the performance of optimization models.

        Args:
            validation_data: Data for model validation

        Returns:
            Dict[str, Any]: Validation results
        """
        try:
            validation_results = {
                "models_validated": 0,
                "validation_accuracy": {},
                "recommendations": []
            }

            for model_key, model in self.regression_models.items():
                if not model.trained:
                    continue

                target_metric = model_key.replace("_optimizer", "")

                # Prepare validation data
                features_val = []
                y_val = []

                for data_point in validation_data:
                    if target_metric in data_point:
                        features = {k: v for k, v in data_point.items() if k != target_metric and isinstance(v, (int, float))}
                        if features:
                            features_val.append(features)
                            y_val.append(data_point[target_metric])

                if len(features_val) < 5:  # Need minimum validation samples
                    continue

                # Calculate predictions
                predictions = [model.predict(features) for features in features_val]

                # Calculate accuracy metrics
                mse = sum((pred - actual) ** 2 for pred, actual in zip(predictions, y_val)) / len(predictions)
                rmse = mse ** 0.5

                # Calculate R-squared
                y_mean = sum(y_val) / len(y_val)
                ss_tot = sum((actual - y_mean) ** 2 for actual in y_val)
                ss_res = sum((pred - actual) ** 2 for pred, actual in zip(predictions, y_val))
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                validation_results["validation_accuracy"][model_key] = {
                    "rmse": rmse,
                    "r_squared": r_squared,
                    "validation_samples": len(features_val)
                }

                validation_results["models_validated"] += 1

                # Generate recommendations based on validation
                if r_squared < 0.5:
                    validation_results["recommendations"].append({
                        "model": model_key,
                        "issue": f"Poor model performance (R² = {r_squared:.3f})",
                        "recommendation": "Consider retraining with more diverse data or different algorithm"
                    })

            return validation_results

        except (ValueError, TypeError, KeyError, ZeroDivisionError) as e:
            print(f"Error validating optimization model: {e}")
            return {"error": str(e)}

    def _calculate_model_accuracy(self, model: SimpleRegressionModel,
                                x_test: List[Dict[str, float]], y_test: List[float]) -> float:
        """Calculate model accuracy on test data."""
        try:
            if not x_test or not y_test:
                return 0.0

            predictions = [model.predict(features) for features in x_test]

            # Calculate R-squared
            y_mean = sum(y_test) / len(y_test)
            ss_tot = sum((actual - y_mean) ** 2 for actual in y_test)
            ss_res = sum((pred - actual) ** 2 for pred, actual in zip(predictions, y_test))
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            return max(0.0, r_squared)  # Ensure non-negative

        except (ValueError, TypeError, ZeroDivisionError):
            return 0.0

    def _calculate_feature_importance(self, model: SimpleRegressionModel) -> Dict[str, float]:
        """Calculate feature importance based on coefficient magnitudes."""
        try:
            total_importance = sum(abs(coeff) for coeff in model.coefficients.values())

            if total_importance == 0:
                return {feature: 1.0 / len(model.coefficients) for feature in model.coefficients.keys()}

            return {
                feature: abs(coeff) / total_importance
                for feature, coeff in model.coefficients.items()
            }

        except (ValueError, TypeError, ZeroDivisionError):
            return {}

    def _evaluate_parameters(self, parameters: Dict[str, float], optimization_target: str) -> float:
        """Evaluate parameter configuration for optimization."""
        try:
            # Simple evaluation function (would be replaced with actual system evaluation)
            score = 0.0

            if optimization_target == "performance":
                # Performance score based on batch size and learning rate
                batch_size = parameters.get("batch_size", 256)
                learning_rate = parameters.get("learning_rate", 0.01)

                # Larger batch sizes generally improve performance
                batch_score = min(batch_size / 256, 2.0)  # Optimal around 256

                # Learning rate in reasonable range
                if 0.005 <= learning_rate <= 0.05:
                    lr_score = 1.0
                else:
                    lr_score = 0.5

                score = (batch_score + lr_score) / 2

            elif optimization_target == "energy":
                # Energy score based on decay rate
                decay_rate = parameters.get("energy_decay_rate", 0.99)

                # Higher decay rates are more energy efficient
                score = decay_rate * 100  # Convert to 0-100 scale

            else:
                # Generic score
                score = sum(parameters.values()) / len(parameters) * 10

            # Add some randomness to simulate real evaluation variance
            score += random.gauss(0, 0.1)

            return max(0.0, min(100.0, score))  # Clamp to reasonable range

        except (ValueError, TypeError, KeyError):
            return 50.0  # Default score

    def _estimate_prediction_confidence(self, features: Dict[str, float]) -> float:
        """Estimate confidence in model prediction."""
        try:
            # Simple confidence estimation based on feature values
            confidence = 0.5  # Base confidence

            # Higher confidence for features within typical ranges
            for feature, value in features.items():
                if feature == "cpu_usage" and 0 <= value <= 100:
                    confidence += 0.1
                elif feature == "memory_usage" and 0 <= value <= 1:
                    confidence += 0.1
                elif feature == "step_time" and 0 <= value <= 1:
                    confidence += 0.1

            return min(1.0, confidence)

        except (ValueError, TypeError, KeyError):
            return 0.5

    def cleanup(self) -> None:
        """Clean up resources."""
        self.optimization_models.clear()
        self.regression_models.clear()
        self.bayesian_optimizers.clear()
        self.experiments.clear()
        self.optimization_history.clear()
        self.performance_history.clear()






