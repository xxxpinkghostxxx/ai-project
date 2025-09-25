# Refactoring Plan: From Monolith to Service-Oriented Architecture

## 1. Executive Summary

This document outlines a comprehensive, phased refactoring plan to transition the application from its current dual-architecture state to a singular, consistent Service-Oriented Architecture (SOA). The primary objective is to eliminate the monolithic `SimulationManager` god object and fully adopt the modern, testable, and maintainable service-based architecture that already exists within the codebase but is currently only utilized in tests. This refactoring is critical for the long-term health, scalability, and stability of the platform.

## 2. Phased Refactoring Approach

The transition will be executed in three distinct phases to minimize disruption and allow for thorough testing at each stage.

*   **Phase 1: Composition Root and Service Foundation.** Establish a single entry point for the application that initializes the SOA, and begin migrating non-critical responsibilities from `SimulationManager`.
*   **Phase 2: Core Logic Migration.** Migrate the primary simulation and business logic from `SimulationManager` to their respective services. This is the most complex phase.
*   **Phase 3: Cleanup and Finalization.** Remove the `SimulationManager` and all obsolete components, and refactor any remaining code that still has legacy dependencies.

## 3. Component Mapping: `SimulationManager` to SOA Services

The responsibilities of the `SimulationManager` will be re-assigned to the following dedicated services, as defined in `core/interfaces` and implemented in `core/services`.

| `SimulationManager` Responsibility | Target Service | Service Interface |
| ---------------------------------- | ------------------------------------- | ----------------------------- |
| Simulation Lifecycle (start, stop) | `SimulationCoordinator` | `ISimulationCoordinator` |
| Simulation Step Logic | `SimulationCoordinator` | `ISimulationCoordinator` |
| Component Initialization | `ServiceRegistry` | `IServiceRegistry` |
| Configuration Loading | `ConfigurationService` | `IConfigurationService` |
| Performance Monitoring | `PerformanceMonitoringService` | `IPerformanceMonitor` |
| UI/Visualization Updates | `RealTimeVisualizationService` | `IRealTimeVisualization` |
| Sensory Input Processing | `SensoryProcessingService` | `ISensoryProcessor` |
| Neural Processing | `NeuralProcessingService` | `INeuralProcessor` |
| Energy Management | `EnergyManagementService` | `IEnergyManager` |
| Learning & Plasticity | `LearningService` | `ILearningEngine` |
| Graph/Network Management | `GraphManagementService` | `IGraphManager` |
| Event Coordination | `EventCoordinationService` | `IEventCoordinator` |

## 4. Detailed Refactoring Steps

### Phase 1: Composition Root and Service Foundation

1.  **Modify the Entry Point:**
    *   Update `core/unified_launcher.py` to act as the **Composition Root**.
    *   Instantiate the `ServiceRegistry` at the beginning of the main execution block.
    *   Register all core services (`ConfigurationService`, `PerformanceMonitoringService`, etc.) with the `ServiceRegistry`.

2.  **Migrate Configuration Loading:**
    *   Move all configuration file parsing logic from `SimulationManager` into `ConfigurationService`.
    *   Refactor `unified_launcher.py` to use `ConfigurationService` (retrieved from the `ServiceRegistry`) to load settings.

3.  **Consolidate Performance Monitoring:**
    *   Identify all code using the legacy `utils/performance_monitor.py`.
    *   Refactor these sections to use the `PerformanceMonitoringService`. Note: This service should wrap the more capable `utils/unified_performance_system.py`.
    *   Delete the redundant `utils/performance_monitor.py`.

### Phase 2: Core Logic Migration

1.  **Refactor `LearningEngine`:**
    *   Modify the constructor of `learning/learning_engine.py` to accept its dependencies (`INeuralProcessor`, `IEnergyManager`, etc.) via **constructor injection**.
    *   Remove the Service Locator pattern (`get_simulation_manager()`) and the lazy-loading mechanism.
    *   Update the `ServiceRegistry` in `unified_launcher.py` to correctly instantiate and inject dependencies into the new `LearningService`.

2.  **Implement the Simulation Loop in `SimulationCoordinator`:**
    *   Extract the main `_step()` method and simulation loop logic from `SimulationManager`.
    *   Implement this logic within the `SimulationCoordinator` service.
    *   The `SimulationCoordinator` will resolve its dependencies (e.g., `INeuralProcessor`, `ILearningEngine`) from the `ServiceRegistry` to orchestrate the simulation steps.

3.  **Migrate Remaining Responsibilities:**
    *   Systematically move all remaining logic (sensory processing, UI updates, etc.) from `SimulationManager` methods to the corresponding services as mapped in Section 3.
    *   Each service should be self-contained and only interact with other services through their interfaces, orchestrated by the `SimulationCoordinator`.

### Phase 3: Cleanup and Finalization

1.  **Update the UI Engine:**
    *   Refactor `ui/ui_engine.py` to interact with the new services (`RealTimeVisualizationService`, `ConfigurationService`) instead of `SimulationManager`.
    *   It will receive service instances from the `unified_launcher.py`.

2.  **Remove the God Object:**
    *   Globally search for any remaining usages of `get_simulation_manager()` or direct instantiations of `SimulationManager`. Refactor these to use dependency injection.
    *   Once all responsibilities are migrated and no references remain, **delete the file `core/simulation_manager.py`**.

3.  **Final Code Review:**
    *   Perform a final review of the codebase to ensure all legacy code paths have been removed and the application exclusively uses the SOA pattern.
    *   Ensure all tests are passing with the new architecture.

## 5. Testing Strategy

1.  **Unit Test Augmentation:** For each service that inherits logic from `SimulationManager`, new unit tests will be written to ensure its functionality is correct in isolation. Mocking will be used heavily to isolate service dependencies.
2.  **Preserve Existing Integration Tests:** The existing test suite, which already uses the SOA, will serve as the primary validation mechanism. As `SimulationManager` is phased out, these tests will become the official integration tests for the application.
3.  **Full-System Regression Testing:** After each phase, a full regression test will be executed by running the main application (`unified_launcher.py`) and verifying key outputs and behaviors. This ensures the refactored application remains functionally equivalent to the original.
4.  **Performance Benchmarking:** After the refactoring is complete, run the performance benchmarks to ensure the new architecture has not introduced any significant performance regressions.

## 6. Risk Analysis and Mitigation

| Risk | Likelihood | Impact | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Introduction of Bugs** | High | High | The phased approach with comprehensive regression testing after each phase is designed to catch bugs early. The existing SOA test suite provides a robust safety net. |
| **Performance Degradation** | Medium | Medium | Service instantiation overhead might introduce minor latency. This will be monitored via performance benchmarking tests. Caching strategies can be implemented in the `ServiceRegistry` if needed. |
| **Extended Development Time** | Medium | High | The complexity of untangling dependencies in `SimulationManager` might take longer than estimated. We will mitigate this by ensuring the component mapping is precise and by prioritizing the most critical logic first. |
| **Hidden Dependencies** | High | Medium | The global singleton pattern may hide dependencies. A thorough, tool-assisted search for all usages of `get_simulation_manager()` will be required before its final deletion. |
