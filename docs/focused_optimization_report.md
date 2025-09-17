# Focused Code Optimization Report
==================================================

## Regex Fix
**Count**: 1

- File: .\focused_optimizer.py
  - Description: Simplified redundant None check

## Duplicate Logging
**Count**: 1

- Message: 'Invalid slot number'
  - Occurrences: 4
  - Files: .\neural_map_persistence.py, .\neural_map_persistence.py, .\neural_map_persistence.py, .\neural_map_persistence.py

## Duplicate String
**Count**: 123

- String: ':
    print(...'
  - Files: .\audio_to_neural_bridge.py, .\enhanced_connection_system.py, .\enhanced_neural_dynamics.py, .\enhanced_neural_integration.py, .\enhanced_node_behaviors.py, .\error_handler.py, .\event_driven_system.py, .\live_hebbian_learning.py, .\neural_map_persistence.py, .\sensory_workspace_mapper.py, .\simulation_manager.py, .\spike_queue_system.py, .\visual_energy_bridge.py, .\workspace_engine.py

- String: ')
    print(...'
  - Files: .\audio_to_neural_bridge.py, .\audio_to_neural_bridge.py, .\audio_to_neural_bridge.py, .\audio_to_neural_bridge.py, .\audio_to_neural_bridge.py, .\audio_to_neural_bridge.py, .\audio_to_neural_bridge.py, .\duplicate_code_detector.py, .\duplicate_code_detector.py, .\duplicate_code_detector.py, .\enhanced_connection_system.py, .\enhanced_connection_system.py, .\enhanced_connection_system.py, .\enhanced_connection_system.py, .\enhanced_connection_system.py, .\enhanced_connection_system.py, .\enhanced_connection_system.py, .\enhanced_connection_system.py, .\enhanced_neural_dynamics.py, .\enhanced_neural_dynamics.py, .\enhanced_neural_dynamics.py, .\enhanced_neural_dynamics.py, .\enhanced_neural_dynamics.py, .\enhanced_neural_dynamics.py, .\enhanced_neural_dynamics.py, .\enhanced_neural_dynamics.py, .\enhanced_neural_dynamics.py, .\enhanced_neural_dynamics.py, .\enhanced_neural_dynamics.py, .\enhanced_neural_integration.py, .\enhanced_neural_integration.py, .\enhanced_neural_integration.py, .\enhanced_neural_integration.py, .\enhanced_neural_integration.py, .\enhanced_neural_integration.py, .\enhanced_neural_integration.py, .\enhanced_neural_integration.py, .\enhanced_node_behaviors.py, .\enhanced_node_behaviors.py, .\enhanced_node_behaviors.py, .\enhanced_node_behaviors.py, .\enhanced_node_behaviors.py, .\enhanced_node_behaviors.py, .\enhanced_node_behaviors.py, .\enhanced_node_behaviors.py, .\enhanced_node_behaviors.py, .\error_handler.py, .\error_handler.py, .\error_handler.py, .\error_handler.py, .\error_handler.py, .\error_handler.py, .\error_handler.py, .\event_driven_system.py, .\event_driven_system.py, .\event_driven_system.py, .\event_driven_system.py, .\event_driven_system.py, .\event_driven_system.py, .\event_driven_system.py, .\focused_optimizer.py, .\live_hebbian_learning.py, .\live_hebbian_learning.py, .\live_hebbian_learning.py, .\live_hebbian_learning.py, .\live_hebbian_learning.py, .\live_hebbian_learning.py, .\live_hebbian_learning.py, .\memory_system.py, .\neural_map_persistence.py, .\neural_map_persistence.py, .\neural_map_persistence.py, .\neural_map_persistence.py, .\neural_map_persistence.py, .\neural_map_persistence.py, .\neural_map_persistence.py, .\sensory_workspace_mapper.py, .\sensory_workspace_mapper.py, .\sensory_workspace_mapper.py, .\sensory_workspace_mapper.py, .\sensory_workspace_mapper.py, .\sensory_workspace_mapper.py, .\sensory_workspace_mapper.py, .\sensory_workspace_mapper.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py, .\simulation_manager.py, .\simulation_manager.py, .\simulation_manager.py, .\simulation_manager.py, .\simulation_manager.py, .\simulation_manager.py, .\simulation_manager.py, .\simulation_manager.py, .\spike_queue_system.py, .\spike_queue_system.py, .\spike_queue_system.py, .\spike_queue_system.py, .\spike_queue_system.py, .\spike_queue_system.py, .\spike_queue_system.py, .\unified_test_suite.py, .\unified_test_suite.py, .\verify_nasa_compliance.py, .\verify_nasa_compliance.py, .\visual_energy_bridge.py, .\visual_energy_bridge.py, .\visual_energy_bridge.py, .\visual_energy_bridge.py, .\visual_energy_bridge.py, .\visual_energy_bridge.py, .\visual_energy_bridge.py, .\visual_energy_bridge.py, .\workspace_engine.py, .\workspace_engine.py, .\workspace_engine.py, .\workspace_engine.py, .\workspace_engine.py, .\workspace_engine.py

- String: ')
    except Exception as e:
        print(f...'
  - Files: .\audio_to_neural_bridge.py, .\enhanced_connection_system.py, .\enhanced_neural_dynamics.py, .\enhanced_neural_integration.py, .\enhanced_node_behaviors.py, .\event_driven_system.py, .\live_hebbian_learning.py, .\neural_map_persistence.py, .\sensory_workspace_mapper.py, .\spike_queue_system.py, .\visual_energy_bridge.py, .\workspace_engine.py

- String: ')
            except Exception as e:
             ...'
  - Files: .\behavior_engine.py, .\behavior_engine.py, .\behavior_engine.py, .\behavior_engine.py, .\behavior_engine.py

- String: ', error=str(e), node_id=node_id)
            if se...'
  - Files: .\behavior_engine.py, .\behavior_engine.py

- String: ',
                        recovery_func=lambda: se...'
  - Files: .\behavior_engine.py, .\behavior_engine.py

- String: ', node_id=node_id)
                        access_...'
  - Files: .\behavior_engine.py, .\behavior_engine.py

- String: 'Edge {edge_idx} weight changed from {old_weight:.3...'
  - Files: .\connection_logic.py, .\connection_logic.py

- String: ') or not hasattr(graph, ...'
  - Files: .\connection_logic.py, .\death_and_birth_logic.py

- String: ')
        return False
    except Exception as e:
...'
  - Files: .\death_and_birth_logic.py, .\death_and_birth_logic.py

- String: ')
        or not hasattr(graph, ...'
  - Files: .\death_and_birth_logic.py, .\death_and_birth_logic.py

- String: ',
                ...'
  - Files: .\death_and_birth_logic.py, .\death_and_birth_logic.py, .\death_and_birth_logic.py, .\main_graph.py, .\main_graph.py, .\main_graph.py, .\screen_graph.py, .\screen_graph.py, .\screen_graph.py

- String: ': 0,
                ...'
  - Files: .\death_and_birth_logic.py, .\death_and_birth_logic.py, .\death_and_birth_logic.py, .\main_graph.py, .\main_graph.py, .\screen_graph.py, .\screen_graph.py, .\screen_graph.py

- String: ': 0.0,
                ...'
  - Files: .\death_and_birth_logic.py, .\death_and_birth_logic.py, .\death_and_birth_logic.py, .\main_graph.py, .\main_graph.py, .\main_graph.py, .\main_graph.py, .\screen_graph.py, .\screen_graph.py, .\screen_graph.py

- String: ': True,
                ...'
  - Files: .\death_and_birth_logic.py, .\death_and_birth_logic.py, .\death_and_birth_logic.py, .\main_graph.py, .\screen_graph.py

- String: ': False,
                ...'
  - Files: .\death_and_birth_logic.py, .\screen_graph.py, .\screen_graph.py, .\screen_graph.py

- String: '
    
    def __init__(self, project_dir: str = ...'
  - Files: .\duplicate_code_detector.py, .\focused_optimizer.py, .\simple_duplicate_detector.py

- String: 'Comprehensive analysis of the entire codebase....'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: '
        print(...'
  - Files: .\duplicate_code_detector.py, .\duplicate_code_detector.py, .\duplicate_code_detector.py, .\duplicate_code_detector.py, .\duplicate_code_detector.py, .\duplicate_code_detector.py, .\duplicate_code_detector.py, .\duplicate_code_detector.py, .\duplicate_code_detector.py, .\duplicate_code_detector.py, .\focused_optimizer.py, .\focused_optimizer.py, .\focused_optimizer.py, .\focused_optimizer.py, .\focused_optimizer.py, .\focused_optimizer.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py

- String: 'Load all Python files in the project....'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: 'Find duplicate or very similar functions....'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: 'Find functions that are too dense or complex....'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: ',
                        file_path=file_path,
   ...'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: 'Find redundant or unused imports....'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: ',
                            file_path=file_path,...'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: '
                        ))
            except Exc...'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: 'Find duplicate string literals....'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: ',
                        file_path=file_path,
   ...'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: 'Find similar or duplicate classes....'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: ',
                        file_path=file_path,
   ...'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: 'Create a signature for function comparison....'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: '
        args = [arg.arg for arg in func.args.args...'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: '
    
    def _create_class_signature(self, cls: a...'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: 'Create a signature for class comparison....'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: '
        methods = [node.name for node in cls.body...'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: 'Calculate cyclomatic complexity of a function....'
  - Files: .\duplicate_code_detector.py, .\nasa_code_analyzer.py, .\simple_duplicate_detector.py

- String: 'Generate a comprehensive optimization report....'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: '
        report = [...'
  - Files: .\duplicate_code_detector.py, .\focused_optimizer.py, .\nasa_code_analyzer.py, .\simple_duplicate_detector.py

- String: ']
        
        total_issues = sum(len(patterns...'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: ')
        report.append(...'
  - Files: .\duplicate_code_detector.py, .\duplicate_code_detector.py, .\nasa_code_analyzer.py, .\nasa_code_analyzer.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py

- String: ')
        
        # Summary by category
        f...'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: ')
                report.append(f...'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: ')
                report.append(...'
  - Files: .\duplicate_code_detector.py, .\nasa_code_analyzer.py, .\simple_duplicate_detector.py

- String: ')
                
                # Group by file...'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: ')
                        report.append(f...'
  - Files: .\duplicate_code_detector.py, .\duplicate_code_detector.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py

- String: ')
                        report.append(...'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: ')
                
                report.append(...'
  - Files: .\duplicate_code_detector.py, .\focused_optimizer.py, .\simple_duplicate_detector.py

- String: ')
            report.append(f...'
  - Files: .\duplicate_code_detector.py, .\duplicate_code_detector.py, .\duplicate_code_detector.py, .\focused_optimizer.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py

- String: ')
            report.append(...'
  - Files: .\duplicate_code_detector.py, .\focused_optimizer.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py

- String: ')
        
        return ...'
  - Files: .\duplicate_code_detector.py, .\duplicate_code_detector.py, .\focused_optimizer.py, .\nasa_code_analyzer.py, .\simple_duplicate_detector.py

- String: '.join(report)
    
    def create_optimization_pla...'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: 'Create a step-by-step optimization plan....'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: '
        plan = [...'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: ')
        plan.append(...'
  - Files: .\duplicate_code_detector.py, .\duplicate_code_detector.py, .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: ')
            plan.append(f...'
  - Files: .\duplicate_code_detector.py, .\duplicate_code_detector.py, .\duplicate_code_detector.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py

- String: ')
            plan.append(...'
  - Files: .\duplicate_code_detector.py, .\duplicate_code_detector.py, .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: ')
        
        plan.append(...'
  - Files: .\duplicate_code_detector.py, .\duplicate_code_detector.py

- String: '.join(plan)


def main():
    ...'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: 'Main function to run duplicate code detection....'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: '
    print(...'
  - Files: .\duplicate_code_detector.py, .\focused_optimizer.py, .\simple_duplicate_detector.py, .\verify_nasa_compliance.py, .\verify_nasa_compliance.py, .\verify_nasa_compliance.py, .\verify_nasa_compliance.py, .\verify_nasa_compliance.py, .\verify_nasa_compliance.py, .\verify_nasa_compliance.py, .\verify_nasa_compliance.py, .\verify_nasa_compliance.py, .\verify_nasa_compliance.py, .\verify_nasa_compliance.py

- String: ')
    
    # Generate reports
    report = detecto...'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: ')
    
    return results


if __name__ == ...'
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- String: ',
            ...'
  - Files: .\dynamic_nodes.py, .\dynamic_nodes.py, .\ui_state_manager.py

- String: ': 0.0,
            ...'
  - Files: .\dynamic_nodes.py, .\dynamic_nodes.py, .\dynamic_nodes.py, .\dynamic_nodes.py, .\ui_state_manager.py

- String: ': 0,
            ...'
  - Files: .\dynamic_nodes.py, .\dynamic_nodes.py, .\dynamic_nodes.py, .\ui_state_manager.py

- String: ': True,
            ...'
  - Files: .\dynamic_nodes.py, .\dynamic_nodes.py, .\dynamic_nodes.py

- String: ',
                node_id=node_id,
               ...'
  - Files: .\energy_behavior.py, .\energy_behavior.py

- String: 'IEG activated...'
  - Files: .\enhanced_neural_dynamics.py, .\enhanced_node_behaviors.py

- String: 'Memory consolidated...'
  - Files: .\enhanced_neural_dynamics.py, .\memory_system.py

- String: 'Enhanced connection created...'
  - Files: .\enhanced_neural_integration.py, .\sensory_workspace_mapper.py

- String: 'Error creating enhanced connection...'
  - Files: .\enhanced_neural_integration.py, .\sensory_workspace_mapper.py

- String: 'Error updating node behavior...'
  - Files: .\enhanced_node_behaviors.py, .\enhanced_node_behaviors.py

- String: ')
            except Exception as callback_error:
...'
  - Files: .\error_handler.py, .\simulation_manager.py

- String: ')
    def stop(self):
        self.running = False...'
  - Files: .\event_driven_system.py, .\spike_queue_system.py

- String: ')
        stats = system.get_statistics()
        ...'
  - Files: .\event_driven_system.py, .\spike_queue_system.py

- String: ')
                    report.append(f...'
  - Files: .\focused_optimizer.py, .\focused_optimizer.py, .\focused_optimizer.py, .\focused_optimizer.py, .\focused_optimizer.py, .\focused_optimizer.py, .\focused_optimizer.py, .\nasa_code_analyzer.py

- String: '.join(report)


def main():
    ...'
  - Files: .\focused_optimizer.py, .\nasa_code_analyzer.py

- String: 'Metrics calculation failed, using fallback...'
  - Files: .\homeostasis_controller.py, .\homeostasis_controller.py

- String: 'Energy regulation needed...'
  - Files: .\homeostasis_controller.py, .\homeostasis_controller.py

- String: 'Memory traces formed...'
  - Files: .\learning_engine.py, .\memory_system.py

- String: ': node_id,
                ...'
  - Files: .\main_graph.py, .\screen_graph.py

- String: ': y,
                ...'
  - Files: .\main_graph.py, .\screen_graph.py

- String: ')
            print(f...'
  - Files: .\main_graph.py, .\main_graph.py

- String: ')
    print(f...'
  - Files: .\memory_system.py, .\memory_system.py, .\memory_system.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py, .\unified_test_suite.py, .\unified_test_suite.py, .\unified_test_suite.py, .\unified_test_suite.py, .\unified_test_suite.py

- String: ',
                        line_number=node.lineno,...'
  - Files: .\nasa_code_analyzer.py, .\nasa_code_analyzer.py, .\nasa_code_analyzer.py, .\nasa_code_analyzer.py, .\nasa_code_analyzer.py

- String: ',
                        line_number=i,
         ...'
  - Files: .\nasa_code_analyzer.py, .\nasa_code_analyzer.py

- String: ')
        report.append(f...'
  - Files: .\nasa_code_analyzer.py, .\nasa_code_analyzer.py, .\nasa_code_analyzer.py

- String: 'neural_maps...'
  - Files: .\neural_map_persistence.py, .\neural_map_persistence.py

- String: 'Invalid slot number...'
  - Files: .\neural_map_persistence.py, .\neural_map_persistence.py, .\neural_map_persistence.py, .\neural_map_persistence.py

- String: 'neural_map_slot_{slot_number}.json...'
  - Files: .\neural_map_persistence.py, .\neural_map_persistence.py, .\neural_map_persistence.py

- String: 'slot_metadata.json...'
  - Files: .\neural_map_persistence.py, .\neural_map_persistence.py

- String: ', {
                    ...'
  - Files: .\performance_monitor.py, .\performance_monitor.py, .\performance_monitor.py, .\performance_monitor.py

- String: ': self.current_metrics.memory_usage_mb,
          ...'
  - Files: .\performance_monitor.py, .\performance_monitor.py

- String: ': self.current_metrics.cpu_percent,
              ...'
  - Files: .\performance_monitor.py, .\performance_monitor.py

- String: 'performance...'
  - Files: .\performance_monitor.py, .\performance_monitor.py

- String: 'step_time_ms...'
  - Files: .\performance_monitor.py, .\performance_monitor.py

- String: 'threshold_ms...'
  - Files: .\performance_monitor.py, .\performance_monitor.py

- String: ', {
                        ...'
  - Files: .\performance_monitor.py, .\performance_monitor.py

- String: ': self.current_metrics.fps,
                      ...'
  - Files: .\performance_monitor.py, .\performance_monitor.py

- String: 'error_rate...'
  - Files: .\performance_monitor.py, .\performance_monitor.py, .\performance_monitor.py, .\performance_monitor.py

- String: 'threshold_rate...'
  - Files: .\performance_monitor.py, .\performance_monitor.py

- String: 'Error updating workspace nodes...'
  - Files: .\sensory_workspace_mapper.py, .\workspace_engine.py

- String: ',
                        severity=...'
  - Files: .\simple_duplicate_detector.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py, .\simple_duplicate_detector.py

- String: ',
                            severity=...'
  - Files: .\simple_duplicate_detector.py, .\simple_duplicate_detector.py

- String: ',
                        occurrences=1,
         ...'
  - Files: .\simple_duplicate_detector.py, .\simple_duplicate_detector.py

- String: ')
        
        report.append(...'
  - Files: .\simple_duplicate_detector.py, .\simple_duplicate_detector.py

- String: ')
                fallback = self._get_fallback_co...'
  - Files: .\simulation_manager.py, .\simulation_manager.py

- String: ')
            return True
        except Exception...'
  - Files: .\simulation_manager.py, .\simulation_manager.py

- String: ')
            except Exception as e:
             ...'
  - Files: .\simulation_manager.py, .\simulation_manager.py, .\simulation_manager.py, .\simulation_manager.py, .\simulation_manager.py

- String: '
                   f...'
  - Files: .\simulation_manager.py, .\simulation_manager.py

- String: '
        return ...'
  - Files: .\simulation_manager.py, .\simulation_manager.py, .\simulation_manager.py, .\simulation_manager.py, .\simulation_manager.py, .\simulation_manager.py, .\simulation_manager.py, .\simulation_manager.py, .\simulation_manager.py, .\simulation_manager.py, .\simulation_manager.py, .\simulation_manager.py, .\simulation_manager.py, .\simulation_manager.py, .\simulation_manager.py, .\simulation_manager.py, .\simulation_manager.py

- String: 'Neural Simulation System...'
  - Files: .\ui_engine.py, .\ui_engine.py

- String: 'nodes_text...'
  - Files: .\ui_engine.py, .\ui_engine.py

- String: 'edges_text...'
  - Files: .\ui_engine.py, .\ui_engine.py

- String: 'energy_text...'
  - Files: .\ui_engine.py, .\ui_engine.py

- String: 'connections_text...'
  - Files: .\ui_engine.py, .\ui_engine.py

- String: ')
        dpg.add_text(...'
  - Files: .\ui_engine.py, .\ui_engine.py, .\ui_engine.py, .\ui_engine.py, .\ui_engine.py, .\ui_engine.py, .\ui_engine.py, .\ui_engine.py, .\ui_engine.py, .\ui_engine.py, .\ui_engine.py

- String: ')
        print(f...'
  - Files: .\ui_engine.py, .\visual_energy_bridge.py

- String: ')
        print(...'
  - Files: .\unified_launcher.py, .\unified_launcher.py, .\unified_launcher.py, .\unified_launcher.py, .\unified_launcher.py, .\unified_launcher.py, .\unified_launcher.py, .\unified_launcher.py, .\unified_launcher.py

- String: '
        print(f...'
  - Files: .\unified_test_suite.py, .\verify_nasa_compliance.py

- String: ')
        return False
    else:
        print(...'
  - Files: .\verify_nasa_compliance.py, .\verify_nasa_compliance.py, .\verify_nasa_compliance.py, .\verify_nasa_compliance.py, .\verify_nasa_compliance.py

- String: ')
    else:
        print(...'
  - Files: .\verify_nasa_compliance.py, .\verify_nasa_compliance.py

- String: ')
        return True
    else:
        print(f...'
  - Files: .\verify_nasa_compliance.py, .\verify_nasa_compliance.py

## Duplicate Function
**Count**: 39

- Function: __init__
  - Files: .\audio_to_neural_bridge.py, .\behavior_engine.py, .\config_manager.py, .\connection_logic.py, .\duplicate_code_detector.py, .\enhanced_connection_system.py, .\enhanced_connection_system.py, .\enhanced_neural_dynamics.py, .\enhanced_neural_integration.py, .\enhanced_node_behaviors.py, .\enhanced_node_behaviors.py, .\error_handler.py, .\event_driven_system.py, .\event_driven_system.py, .\event_driven_system.py, .\focused_optimizer.py, .\homeostasis_controller.py, .\learning_engine.py, .\live_hebbian_learning.py, .\logging_utils.py, .\memory_system.py, .\nasa_code_analyzer.py, .\network_metrics.py, .\neural_map_persistence.py, .\node_access_layer.py, .\node_id_manager.py, .\performance_monitor.py, .\random_seed_manager.py, .\simple_duplicate_detector.py, .\simulation_manager.py, .\spike_queue_system.py, .\spike_queue_system.py, .\spike_queue_system.py, .\static_allocator.py, .\ui_state_manager.py, .\unified_launcher.py, .\unified_test_suite.py, .\unified_test_suite.py, .\visual_energy_bridge.py, .\workspace_engine.py

- Function: set_neuromodulator_level
  - Files: .\behavior_engine.py, .\enhanced_connection_system.py, .\enhanced_neural_dynamics.py, .\enhanced_neural_integration.py, .\simulation_manager.py

- Function: _load_config
  - Files: .\config_manager.py, .\simulation_manager.py

- Function: _create_default_config
  - Files: .\config_manager.py, .\simulation_manager.py

- Function: _precache_frequent_sections
  - Files: .\config_manager.py, .\simulation_manager.py

- Function: update_eligibility_trace
  - Files: .\connection_logic.py, .\enhanced_connection_system.py

- Function: record_activation
  - Files: .\connection_logic.py, .\enhanced_connection_system.py

- Function: _load_python_files
  - Files: .\duplicate_code_detector.py, .\simple_duplicate_detector.py

- Function: main
  - Files: .\duplicate_code_detector.py, .\focused_optimizer.py, .\nasa_code_analyzer.py, .\simple_duplicate_detector.py, .\ui_engine.py, .\unified_launcher.py

- Function: reset_statistics
  - Files: .\enhanced_connection_system.py, .\enhanced_neural_dynamics.py, .\enhanced_node_behaviors.py, .\event_driven_system.py, .\event_driven_system.py, .\homeostasis_controller.py, .\learning_engine.py, .\memory_system.py, .\spike_queue_system.py, .\spike_queue_system.py, .\spike_queue_system.py, .\workspace_engine.py

- Function: cleanup
  - Files: .\enhanced_connection_system.py, .\enhanced_neural_dynamics.py, .\enhanced_neural_integration.py, .\enhanced_node_behaviors.py, .\live_hebbian_learning.py, .\neural_map_persistence.py, .\simulation_manager.py, .\ui_state_manager.py, .\workspace_engine.py

- Function: _update_theta_burst_counter
  - Files: .\enhanced_neural_dynamics.py, .\enhanced_node_behaviors.py

- Function: add_error_callback
  - Files: .\error_handler.py, .\simulation_manager.py

- Function: wrapper
  - Files: .\error_handler.py, .\logging_utils.py, .\simulation_manager.py

- Function: __lt__
  - Files: .\event_driven_system.py, .\spike_queue_system.py

- Function: clear
  - Files: .\event_driven_system.py, .\spike_queue_system.py

- Function: start
  - Files: .\event_driven_system.py, .\spike_queue_system.py

- Function: stop
  - Files: .\event_driven_system.py, .\spike_queue_system.py

- Function: schedule_spike
  - Files: .\event_driven_system.py, .\simulation_manager.py

- Function: form_memory_traces
  - Files: .\learning_engine.py, .\memory_system.py

- Function: _has_stable_pattern
  - Files: .\learning_engine.py, .\memory_system.py

- Function: _create_memory_trace
  - Files: .\learning_engine.py, .\memory_system.py

- Function: get_memory_trace_count
  - Files: .\learning_engine.py, .\memory_system.py

- Function: append_log_line
  - Files: .\logging_utils.py, .\simulation_manager.py

- Function: log_runtime
  - Files: .\logging_utils.py, .\simulation_manager.py

- Function: increment_seed
  - Files: .\random_seed_manager.py, .\random_seed_manager.py

- Function: _create_enhanced_connection
  - Files: .\sensory_workspace_mapper.py, .\simulation_manager.py

- Function: reset_simulation
  - Files: .\simulation_manager.py, .\ui_engine.py

- Function: set_visual_sensitivity
  - Files: .\simulation_manager.py, .\visual_energy_bridge.py

- Function: set_pattern_threshold
  - Files: .\simulation_manager.py, .\visual_energy_bridge.py

- Function: _clear_graph_references
  - Files: .\simulation_manager.py, .\ui_state_manager.py

- Function: get_simulation_running
  - Files: .\ui_engine.py, .\ui_state_manager.py

- Function: set_simulation_running
  - Files: .\ui_engine.py, .\ui_state_manager.py, .\ui_state_manager.py

- Function: get_latest_graph
  - Files: .\ui_engine.py, .\ui_state_manager.py, .\ui_state_manager.py

- Function: get_latest_graph_for_ui
  - Files: .\ui_engine.py, .\ui_state_manager.py, .\ui_state_manager.py

- Function: update_graph
  - Files: .\ui_engine.py, .\ui_state_manager.py, .\ui_state_manager.py

- Function: add_live_feed_data
  - Files: .\ui_engine.py, .\ui_state_manager.py, .\ui_state_manager.py

- Function: get_live_feed_data
  - Files: .\ui_engine.py, .\ui_state_manager.py

- Function: clear_live_feed_data
  - Files: .\ui_engine.py, .\ui_state_manager.py, .\ui_state_manager.py
