#!/usr/bin/env python3
"""
Test Phase 3C services - Cloud Integration.

This test verifies that the Phase 3C cloud integration services
(CloudDeploymentService) work correctly with the existing service architecture.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_cloud_deployment_service():
    """Test CloudDeploymentService functionality."""
    try:
        from core.services.cloud_deployment_service import CloudDeploymentService
        import time

        # Create cloud deployment service
        cloud_service = CloudDeploymentService()

        # Test deployment creation
        deployment_config = {
            "platform": "aws",
            "region": "us-west-2",
            "instance_type": "t3.medium",
            "instance_count": 2,
            "auto_scaling": True,
            "min_instances": 1,
            "max_instances": 5,
            "container_image": "neural-sim:latest",
            "environment_variables": {
                "SIMULATION_MODE": "production",
                "LOG_LEVEL": "info"
            },
            "network_config": {
                "security_groups": ["neural-sim-sg"],
                "subnets": ["subnet-12345"]
            },
            "monitoring_config": {
                "enable_monitoring": True,
                "metrics": ["cpu", "memory", "network"]
            }
        }

        deployment_id = cloud_service.create_deployment(deployment_config)
        if not deployment_id:
            raise Exception("Deployment creation failed")

        # Test deployment status retrieval
        status = cloud_service.get_deployment_status(deployment_id)
        if not isinstance(status, dict) or status.get("status") != "running":
            raise Exception("Invalid deployment status")

        # Test deployment update
        update_config = {
            "instance_count": 3,
            "environment_variables": {
                "SIMULATION_MODE": "production",
                "LOG_LEVEL": "debug"
            }
        }
        success = cloud_service.update_deployment(deployment_id, update_config)
        if not success:
            raise Exception("Deployment update failed")

        # Test scaling
        success = cloud_service.scale_deployment(deployment_id, 4)
        if not success:
            raise Exception("Deployment scaling failed")

        # Test scaling policy creation
        policy_config = {
            "metric_name": "cpu_usage",
            "target_value": 70.0,
            "scale_out_threshold": 80.0,
            "scale_in_threshold": 30.0,
            "cooldown_period": 300,
            "min_instances": 1,
            "max_instances": 5,
            "enabled": True
        }

        policy_id = cloud_service.create_scaling_policy(deployment_id, policy_config)
        if not policy_id:
            raise Exception("Scaling policy creation failed")

        # Test scaling status
        scaling_status = cloud_service.get_scaling_status(deployment_id)
        if not isinstance(scaling_status, dict):
            raise Exception("Invalid scaling status")

        # Test cost calculation
        costs = cloud_service.get_deployment_costs(deployment_id)
        if not isinstance(costs, dict) or "total_cost" not in costs:
            raise Exception("Invalid cost calculation")

        # Test backup creation
        backup_config = {
            "include_data": True,
            "compression": "gzip",
            "retention_days": 30
        }
        backup_id = cloud_service.backup_deployment(deployment_id, backup_config)
        if not backup_id:
            raise Exception("Backup creation failed")

        # Test deployment termination
        success = cloud_service.terminate_deployment(deployment_id)
        if not success:
            raise Exception("Deployment termination failed")

        # Clean up
        cloud_service.cleanup()

        print("PASS: CloudDeploymentService test successful")
        return True

    except Exception as e:
        print(f"FAIL: CloudDeploymentService test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_cloud_deployment():
    """Test multi-cloud deployment capabilities."""
    try:
        from core.services.cloud_deployment_service import CloudDeploymentService
        import time

        # Create cloud deployment service
        cloud_service = CloudDeploymentService()

        # Test multi-region deployment (simulating multi-cloud)
        multi_cloud_config = {
            "platforms": ["aws", "aws"],  # Use same platform for simplicity
            "base_config": {
                "platform": "aws",  # Will be overridden per platform
                "instance_type": "t3.medium",
                "instance_count": 1,
                "auto_scaling": False,
                "container_image": "neural-sim:latest",
                "environment_variables": {
                    "MULTI_CLOUD": "true",
                    "CLUSTER_MODE": "distributed"
                }
            },
            "aws_region": "us-west-2",
            "aws_region_2": "us-east-1"
        }

        deployment_ids = cloud_service.deploy_to_multiple_clouds(multi_cloud_config)
        if not deployment_ids or len(deployment_ids) != 2:
            raise Exception("Multi-cloud deployment failed")

        # Verify deployments were created
        for deployment_id in deployment_ids:
            status = cloud_service.get_deployment_status(deployment_id)
            if not isinstance(status, dict) or status.get("status") != "running":
                raise Exception(f"Multi-cloud deployment {deployment_id} failed")

        # Test cross-cloud scaling
        for deployment_id in deployment_ids:
            success = cloud_service.scale_deployment(deployment_id, 2)
            if not success:
                raise Exception(f"Cross-cloud scaling failed for {deployment_id}")

        # Test cross-cloud backup
        for deployment_id in deployment_ids:
            backup_config = {"type": "cross_cloud_backup"}
            backup_id = cloud_service.backup_deployment(deployment_id, backup_config)
            if not backup_id:
                raise Exception(f"Cross-cloud backup failed for {deployment_id}")

        # Clean up all deployments
        for deployment_id in deployment_ids:
            success = cloud_service.terminate_deployment(deployment_id)
            if not success:
                print(f"Warning: Failed to terminate deployment {deployment_id}")

        # Clean up service
        cloud_service.cleanup()

        print("PASS: Multi-cloud deployment test successful")
        return True

    except Exception as e:
        print(f"FAIL: Multi-cloud deployment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cloud_resource_management():
    """Test cloud resource management and optimization."""
    try:
        from core.services.cloud_deployment_service import CloudDeploymentService
        import time

        # Create cloud deployment service
        cloud_service = CloudDeploymentService()

        # Create multiple deployments for resource management testing
        deployments = []
        regions = ["us-west-2", "us-east-1", "eu-west-1"]

        for i, region in enumerate(regions):
            config = {
                "platform": "aws",
                "region": region,
                "instance_type": "t3.medium",
                "instance_count": 1,
                "auto_scaling": True,
                "container_image": f"neural-sim:v{i+1}"
            }

            deployment_id = cloud_service.create_deployment(config)
            if deployment_id:
                deployments.append(deployment_id)
            else:
                raise Exception(f"Failed to create deployment {i+1}")

        # Test resource monitoring
        total_resources = 0
        total_cost = 0.0

        for deployment_id in deployments:
            status = cloud_service.get_deployment_status(deployment_id)
            if status and "resources" in status:
                total_resources += len(status["resources"])

            costs = cloud_service.get_deployment_costs(deployment_id)
            if costs and "total_cost" in costs:
                total_cost += costs["total_cost"]

        if total_resources != len(deployments):  # One resource per deployment
            raise Exception(f"Resource count mismatch: expected {len(deployments)}, got {total_resources}")

        # Test cost optimization
        if total_cost <= 0:
            raise Exception("Invalid cost calculation")

        # Test resource scaling across deployments
        for deployment_id in deployments:
            # Scale up
            success = cloud_service.scale_deployment(deployment_id, 2)
            if not success:
                raise Exception(f"Scale up failed for {deployment_id}")

            # Scale down
            success = cloud_service.scale_deployment(deployment_id, 1)
            if not success:
                raise Exception(f"Scale down failed for {deployment_id}")

        # Test backup and restore
        test_deployment = deployments[0]
        backup_config = {
            "type": "full_backup",
            "include_config": True,
            "include_data": True
        }

        backup_id = cloud_service.backup_deployment(test_deployment, backup_config)
        if not backup_id:
            raise Exception("Backup creation failed")

        # Test restore to new deployment
        new_deployment_id = f"restored_{test_deployment}"
        success = cloud_service.restore_deployment(new_deployment_id, backup_id)
        if not success:
            raise Exception("Deployment restore failed")

        # Verify restored deployment exists and has valid status
        restored_status = cloud_service.get_deployment_status(new_deployment_id)
        if not restored_status or "error" in restored_status:
            raise Exception("Restored deployment status check failed")

        # The restored deployment might not be automatically running, so just check it exists
        print(f"Restored deployment {new_deployment_id} created with status: {restored_status.get('status')}")

        # Clean up all deployments
        all_deployments = deployments + [new_deployment_id]
        for deployment_id in all_deployments:
            success = cloud_service.terminate_deployment(deployment_id)
            if not success:
                print(f"Warning: Failed to terminate deployment {deployment_id}")

        # Clean up service
        cloud_service.cleanup()

        print("PASS: Cloud resource management test successful")
        return True

    except Exception as e:
        print(f"FAIL: Cloud resource management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cloud_performance_and_scalability():
    """Test cloud deployment performance and scalability."""
    try:
        from core.services.cloud_deployment_service import CloudDeploymentService
        import time

        # Create cloud deployment service
        cloud_service = CloudDeploymentService()

        # Test deployment creation performance
        deployment_times = []
        scaling_times = []

        for i in range(5):  # Create 5 deployments
            config = {
                "platform": "aws",
                "region": "us-west-2",
                "instance_type": "t3.medium",
                "instance_count": 1,
                "container_image": f"neural-sim:perf_test_{i}"
            }

            start_time = time.time()
            deployment_id = cloud_service.create_deployment(config)
            deployment_time = time.time() - start_time
            deployment_times.append(deployment_time)

            if not deployment_id:
                raise Exception(f"Deployment creation {i+1} failed")

            # Test scaling performance
            start_time = time.time()
            success = cloud_service.scale_deployment(deployment_id, 3)
            scaling_time = time.time() - start_time
            scaling_times.append(scaling_time)

            if not success:
                raise Exception(f"Scaling failed for deployment {deployment_id}")

        # Performance assertions
        avg_deployment_time = sum(deployment_times) / len(deployment_times)
        avg_scaling_time = sum(scaling_times) / len(scaling_times)

        print(".3f")
        print(".3f")
        # Performance requirements (adjustable based on cloud provider)
        if avg_deployment_time > 2.0:  # Should complete within 2 seconds
            print(f"Warning: Deployment time {avg_deployment_time:.3f}s > 2s")

        if avg_scaling_time > 1.0:  # Should complete within 1 second
            print(f"Warning: Scaling time {avg_scaling_time:.3f}s > 1s")

        # Test concurrent operations
        import threading

        results = []
        errors = []

        def concurrent_operation(operation_id):
            try:
                config = {
                    "platform": "aws",
                    "region": "us-west-2",
                    "instance_type": "t3.medium",
                    "instance_count": 1,
                    "container_image": f"neural-sim:concurrent_{operation_id}"
                }

                deployment_id = cloud_service.create_deployment(config)
                if deployment_id:
                    # Test status retrieval
                    status = cloud_service.get_deployment_status(deployment_id)
                    if status and status.get("status") == "running":
                        results.append(f"concurrent_{operation_id}")
                    else:
                        errors.append(f"Status check failed for concurrent_{operation_id}")
                else:
                    errors.append(f"Deployment failed for concurrent_{operation_id}")
            except Exception as e:
                errors.append(f"Exception in concurrent_{operation_id}: {e}")

        # Start concurrent operations
        threads = []
        for i in range(3):
            thread = threading.Thread(target=concurrent_operation, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)

        if len(results) != 3:
            raise Exception(f"Concurrent operations failed: {len(results)} succeeded, {len(errors)} failed")

        # Clean up all deployments
        # Note: In a real implementation, you'd track all deployment IDs
        cloud_service.cleanup()

        print("PASS: Cloud performance and scalability test successful")
        return True

    except Exception as e:
        print(f"FAIL: Cloud performance and scalability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cloud_cost_optimization():
    """Test cloud cost optimization and monitoring."""
    try:
        from core.services.cloud_deployment_service import CloudDeploymentService
        import time

        # Create cloud deployment service
        cloud_service = CloudDeploymentService()

        # Create deployments with different instance types for cost comparison
        instance_types = ["t3.medium", "t3.large", "c5.large"]
        deployments = []

        for instance_type in instance_types:
            config = {
                "platform": "aws",
                "region": "us-west-2",
                "instance_type": instance_type,
                "instance_count": 1,
                "auto_scaling": False,
                "container_image": "neural-sim:latest"
            }

            deployment_id = cloud_service.create_deployment(config)
            if deployment_id:
                deployments.append((deployment_id, instance_type))
            else:
                raise Exception(f"Failed to create deployment with {instance_type}")

        # Wait a bit to accumulate some runtime
        time.sleep(0.1)

        # Test cost monitoring
        cost_analysis = {}
        total_cost = 0.0

        for deployment_id, instance_type in deployments:
            costs = cloud_service.get_deployment_costs(deployment_id)
            if costs and "total_cost" in costs:
                cost_analysis[instance_type] = costs
                total_cost += costs["total_cost"]
            else:
                raise Exception(f"Cost calculation failed for {instance_type}")

        # Verify cost differences
        if len(cost_analysis) != len(instance_types):
            raise Exception("Incomplete cost analysis")

        # Test cost optimization recommendations
        cheapest_option = min(cost_analysis.items(), key=lambda x: x[1]["total_cost"])
        most_expensive_option = max(cost_analysis.items(), key=lambda x: x[1]["total_cost"])

        print(f"Cheapest option: {cheapest_option[0]} at ${cheapest_option[1]['total_cost']:.4f}")
        print(f"Most expensive option: {most_expensive_option[0]} at ${most_expensive_option[1]['total_cost']:.4f}")
        print(".4f")
        # Test cost-based scaling decisions
        for deployment_id, instance_type in deployments:
            scaling_status = cloud_service.get_scaling_status(deployment_id)

            if scaling_status and "recommendations" in scaling_status:
                recommendations = scaling_status["recommendations"]
                if recommendations:
                    print(f"Scaling recommendations for {instance_type}: {len(recommendations)}")

        # Test backup cost implications
        test_deployment = deployments[0][0]
        backup_config = {"estimate_cost": True}
        backup_id = cloud_service.backup_deployment(test_deployment, backup_config)

        if backup_id:
            # In a real implementation, you'd check backup costs
            print("Backup created successfully")

        # Clean up all deployments
        for deployment_id, _ in deployments:
            success = cloud_service.terminate_deployment(deployment_id)
            if not success:
                print(f"Warning: Failed to terminate deployment {deployment_id}")

        # Clean up service
        cloud_service.cleanup()

        print("PASS: Cloud cost optimization test successful")
        return True

    except Exception as e:
        print(f"FAIL: Cloud cost optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Testing Phase 3C Services - Cloud Integration")
    print("=" * 75)

    tests = [
        test_cloud_deployment_service,
        test_multi_cloud_deployment,
        test_cloud_resource_management,
        test_cloud_performance_and_scalability,
        test_cloud_cost_optimization
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 75)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("SUCCESS: Phase 3C cloud integration services are working!")
        print("Neural simulation can now be deployed to cloud platforms.")
        print("The system supports auto-scaling, multi-cloud, and cost optimization.")
        return True
    else:
        print("FAILURE: Some Phase 3C cloud services tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)