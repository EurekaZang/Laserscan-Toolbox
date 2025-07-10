#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import LaserScan
import message_filters
import numpy as np
from scipy.optimize import minimize
import threading
from rclpy.executors import MultiThreadedExecutor

class CalibrationControllerNode(Node):
    def __init__(self):
        super().__init__('calibration_controller_node')

        self.declare_parameter('preprocessor_node_name', 'preprocessor_node')
        self.preprocessor_name = self.get_parameter('preprocessor_node_name').get_parameter_value().string_value
        
        # --- Threading and Synchronization ---
        self.lock = threading.Lock()
        self.data_received_event = threading.Event()
        self.latest_scans = None
        self.optimization_thread = threading.Thread(target=self.optimization_loop)
        self.is_running = True

        # --- Subscribers ---
        self.real_scan_sub = message_filters.Subscriber(self, LaserScan, '/scan')
        self.unscaled_scan_sub = message_filters.Subscriber(self, LaserScan, '/scan_unscaled')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.real_scan_sub, self.unscaled_scan_sub], queue_size=10, slop=0.2
        )
        self.ts.registerCallback(self.sync_callback)

        # --- Client for setting parameters ---
        self.param_client = self.create_client(
            SetParameters, f'/{self.preprocessor_name}/set_parameters'
        )
        while not self.param_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Waiting for {self.preprocessor_name} parameter service...')

        self.get_logger().info('Calibration Controller is ready.')
        self.optimization_thread.start()

    def destroy_node(self):
        """Custom cleanup."""
        self.is_running = False
        self.data_received_event.set() # Wake up the thread so it can exit
        self.optimization_thread.join()
        super().destroy_node()

    def sync_callback(self, real_scan, unscaled_scan):
        """This callback just stores the latest data and signals the optimizer."""
        with self.lock:
            self.latest_scans = (real_scan, unscaled_scan)
        self.data_received_event.set() # Signal that new data is available

    def set_and_wait_for_params(self, A, B):
        """Sets parameters and properly waits for the service call to complete."""
        request = SetParameters.Request()
        request.parameters.append(Parameter(name='calib_A', value=A).to_parameter_msg())
        request.parameters.append(Parameter(name='calib_B', value=B).to_parameter_msg())
        
        future = self.param_client.call_async(request)
        
        # We need to spin the node to process the service response.
        # This is why we need a MultiThreadedExecutor.
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        
        if future.result() is None:
            self.get_logger().error('Service call to set parameters failed or timed out.')
            return False
            
        # Check the result for success
        for res in future.result().results:
            if not res.successful:
                self.get_logger().error(f"Failed to set parameter: {res.reason}")
                return False
        return True

    def calculate_error(self, real_scan, processed_scan):
        """A pure function to calculate error between two scans."""
        real_ranges = np.array(real_scan.ranges)
        processed_ranges = np.array(processed_scan.ranges)

        valid_mask = np.isfinite(real_ranges) & np.isfinite(processed_ranges)
        valid_mask &= (real_ranges > real_scan.range_min)
        valid_mask &= (processed_ranges > unscaled_scan.range_min if hasattr(unscaled_scan, 'range_min') else 0.1)

        if np.sum(valid_mask) < 20:
            return 1e9

        error = np.mean(np.abs(real_ranges[valid_mask] - processed_ranges[valid_mask]))
        return error

    def objective_function(self, params):
        """This function is called by the optimizer."""
        A, B = params
        
        # 1. Set the new parameters and wait for confirmation
        if not self.set_and_wait_for_params(A, B):
            return 1e9 # Return a large error if setting parameters failed

        # 2. Wait for the next fresh scan data generated with these new parameters
        self.data_received_event.clear() # Clear the event flag
        if not self.data_received_event.wait(timeout=2.0): # Wait up to 2 seconds for new data
            self.get_logger().warn("Did not receive new scan data after parameter update. Timeout.")
            return 1e9 # Large error

        # 3. Calculate error using the newly received data
        with self.lock:
            if self.latest_scans is None:
                return 1e9
            real_scan, processed_scan = self.latest_scans
        
        error = self.calculate_error(real_scan, processed_scan)
        self.get_logger().info(f'Testing params A={A:.4f}, B={B:.4f} -> Error={error:.4f}', throttle_duration_sec=0.5)
        return error

    def optimization_loop(self):
        """The main logic loop running in a separate thread."""
        self.get_logger().info('Optimization thread started. Waiting for first data...')
        
        # Wait for the very first message to arrive
        self.data_received_event.wait() 
        if not self.is_running: return

        self.get_logger().info("Received first scan data. Starting optimization...")
        
        initial_guess = [1.0, 0.0]

        result = minimize(
            self.objective_function,
            initial_guess,
            method='Nelder-Mead',
            options={'xatol': 1e-3, 'fatol': 1e-3, 'disp': False, 'maxiter': 50}
        )

        self.get_logger().info("--- CALIBRATION COMPLETE ---")
        # ... (rest of the logging is the same) ...
        final_A, final_B = result.x
        self.get_logger().info(f"Optimal parameter calib_A: {final_A:.6f}")
        self.get_logger().info(f"Optimal parameter calib_B: {final_B:.6f}")
        # ...

        # Set the final best parameters one last time
        self.set_and_wait_for_params(final_A, final_B)
        self.is_running = False # Signal main thread to shut down

def main(args=None):
    rclpy.init(args=args)
    node = CalibrationControllerNode()
    
    # A MultiThreadedExecutor is required to allow the service client (in the optimization thread)
    # and the subscriptions (in the main thread) to be processed concurrently.
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        # Spin the executor instead of the node directly
        while node.is_running:
            executor.spin_once(timeout_sec=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
