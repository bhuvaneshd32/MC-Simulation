#!/usr/bin/env python3
"""
CUDA-Accelerated Monte Carlo Simulation for Barrier Options with Advanced Features
and CPU Fallback
"""

import numpy as np
import argparse
import time
import math
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Union, Tuple, Optional

# Try to import CuPy; fall back to NumPy if not available
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

class MonteCarloBarrierOption:
    def __init__(self,
                 S0: float,
                 K: float,
                 T: float,
                 r: float,
                 sigma: float,
                 barrier: float,
                 steps: int,
                 paths: int,
                 option_type: str = 'call',
                 barrier_type: str = 'down-and-out',
                 use_antithetic: bool = True,
                 seed: Optional[int] = None,
                 device_id: int = 0,
                 force_cpu: bool = False):
        # Parameter validation
        if not all(isinstance(x, (int, float)) and x > 0 for x in [S0, K, T, r, sigma, barrier]):
            raise ValueError("All financial parameters must be positive numbers")
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError("Steps must be a positive integer")
        if not isinstance(paths, int) or paths <= 0:
            raise ValueError("Paths must be a positive integer")
        if option_type.lower() not in ['call', 'put']:
            raise ValueError("Option type must be either 'call' or 'put'")
        if barrier_type.lower() not in ['down-and-out', 'up-and-out', 'down-and-in', 'up-and-in']:
            raise ValueError("Invalid barrier type")

        # Financial parameters
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.barrier = float(barrier)

        # Simulation parameters
        self.steps = int(steps)
        self.paths = int(paths)
        self.option_type = option_type.lower()
        self.barrier_type = barrier_type.lower()
        self.use_antithetic = bool(use_antithetic)
        self.seed = seed if seed is not None else int(time.time())
        self.device_id = device_id
        self.force_cpu = force_cpu

        # Results storage
        self.option_price = None
        self.std_error = None
        self.simulation_time = None
        self.convergence_data = None
        self.paths_generated = 0

        # Determine whether to use GPU or CPU
        self.use_gpu = GPU_AVAILABLE and not force_cpu
        self.xp = cp if self.use_gpu else np

        # GPU initialization if applicable
        if self.use_gpu:
            self._initialize_gpu()
        else:
            print("Using CPU (NumPy) for computation")

        # Set random seed
        self.xp.random.seed(self.seed)

        # Print initialization message
        self._print_init_message()

    def _initialize_gpu(self) -> None:
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            if self.device_id >= device_count:
                raise ValueError(f"Device ID {self.device_id} not available. Only {device_count} GPU(s) detected.")
            cp.cuda.Device(self.device_id).use()
            print(f"Using GPU Device {self.device_id}: {cp.cuda.runtime.getDeviceProperties(self.device_id)['name'].decode()}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GPU: {str(e)}")

    def _print_init_message(self) -> None:
        print("\n" + "="*80)
        print("MONTE CARLO BARRIER OPTION PRICING (CUDA ACCELERATED)" if self.use_gpu else "MONTE CARLO BARRIER OPTION PRICING (CPU)")
        print("="*80)
        print(f"Initialization Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Random Seed: {self.seed}")
        print("\nFinancial Parameters:")
        print(f"  Initial Stock Price (S0): {self.S0:.2f}")
        print(f"  Strike Price (K): {self.K:.2f}")
        print(f"  Time to Maturity (T): {self.T:.2f} years")
        print(f"  Risk-Free Rate (r): {self.r:.4f}")
        print(f"  Volatility (σ): {self.sigma:.4f}")
        print(f"  Barrier Level: {self.barrier:.2f}")
        print("\nSimulation Parameters:")
        print(f"  Option Type: {self.option_type}")
        print(f"  Barrier Type: {self.barrier_type}")
        print(f"  Time Steps: {self.steps}")
        print(f"  Paths: {self.paths}")
        print(f"  Antithetic Variates: {'Enabled' if self.use_antithetic else 'Disabled'}")
        print("="*80 + "\n")

    def _generate_paths(self) -> 'xp.ndarray':
        dt = self.T / self.steps
        drift = (self.r - 0.5 * self.sigma**2) * dt
        volatility = self.sigma * math.sqrt(dt)

        actual_paths = self.paths // 2 if self.use_antithetic else self.paths
        # Handle dtype conditionally
        if self.use_gpu:
            Z = self.xp.random.standard_normal((actual_paths, self.steps), dtype=self.xp.float32)
            S = self.xp.empty((actual_paths, self.steps), dtype=self.xp.float32)
        else:
            Z = self.xp.random.standard_normal((actual_paths, self.steps)).astype(np.float32)
            S = self.xp.empty((actual_paths, self.steps), dtype=np.float32)
        S[:, 0] = self.S0

        for t in range(1, self.steps):
            S[:, t] = S[:, t-1] * self.xp.exp(drift + volatility * Z[:, t])

        if self.use_antithetic:
            if self.use_gpu:
                S_antithetic = self.xp.empty_like(S, dtype=self.xp.float32)
            else:
                S_antithetic = self.xp.empty_like(S, dtype=np.float32)
            S_antithetic[:, 0] = self.S0
            for t in range(1, self.steps):
                S_antithetic[:, t] = S_antithetic[:, t-1] * self.xp.exp(drift - volatility * Z[:, t])
            S = self.xp.concatenate((S, S_antithetic), axis=0)

        self.paths_generated = S.shape[0]
        return S

    def _check_barrier_condition(self, S: 'xp.ndarray') -> 'xp.ndarray':
        if 'down' in self.barrier_type:
            barrier_hit = self.xp.any(S <= self.barrier, axis=1)
        elif 'up' in self.barrier_type:
            barrier_hit = self.xp.any(S >= self.barrier, axis=1)
        else:
            raise ValueError("Invalid barrier type")

        if 'in' in self.barrier_type:
            barrier_hit = ~barrier_hit
        return barrier_hit

    def _calculate_payoffs(self, S: 'xp.ndarray') -> 'xp.ndarray':
        final_prices = S[:, -1]
        if self.option_type == 'call':
            payoffs = self.xp.maximum(final_prices - self.K, 0)
        else:
            payoffs = self.xp.maximum(self.K - final_prices, 0)
        return payoffs

    def _compute_statistics(self, payoffs: 'xp.ndarray') -> Tuple[float, float]:
        discount_factor = self.xp.exp(-self.r * self.T)
        discounted_payoffs = discount_factor * payoffs
        option_price = self.xp.mean(discounted_payoffs).item() if self.use_gpu else float(self.xp.mean(discounted_payoffs))
        std_error = (self.xp.std(discounted_payoffs) / self.xp.sqrt(len(payoffs))).item() if self.use_gpu else float(self.xp.std(discounted_payoffs) / self.xp.sqrt(len(payoffs)))
        return option_price, std_error

    def simulate(self, batch_size: Optional[int] = None) -> float:
        start_time = time.time()

        if batch_size is None:
            S = self._generate_paths()
            barrier_hit = self._check_barrier_condition(S)
            payoffs = self._calculate_payoffs(S)
            payoffs[barrier_hit] = 0.0
            self.option_price, self.std_error = self._compute_statistics(payoffs)
        else:
            total_batches = math.ceil(self.paths / batch_size)
            batch_results = []
            for batch in range(total_batches):
                current_paths = min(batch_size, self.paths - batch * batch_size)
                self.paths = current_paths
                S = self._generate_paths()
                barrier_hit = self._check_barrier_condition(S)
                payoffs = self._calculate_payoffs(S)
                payoffs[barrier_hit] = 0.0
                batch_price, _ = self._compute_statistics(payoffs)
                batch_results.append(batch_price)
                print(f"Batch {batch+1}/{total_batches} complete - Current estimate: {batch_price:.4f}")
            self.paths = batch_size * total_batches
            self.option_price = np.mean(batch_results)
            self.std_error = np.std(batch_results) / np.sqrt(total_batches)

        self.simulation_time = time.time() - start_time
        self._print_results()
        return self.option_price

    def convergence_study(self, min_paths: int = 1000, max_paths: int = None,
                          steps: int = 10, log_scale: bool = True) -> None:
        if max_paths is None:
            max_paths = self.paths

        if log_scale:
            path_counts = np.logspace(np.log10(min_paths), np.log10(max_paths), steps, dtype=int)
        else:
            path_counts = np.linspace(min_paths, max_paths, steps, dtype=int)

        prices = []
        errors = []

        print("\nRunning Convergence Study...")
        original_paths = self.paths

        for count in path_counts:
            self.paths = count
            price = self.simulate()
            prices.append(price)
            errors.append(self.std_error)
            print(f"Paths: {count:8d} | Price: {price:.4f} | Std Error: {self.std_error:.6f}")

        self.paths = original_paths
        self.convergence_data = (path_counts, prices, errors)
        self.plot_convergence()

    def plot_convergence(self) -> None:
        if self.convergence_data is None:
            raise ValueError("No convergence data available. Run convergence_study() first.")

        path_counts, prices, errors = self.convergence_data
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(path_counts, prices, 'b-o')
        plt.xlabel('Number of Paths')
        plt.ylabel('Option Price')
        plt.title('Price Convergence')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(path_counts, errors, 'r-o')
        plt.xlabel('Number of Paths')
        plt.ylabel('Standard Error')
        plt.title('Error Reduction')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def _print_results(self) -> None:
        print("\n" + "="*80)
        print("SIMULATION RESULTS")
        print("="*80)
        print(f"Option Type: {self.option_type.upper()} {self.barrier_type.upper().replace('-', ' ')}")
        print(f"Final Price Estimate: {self.option_price:.4f}")
        print(f"Standard Error: {self.std_error:.6f}")
        print(f"95% Confidence Interval: [{self.option_price - 1.96*self.std_error:.4f}, {self.option_price + 1.96*self.std_error:.4f}]")
        print(f"Paths Generated: {self.paths_generated:,}")
        print(f"Simulation Time: {self.simulation_time:.4f} seconds")
        print(f"Paths per Second: {self.paths_generated/self.simulation_time:,.0f}")
        print("="*80 + "\n")

    def __str__(self) -> str:
        return (f"MonteCarloBarrierOption(S0={self.S0:.2f}, K={self.K:.2f}, T={self.T:.2f}, "
                f"r={self.r:.4f}, σ={self.sigma:.4f}, barrier={self.barrier:.2f}, "
                f"type={self.option_type}, barrier_type={self.barrier_type})")

def run_interactive() -> None:
    print("\n" + "="*80)
    print("INTERACTIVE MONTE CARLO BARRIER OPTION PRICING")
    print("="*80)

    try:
        S0 = float(input("Initial Stock Price (S0): "))
        K = float(input("Strike Price (K): "))
        T = float(input("Time to Maturity (years, T): "))
        r = float(input("Risk-Free Rate (decimal, r): "))
        sigma = float(input("Volatility (decimal, σ): "))
        barrier = float(input("Barrier Level: "))
        steps = int(input("Number of Time Steps: "))
        paths = int(input("Number of Monte Carlo Paths: "))
        option_type = input("Option Type (call/put): ").strip().lower()
        barrier_type = input("Barrier Type (down-and-out/up-and-out/down-and-in/up-and-in): ").strip().lower()
        use_antithetic = input("Use Antithetic Variates? (y/n): ").strip().lower() == 'y'
        force_cpu = input("Force CPU usage? (y/n): ").strip().lower() == 'y'

        mc_option = MonteCarloBarrierOption(
            S0=S0, K=K, T=T, r=r, sigma=sigma, barrier=barrier,
            steps=steps, paths=paths, option_type=option_type,
            barrier_type=barrier_type, use_antithetic=use_antithetic,
            force_cpu=force_cpu
        )

        run_convergence = input("Run Convergence Study? (y/n): ").strip().lower() == 'y'
        if run_convergence:
            mc_option.convergence_study()
        else:
            mc_option.simulate()

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please try again with valid inputs.\n")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="CUDA Monte Carlo Barrier Option Pricing with Advanced Features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--S0', type=float, default=100.0, help="Initial stock price")
    parser.add_argument('--K', type=float, default=100.0, help="Strike price")
    parser.add_argument('--T', type=float, default=1.0, help="Time to maturity (years)")
    parser.add_argument('--r', type=float, default=0.05, help="Risk-free interest rate")
    parser.add_argument('--sigma', type=float, default=0.2, help="Volatility")
    parser.add_argument('--barrier', type=float, default=90.0, help="Barrier level")
    parser.add_argument('--steps', type=int, default=252, help="Number of time steps")
    parser.add_argument('--paths', type=int, default=1000000, help="Number of Monte Carlo paths")
    parser.add_argument('--option_type', type=str, choices=['call', 'put'], default='call', help="Option type")
    parser.add_argument('--barrier_type', type=str,
                        choices=['down-and-out', 'up-and-out', 'down-and-in', 'up-and-in'],
                        default='down-and-out', help="Barrier type")
    parser.add_argument('--no_antithetic', action='store_false', dest='use_antithetic',
                        help="Disable antithetic variates")
    parser.add_argument('--seed', type=int, default=None, help="Random seed")
    parser.add_argument('--device', type=int, default=0, help="GPU device ID")
    parser.add_argument('--batch_size', type=int, default=None, help="Run simulation in batches of this size")
    parser.add_argument('--convergence', action='store_true', help="Run convergence study")
    parser.add_argument('--force_cpu', action='store_true', help="Force CPU usage even if GPU is available")

    return parser.parse_args()

def main():
    try:
        args = parse_arguments()
        mc_option = MonteCarloBarrierOption(
            S0=args.S0, K=args.K, T=args.T, r=args.r, sigma=args.sigma,
            barrier=args.barrier, steps=args.steps, paths=args.paths,
            option_type=args.option_type, barrier_type=args.barrier_type,
            use_antithetic=args.use_antithetic, seed=args.seed,
            device_id=args.device, force_cpu=args.force_cpu
        )
        if args.convergence:
            mc_option.convergence_study()
        else:
            mc_option.simulate(batch_size=args.batch_size)
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Use --help for usage information.\n")

if __name__ == "__main__":
    try:
        run_interactive()
    except:
        main()