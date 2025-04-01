#!/usr/bin/env python3
"""
Monte Carlo Simulation for Barrier Options with Streamlit Interface (CUDA Version)

This implementation provides:
- Interactive Streamlit interface
- GPU-accelerated computation with CuPy (CUDA)
- Down-and-Out/Up-and-Out Barrier Options
- Call/Put European Options
- CPU/GPU utilization graphs
- Convergence study with Price Convergence and Error Reduction plots
- Performance Metrics and Visualization
"""

import streamlit as st
import time
import math
import matplotlib.pyplot as plt
import psutil
import GPUtil
from datetime import datetime
from typing import Union, Tuple, Optional

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    st.error("CuPy not installed. CUDA acceleration unavailable. Please install CuPy (e.g., 'pip install cupy-cuda11x').")
    raise SystemExit

class MonteCarloBarrierOption:
    """
    Monte Carlo Simulation for Barrier Options Pricing using CuPy (CUDA) with Streamlit integration.
    """

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
                 seed: Optional[int] = None):
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available. Cannot proceed with GPU computation.")
        
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.barrier = float(barrier)
        self.steps = int(steps)
        self.paths = int(paths)
        self.option_type = option_type.lower()
        self.barrier_type = barrier_type.lower()
        self.use_antithetic = bool(use_antithetic)
        self.seed = seed if seed is not None else int(time.time())
        self.option_price = None
        self.std_error = None
        self.simulation_time = None
        self.convergence_data = None
        self.paths_generated = 0
        self.cpu_usage = []
        self.gpu_usage = []
        
        # Initialize CUDA
        cp.cuda.Device(0).use()
        cp.random.seed(self.seed)
        self.xp = cp  # Use CuPy for all array operations

    def _monitor_resources(self):
        """Monitor CPU and GPU usage"""
        self.cpu_usage.append(psutil.cpu_percent(interval=0.1))
        gpus = GPUtil.getGPUs()
        self.gpu_usage.append(gpus[0].load * 100 if gpus else 0)

    def _generate_paths(self) -> cp.ndarray:
        dt = self.T / self.steps
        drift = (self.r - 0.5 * self.sigma**2) * dt
        volatility = self.sigma * cp.sqrt(dt)
        actual_paths = self.paths // 2 if self.use_antithetic else self.paths
        
        Z = self.xp.random.standard_normal((actual_paths, self.steps))
        S = self.xp.empty((actual_paths, self.steps))
        S[:, 0] = self.S0
        for t in range(1, self.steps):
            self._monitor_resources()
            S[:, t] = S[:, t-1] * self.xp.exp(drift + volatility * Z[:, t])
        if self.use_antithetic:
            S_antithetic = self.xp.empty_like(S)
            S_antithetic[:, 0] = self.S0
            for t in range(1, self.steps):
                self._monitor_resources()
                S_antithetic[:, t] = S_antithetic[:, t-1] * self.xp.exp(drift - volatility * Z[:, t])
            S = self.xp.concatenate((S, S_antithetic), axis=0)
        self.paths_generated = S.shape[0]
        return S

    def _check_barrier_condition(self, S: cp.ndarray) -> cp.ndarray:
        if 'down' in self.barrier_type:
            barrier_hit = self.xp.any(S <= self.barrier, axis=1)
        elif 'up' in self.barrier_type:
            barrier_hit = self.xp.any(S >= self.barrier, axis=1)
        if 'in' in self.barrier_type:
            barrier_hit = ~barrier_hit
        return barrier_hit

    def _calculate_payoffs(self, S: cp.ndarray) -> cp.ndarray:
        final_prices = S[:, -1]
        if self.option_type == 'call':
            payoffs = self.xp.maximum(final_prices - self.K, 0)
        else:
            payoffs = self.xp.maximum(self.K - final_prices, 0)
        return payoffs

    def _compute_statistics(self, payoffs: cp.ndarray) -> Tuple[float, float]:
        discount_factor = self.xp.exp(-self.r * self.T)
        discounted_payoffs = discount_factor * payoffs
        option_price = self.xp.mean(discounted_payoffs).get()  # Transfer to CPU for final result
        std_error = self.xp.std(discounted_payoffs).get() / self.xp.sqrt(len(payoffs)).get()
        return option_price, std_error

    def simulate(self, batch_size: Optional[int] = None) -> float:
        start_time = time.time()
        self.cpu_usage = []
        self.gpu_usage = []
        if batch_size is None:
            S = self._generate_paths()
            barrier_hit = self._check_barrier_condition(S)
            payoffs = self._calculate_payoffs(S)
            payoffs[barrier_hit] = 0.0
            self.option_price, self.std_error = self._compute_statistics(payoffs)
        else:
            total_batches = math.ceil(self.paths / batch_size)
            batch_results = []
            original_paths = self.paths
            for batch in range(total_batches):
                current_paths = min(batch_size, self.paths - batch * batch_size)
                self.paths = current_paths
                S = self._generate_paths()
                barrier_hit = self._check_barrier_condition(S)
                payoffs = self._calculate_payoffs(S)
                payoffs[barrier_hit] = 0.0
                batch_price, _ = self._compute_statistics(payoffs)
                batch_results.append(batch_price)
            self.paths = original_paths
            self.option_price = float(self.xp.mean(self.xp.array(batch_results)).get())
            self.std_error = float(self.xp.std(self.xp.array(batch_results)).get() / self.xp.sqrt(total_batches).get())
        self.simulation_time = time.time() - start_time
        return self.option_price

    def convergence_study(self, min_paths: int = 1000, max_paths: int = None,
                         steps: int = 10, log_scale: bool = True) -> None:
        if max_paths is None:
            max_paths = self.paths
        if log_scale:
            path_counts = self.xp.logspace(self.xp.log10(min_paths), self.xp.log10(max_paths), steps, dtype=self.xp.int32).get()
        else:
            path_counts = self.xp.linspace(min_paths, max_paths, steps, dtype=self.xp.int32).get()
        prices = []
        errors = []
        original_paths = self.paths
        for count in path_counts:
            self.paths = count
            price = self.simulate()
            prices.append(price)
            errors.append(self.std_error)
        self.paths = original_paths
        self.convergence_data = (path_counts, prices, errors)

    def plot_convergence(self):
        """Plot convergence study results in Streamlit, matching the provided image style."""
        if self.convergence_data is None:
            st.error("No convergence data available. Run convergence_study() first.")
            return

        path_counts, prices, errors = self.convergence_data

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Price convergence plot
        ax1.plot(path_counts, prices, 'b-o')
        ax1.set_xlabel('Number of Paths')
        ax1.set_ylabel('Option Price')
        ax1.set_title('Price Convergence')
        ax1.grid(True)

        # Error reduction plot
        ax2.plot(path_counts, errors, 'r-o')
        ax2.set_xlabel('Number of Paths')
        ax2.set_ylabel('Standard Error')
        ax2.set_title('Error Reduction')
        ax2.grid(True)

        plt.tight_layout()
        return fig

    def plot_resources(self):
        """Plot CPU and GPU utilization in Streamlit."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        time_points = [i * (self.simulation_time / len(self.cpu_usage)) for i in range(len(self.cpu_usage))]
        ax1.plot(time_points, self.cpu_usage, 'b-', label='CPU Usage (%)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title('CPU Utilization During Simulation')
        ax1.grid(True)
        ax1.legend()
        ax2.plot(time_points, self.gpu_usage, 'g-', label='GPU Usage (%)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('GPU Usage (%)')
        ax2.set_title('GPU Utilization During Simulation')
        ax2.grid(True)
        ax2.legend()
        plt.tight_layout()
        return fig

def main():
    st.title("Monte Carlo Barrier Option Pricing (CUDA)")
    st.markdown("""
    This tool calculates barrier option prices using Monte Carlo simulation with CUDA (CuPy).
    Adjust the parameters below and click 'Run Simulation' to see the results, convergence plots, and resource utilization.
    """)

    # Sidebar for input parameters
    st.sidebar.header("Simulation Parameters")
    S0 = st.sidebar.number_input("Initial Stock Price (S0)", min_value=1.0, value=100.0, step=1.0)
    K = st.sidebar.number_input("Strike Price (K)", min_value=1.0, value=100.0, step=1.0)
    T = st.sidebar.number_input("Time to Maturity (years)", min_value=0.1, value=1.0, step=0.1)
    r = st.sidebar.number_input("Risk-Free Rate", min_value=0.0, value=0.05, step=0.01)
    sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.01, value=0.2, step=0.01)
    barrier = st.sidebar.number_input("Barrier Level", min_value=1.0, value=90.0, step=1.0)
    steps = st.sidebar.number_input("Time Steps", min_value=10, value=252, step=10)
    paths = st.sidebar.number_input("Number of Paths", min_value=1000, value=100000, step=1000)
    option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
    barrier_type = st.sidebar.selectbox("Barrier Type", ["down-and-out", "up-and-out", "down-and-in", "up-and-in"])
    use_antithetic = st.sidebar.checkbox("Use Antithetic Variates", value=True)
    run_convergence = st.sidebar.checkbox("Run Convergence Study", value=False)

    # Display current parameters
    st.sidebar.subheader("Current Parameters")
    current_seed = int(time.time())
    st.sidebar.write(f"Seed: {current_seed}")
    st.sidebar.write(f"S0: {S0:.2f}")
    st.sidebar.write(f"K: {K:.2f}")
    st.sidebar.write(f"T: {T:.2f} years")
    st.sidebar.write(f"r: {r:.4f}")
    st.sidebar.write(f"σ: {sigma:.4f}")
    st.sidebar.write(f"Barrier: {barrier:.2f}")
    st.sidebar.write(f"Steps: {steps}")
    st.sidebar.write(f"Paths: {paths:,}")
    st.sidebar.write("Compute: GPU (CUDA via CuPy)")

    if st.button("Run Simulation"):
        with st.spinner("Running simulation on GPU..."):
            try:
                mc_option = MonteCarloBarrierOption(
                    S0=S0,
                    K=K,
                    T=T,
                    r=r,
                    sigma=sigma,
                    barrier=barrier,
                    steps=steps,
                    paths=paths,
                    option_type=option_type,
                    barrier_type=barrier_type,
                    use_antithetic=use_antithetic,
                    seed=current_seed
                )

                if run_convergence:
                    mc_option.convergence_study()
                    st.subheader("Convergence Study Results")
                    fig = mc_option.plot_convergence()
                    st.pyplot(fig)
                else:
                    price = mc_option.simulate()
                    st.subheader("Simulation Results")
                    st.write(f"**Option Type:** {option_type.upper()} {barrier_type.upper().replace('-', ' ')}")
                    st.write(f"**Price Estimate:** {mc_option.option_price:.4f}")
                    st.write(f"**Standard Error:** {mc_option.std_error:.6f}")
                    st.write(f"**95% Confidence Interval:** [{mc_option.option_price - 1.96*mc_option.std_error:.4f}, {mc_option.option_price + 1.96*mc_option.std_error:.4f}]")
                    st.write(f"**Paths Generated:** {mc_option.paths_generated:,}")
                    st.write(f"**Simulation Time:** {mc_option.simulation_time:.4f} seconds")
                    st.write(f"**Paths per Second:** {mc_option.paths_generated/mc_option.simulation_time:,.0f}")

                # Resource utilization plots
                st.subheader("Resource Utilization")
                fig = mc_option.plot_resources()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Simulation failed: {str(e)}")

if __name__ == "__main__":
    main()