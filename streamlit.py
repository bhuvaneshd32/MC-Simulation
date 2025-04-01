#!/usr/bin/env python3
"""
Streamlit Interface for CUDA-Accelerated Monte Carlo Barrier Option Pricing with CPU Fallback
"""

import streamlit as st
import matplotlib.pyplot as plt
from mc_simulation import MonteCarloBarrierOption
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def streamlit_app():
    st.title("Monte Carlo Barrier Option Pricing")
    st.markdown("""
    Monte Carlo simulation for barrier options using GPU acceleration with CuPy (if available) or CPU with NumPy.
    Configure parameters below and run either a single simulation or a convergence study.
    """)

    st.sidebar.header("Parameters")

    st.sidebar.subheader("Financial Parameters")
    S0 = st.sidebar.number_input("Initial Stock Price (S0)", min_value=0.1, value=100.0, step=1.0)
    K = st.sidebar.number_input("Strike Price (K)", min_value=0.1, value=100.0, step=1.0)
    T = st.sidebar.number_input("Time to Maturity (years)", min_value=0.1, value=1.0, step=0.1)
    r = st.sidebar.number_input("Risk-Free Rate", min_value=0.0, value=0.05, step=0.01, format="%.3f")
    sigma = st.sidebar.number_input("Volatility", min_value=0.01, value=0.2, step=0.01, format="%.2f")
    barrier = st.sidebar.number_input("Barrier Level", min_value=0.1, value=90.0, step=1.0)

    st.sidebar.subheader("Simulation Parameters")
    steps = st.sidebar.number_input("Time Steps", min_value=1, value=252, step=10)
    paths = st.sidebar.number_input("Paths", min_value=1000, value=1000000, step=10000)
    option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
    barrier_type = st.sidebar.selectbox("Barrier Type", ["down-and-out", "up-and-out", "down-and-in", "up-and-in"])
    use_antithetic = st.sidebar.checkbox("Use Antithetic Variates", value=True)
    force_cpu = st.sidebar.checkbox("Force CPU Usage", value=False, help="Use CPU even if GPU is available")
    device_id = st.sidebar.number_input("GPU Device ID", min_value=0, value=0, step=1, disabled=force_cpu)
    batch_size = st.sidebar.number_input("Batch Size (optional)", min_value=0, value=0, step=1000, 
                                         help="Set to 0 to run all paths at once")

    st.sidebar.subheader("Convergence Study Parameters")
    min_paths = st.sidebar.number_input("Min Paths", min_value=1000, value=1000, step=1000)
    max_paths = st.sidebar.number_input("Max Paths", min_value=1000, value=paths, step=10000)
    conv_steps = st.sidebar.number_input("Steps", min_value=2, value=10, step=1)

    run_sim = st.button("Run Simulation")
    run_conv = st.button("Run Convergence Study")

    try:
        mc_option = MonteCarloBarrierOption(
            S0=S0, K=K, T=T, r=r, sigma=sigma, barrier=barrier,
            steps=steps, paths=paths, option_type=option_type,
            barrier_type=barrier_type, use_antithetic=use_antithetic,
            device_id=device_id, force_cpu=force_cpu
        )

        if run_sim:
            with st.spinner('Running simulation...'):
                batch_size = None if batch_size == 0 else batch_size
                price = mc_option.simulate(batch_size=batch_size)
            
            st.subheader("Simulation Results")
            st.write(f"**Option Type:** {option_type.upper()} {barrier_type.upper().replace('-', ' ')}")
            st.write(f"**Option Price:** {mc_option.option_price:.4f}")
            st.write(f"**Standard Error:** {mc_option.std_error:.6f}")
            st.write(f"**95% Confidence Interval:** [{mc_option.option_price - 1.96*mc_option.std_error:.4f}, "
                     f"{mc_option.option_price + 1.96*mc_option.std_error:.4f}]")
            st.write(f"**Paths Generated:** {mc_option.paths_generated:,}")
            st.write(f"**Simulation Time:** {mc_option.simulation_time:.4f} seconds")
            st.write(f"**Paths per Second:** {mc_option.paths_generated/mc_option.simulation_time:,.0f}")
            st.write(f"**Computation Device:** {'GPU' if mc_option.use_gpu else 'CPU'}")

        if run_conv:
            st.subheader("Convergence Study")
            with st.spinner('Running convergence study...'):
                mc_option.convergence_study(min_paths=min_paths, max_paths=max_paths, steps=conv_steps)
            
            if mc_option.convergence_data:
                path_counts, prices, errors = mc_option.convergence_data
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                ax1.plot(path_counts, prices, 'b-o')
                ax1.set_xlabel('Number of Paths')
                ax1.set_ylabel('Option Price')
                ax1.set_title('Price Convergence')
                ax1.grid(True)

                ax2.plot(path_counts, errors, 'r-o')
                ax2.set_xlabel('Number of Paths')
                ax2.set_ylabel('Standard Error')
                ax2.set_title('Error Reduction')
                ax2.grid(True)

                plt.tight_layout()
                st.pyplot(fig)

                st.write("Convergence Data:")
                data = {"Paths": path_counts, "Price": prices, "Std Error": errors}
                st.table(data)

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Please check your parameters or GPU availability.")

if __name__ == "__main__":
    streamlit_app()