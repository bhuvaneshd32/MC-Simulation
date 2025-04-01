import streamlit as st
from mc_simulation import MonteCarloBarrierOption
import matplotlib.pyplot as plt

st.set_page_config(page_title="Monte Carlo Barrier Option Pricing", page_icon=":money_with_wings:")

st.title("Monte Carlo Barrier Option Pricing")
st.subheader("Interactive Simulation with Streamlit")

# [Detailed input sections as described, including financial and simulation parameters]

if st.button("Run Simulation"):
    try:
        mc_option = MonteCarloBarrierOption(
            S0=S0, K=K, T=T, r=r, sigma=sigma, barrier=barrier,
            steps=steps, paths=paths, option_type=option_type,
            barrier_type=barrier_type, use_antithetic=use_antithetic,
            seed=seed, device_id=device_id
        )
        if batch_size is not None:
            mc_option.simulate(batch_size=batch_size)
        else:
            mc_option.simulate()
        # Display results
        st.subheader("Simulation Results")
        st.write(f"Option Type: {mc_option.option_type.upper()} {mc_option.barrier_type.upper().replace('-', ' ')}")
        st.write(f"Final Price Estimate: {mc_option.option_price:.4f}")
        st.write(f"Standard Error: {mc_option.std_error:.6f}")
        st.write(f"95% Confidence Interval: [{mc_option.option_price - 1.96*mc_option.std_error:.4f}, {mc_option.option_price + 1.96*mc_option.std_error:.4f}]")
        st.write(f"Paths Generated: {mc_option.paths_generated:,}")
        st.write(f"Simulation Time: {mc_option.simulation_time:.4f} seconds")
        st.write(f"Paths per Second: {mc_option.paths_generated/mc_option.simulation_time:,.0f}")
        if run_convergence:
            mc_option.convergence_study()
            path_counts, prices, errors = mc_option.convergence_data
            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.plot(path_counts, prices, 'b-o')
            ax1.set_xlabel('Number of Paths')
            ax1.set_ylabel('Option Price')
            ax1.set_title('Price Convergence')
            ax1.grid(True)
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.plot(path_counts, errors, 'r-o')
            ax2.set_xlabel('Number of Paths')
            ax2.set_ylabel('Standard Error')
            ax2.set_title('Error Reduction')
            ax2.grid(True)
            fig.tight_layout()
            st.write(fig)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")