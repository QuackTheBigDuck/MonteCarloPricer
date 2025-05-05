import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time


class MonteCarloOptionPricer:
    def __init__(self, S0, K, r, sigma, T, option_type='call', simulation_count=100000):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.option_type = option_type.lower()
        self.simulation_count = simulation_count

        if self.S0 <= 0 or self.K <= 0 or self.T <= 0:
            raise ValueError("Stock price, strike price, and time to maturity must be positive")
        if self.sigma <= 0:
            raise ValueError("Volatility must be positive")
        if self.option_type not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")

    def simulate_paths(self, time_steps=1):
        dt = self.T / time_steps
        Z = np.random.standard_normal(size=(self.simulation_count, time_steps))

        S = np.zeros((self.simulation_count, time_steps + 1))
        S[:, 0] = self.S0

        for t in range(1, time_steps + 1):
            S[:, t] = S[:, t - 1] * np.exp((self.r - 0.5 * self.sigma ** 2) * dt +
                                           self.sigma * np.sqrt(dt) * Z[:, t - 1])

        return S[:, -1]

    def price_european(self, time_steps=1):
        start_time = time.time()

        final_prices = self.simulate_paths(time_steps)

        if self.option_type == 'call':
            payoffs = np.maximum(final_prices - self.K, 0)
        else:
            payoffs = np.maximum(self.K - final_prices, 0)

        option_price = np.exp(-self.r * self.T) * np.mean(payoffs)

        end_time = time.time()
        self.computation_time = end_time - start_time

        return option_price

    def price_asian(self, time_steps=252):
        start_time = time.time()

        dt = self.T / time_steps
        Z = np.random.standard_normal(size=(self.simulation_count, time_steps))

        S = np.zeros((self.simulation_count, time_steps + 1))
        S[:, 0] = self.S0

        for t in range(1, time_steps + 1):
            S[:, t] = S[:, t - 1] * np.exp((self.r - 0.5 * self.sigma ** 2) * dt +
                                           self.sigma * np.sqrt(dt) * Z[:, t - 1])

        avg_prices = np.mean(S, axis=1)

        if self.option_type == 'call':
            payoffs = np.maximum(avg_prices - self.K, 0)
        else:
            payoffs = np.maximum(self.K - avg_prices, 0)

        option_price = np.exp(-self.r * self.T) * np.mean(payoffs)

        end_time = time.time()
        self.computation_time = end_time - start_time

        return option_price

    def price_barrier(self, barrier, barrier_type='up-and-out', time_steps=252):
        start_time = time.time()

        dt = self.T / time_steps
        Z = np.random.standard_normal(size=(self.simulation_count, time_steps))

        S = np.zeros((self.simulation_count, time_steps + 1))
        S[:, 0] = self.S0

        for t in range(1, time_steps + 1):
            S[:, t] = S[:, t - 1] * np.exp((self.r - 0.5 * self.sigma ** 2) * dt +
                                           self.sigma * np.sqrt(dt) * Z[:, t - 1])

        if barrier_type == 'up-and-out':
            barrier_hit = np.any(S > barrier, axis=1)
            valid_paths = ~barrier_hit
        elif barrier_type == 'up-and-in':
            barrier_hit = np.any(S > barrier, axis=1)
            valid_paths = barrier_hit
        elif barrier_type == 'down-and-out':
            barrier_hit = np.any(S < barrier, axis=1)
            valid_paths = ~barrier_hit
        elif barrier_type == 'down-and-in':
            barrier_hit = np.any(S < barrier, axis=1)
            valid_paths = barrier_hit
        else:
            raise ValueError(
                "Invalid barrier type. Must be one of: 'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'")

        final_prices = S[:, -1]

        if self.option_type == 'call':
            payoffs = np.maximum(final_prices - self.K, 0) * valid_paths
        else:
            payoffs = np.maximum(self.K - final_prices, 0) * valid_paths

        option_price = np.exp(-self.r * self.T) * np.sum(payoffs) / np.sum(valid_paths) if np.sum(
            valid_paths) > 0 else 0

        end_time = time.time()
        self.computation_time = end_time - start_time

        return option_price

    def price_lookback(self, time_steps=252):
        start_time = time.time()

        dt = self.T / time_steps
        Z = np.random.standard_normal(size=(self.simulation_count, time_steps))

        S = np.zeros((self.simulation_count, time_steps + 1))
        S[:, 0] = self.S0

        for t in range(1, time_steps + 1):
            S[:, t] = S[:, t - 1] * np.exp((self.r - 0.5 * self.sigma ** 2) * dt +
                                           self.sigma * np.sqrt(dt) * Z[:, t - 1])

        S_max = np.max(S, axis=1)
        S_min = np.min(S, axis=1)

        if self.option_type == 'call':
            payoffs = np.maximum(S_max - self.K, 0)
        else:
            payoffs = np.maximum(self.K - S_min, 0)

        option_price = np.exp(-self.r * self.T) * np.mean(payoffs)

        end_time = time.time()
        self.computation_time = end_time - start_time

        return option_price

    def black_scholes_price(self):
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.option_type == 'call':
            price = self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)

        return price

    def convergence_analysis(self, min_sim=1000, max_sim=100000, steps=10):
        sim_counts = np.logspace(np.log10(min_sim), np.log10(max_sim), steps).astype(int)
        prices = np.zeros(steps)
        analytical_price = self.black_scholes_price()

        for i, sim_count in enumerate(sim_counts):
            temp_pricer = MonteCarloOptionPricer(
                self.S0, self.K, self.r, self.sigma, self.T,
                self.option_type, sim_count
            )
            prices[i] = temp_pricer.price_european()

        return sim_counts, prices, analytical_price

    def plot_convergence(self, min_sim=1000, max_sim=100000, steps=10):
        sim_counts, prices, analytical_price = self.convergence_analysis(min_sim, max_sim, steps)

        plt.figure(figsize=(10, 6))
        plt.plot(sim_counts, prices, 'o-', label='Monte Carlo Price')
        plt.axhline(y=analytical_price, color='r', linestyle='-', label='Black-Scholes Price')
        plt.xscale('log')
        plt.xlabel('Number of Simulations (log scale)')
        plt.ylabel('Option Price')
        plt.title(f'Convergence of {self.option_type.capitalize()} Option Price')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_sample_paths(self, num_paths=10, time_steps=252):
        dt = self.T / time_steps
        time_points = np.linspace(0, self.T, time_steps + 1)

        Z = np.random.standard_normal(size=(num_paths, time_steps))

        S = np.zeros((num_paths, time_steps + 1))
        S[:, 0] = self.S0

        for t in range(1, time_steps + 1):
            S[:, t] = S[:, t - 1] * np.exp((self.r - 0.5 * self.sigma ** 2) * dt +
                                           self.sigma * np.sqrt(dt) * Z[:, t - 1])

        plt.figure(figsize=(10, 6))
        for i in range(num_paths):
            plt.plot(time_points, S[i])

        plt.axhline(y=self.K, color='r', linestyle='--', label='Strike Price')
        plt.xlabel('Time (years)')
        plt.ylabel('Stock Price')
        plt.title('Sample Stock Price Paths')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0

    pricer = MonteCarloOptionPricer(S0, K, r, sigma, T, option_type='call', simulation_count=100000)

    euro_price = pricer.price_european()
    bs_price = pricer.black_scholes_price()
    print(f"European Call Option Price (Monte Carlo): {euro_price:.4f}")
    print(f"European Call Option Price (Black-Scholes): {bs_price:.4f}")
    print(f"Difference: {abs(euro_price - bs_price):.4f}")
    print(f"Computation time: {pricer.computation_time:.4f} seconds")

    asian_price = pricer.price_asian()
    print(f"\nAsian Call Option Price: {asian_price:.4f}")
    print(f"Computation time: {pricer.computation_time:.4f} seconds")

    barrier = 120
    barrier_price = pricer.price_barrier(barrier, barrier_type='up-and-out')
    print(f"\nUp-and-Out Barrier Call Option Price (Barrier = {barrier}): {barrier_price:.4f}")
    print(f"Computation time: {pricer.computation_time:.4f} seconds")

    lookback_price = pricer.price_lookback()
    print(f"\nLookback Call Option Price: {lookback_price:.4f}")
    print(f"Computation time: {pricer.computation_time:.4f} seconds")

    pricer.plot_convergence()

    pricer.plot_sample_paths()