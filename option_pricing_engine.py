#!/usr/bin/env python3
import math
import random

class Option:
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"):
        self.S = S          # Current stock price
        self.K = K          # Strike price
        self.T = T          # Time to maturity in years
        self.r = r          # Risk-free rate
        self.sigma = sigma  # Volatility
        self.option_type = option_type.lower()

    def _d1(self) -> float:
        return (math.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * math.sqrt(self.T))

    def _d2(self) -> float:
        return self._d1() - self.sigma * math.sqrt(self.T)

    def _norm_cdf(self, x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def black_scholes_price(self) -> float:
        d1 = self._d1()
        d2 = self._d2()
        if self.option_type == "call":
            return self.S * self._norm_cdf(d1) - self.K * math.exp(-self.r * self.T) * self._norm_cdf(d2)
        else:
            return self.K * math.exp(-self.r * self.T) * self._norm_cdf(-d2) - self.S * self._norm_cdf(-d1)

    def delta(self) -> float:
        d1 = self._d1()
        return self._norm_cdf(d1) if self.option_type == "call" else self._norm_cdf(d1) - 1

    def gamma(self) -> float:
        d1 = self._d1()
        return math.exp(-0.5 * d1**2) / (self.S * self.sigma * math.sqrt(2 * math.pi * self.T))

    def monte_carlo_price(self, simulations: int = 10000) -> float:
        payoff_sum = 0
        for _ in range(simulations):
            ST = self.S * math.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * math.sqrt(self.T) * random.gauss(0,1))
            payoff = max(0, ST - self.K) if self.option_type == "call" else max(0, self.K - ST)
            payoff_sum += payoff
        return math.exp(-self.r * self.T) * (payoff_sum / simulations)

def demo():
    call_option = Option(S=100, K=105, T=1, r=0.05, sigma=0.2, option_type="call")
    put_option = Option(S=100, K=95, T=1, r=0.05, sigma=0.2, option_type="put")

    print("Call Option (Black-Scholes) Price:", round(call_option.black_scholes_price(),2))
    print("Call Option Delta:", round(call_option.delta(),4))
    print("Call Option Gamma:", round(call_option.gamma(),4))
    print("Call Option (Monte Carlo) Price:", round(call_option.monte_carlo_price(),2))

    print("Put Option (Black-Scholes) Price:", round(put_option.black_scholes_price(),2))
    print("Put Option Delta:", round(put_option.delta(),4))
    print("Put Option Gamma:", round(put_option.gamma(),4))
    print("Put Option (Monte Carlo) Price:", round(put_option.monte_carlo_price(),2))

if __name__ == "__main__":
    demo()
