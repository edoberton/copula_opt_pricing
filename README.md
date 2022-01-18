# copula_opt_pricing

This repo contains the tools for the computation of the price of a European Call Spread Option, whose dependence structure is defined by the Plackett Copula. The functions are implemented to work jointly with those of the Heston-Nandi GARCH library.

Main functions include:
- Estimation of the Plackett Copula's θ parameter
- Simulation of future correlated asset prices
- Montecarlo Simulation for spread option pricing
- Novel closed form option pricing by means of Copula properties

Future development include:
- Disentanglement of HN-GARCH functions, to work with any marginal distribution with closed form characteristic function
