import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import logging
import multiprocessing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataGenerator:
    """
    Data generator for SIRS and Lotka-Volterra model simulations.
    """
    def __init__(self, config: Dict):
        """
        Initializes the data generator with configuration settings.

        Parameters:
            config (Dict): Dictionary containing configuration settings.
        """
        self.config = config

    def generate_sirs_simulations(self, num_simulations: int, timesteps: int) -> pd.DataFrame:
        """
        Generates SIRS (Susceptible-Infected-Recovered-Susceptible) model simulations.

        Parameters:
            num_simulations (int): Number of simulations to generate.
            timesteps (int): Number of timesteps for each simulation.

        Returns:
            pd.DataFrame: DataFrame containing the simulation results. Each row represents a timestep,
                         and the columns are ['time', 'susceptible', 'infected', 'recovered'].
        """
        results = []
        beta = self.config['sirs_beta']
        gamma = self.config['sirs_gamma']
        sigma = self.config['sirs_sigma']

        for _ in range(num_simulations):
            s = 0.99
            i = 0.01
            r = 0.00
            data = []
            for t in range(timesteps):
                data.append([t, s, i, r])
                dn = beta * s * i - gamma * i - sigma * r
                ds = -beta * s * i + sigma * r
                dr = gamma * i - sigma * r
                s += ds
                i += dn
                r += dr
            results.append(pd.DataFrame(data, columns=['time', 'susceptible', 'infected', 'recovered']))

        return pd.concat(results, ignore_index=True)

    def generate_lotka_volterra_simulations(self, num_simulations: int, timesteps: int) -> pd.DataFrame:
        """
        Generates Lotka-Volterra (predator-prey) model simulations.

        Parameters:
            num_simulations (int): Number of simulations to generate.
            timesteps (int): Number of timesteps for each simulation.

        Returns:
            pd.DataFrame: DataFrame containing the simulation results. Each row represents a timestep,
                         and the columns are ['time', 'prey', 'predator'].
        """
        results = []
        alpha = self.config['lotka_volterra_alpha']
        beta = self.config['lotka_volterra_beta']
        delta = self.config['lotka_volterra_delta']
        gamma = self.config['lotka_volterra_gamma']

        for _ in range(num_simulations):
            prey = 20.0
            predator = 20.0
            data = []
            for t in range(timesteps):
                data.append([t, prey, predator])
                dprey = alpha * prey - beta * prey * predator
                dpredator = delta * prey * predator - gamma * predator
                prey += dprey
                predator += dpredator
            results.append(pd.DataFrame(data, columns=['time', 'prey', 'predator']))

        return pd.concat(results, ignore_index=True)

def generate_simulations(config: Dict, num_simulations: int, timesteps: int, model: str) -> pd.DataFrame:
    """
    Generates simulations for the specified model.

    Parameters:
        config (Dict): Configuration settings.
        num_simulations (int): Number of simulations to generate.
        timesteps (int): Number of timesteps for each simulation.
        model (str): Model to simulate ('sirs' or 'lotka_volterra').

    Returns:
        pd.DataFrame: DataFrame containing the simulation results.
    """
    if model == 'sirs':
        generator = DataGenerator(config)
        return generator.generate_sirs_simulations(num_simulations, timesteps)
    elif model == 'lotka_volterra':
        generator = DataGenerator(config)
        return generator.generate_lotka_volterra_simulations(num_simulations, timesteps)
    else:
        raise ValueError(f"Invalid model type: {model}. Supported models: 'sirs', 'lotka_volterra'.")

def main():
    config = {
        'sirs_beta': 0.1,
        'sirs_gamma': 0.05,
        'sirs_sigma': 0.01,
        'lotka_volterra_alpha': 0.1,
        'lotka_volterra_beta': 0.05,
        'lotka_volterra_delta': 0.5,
        'lotka_volterra_gamma': 0.01
    }

    num_simulations = 10
    timesteps = 100

    model = 'sirs'  # Can be 'sirs' or 'lotka_volterra'

    df = generate_simulations(config, num_simulations, timesteps, model)
    output_path = f"simulations_{model}.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Simulations saved to {output_path}")

if __name__ == '__main__':
    main()