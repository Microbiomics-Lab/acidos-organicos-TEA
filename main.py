# libraries

import logging
from src.visualization.composition import run_composition_pipeline
from src.visualization.evaluation import run_evaluation_pipeline
from src.visualization.exploration import run_exploration_pipeline
from src.features.cleaning_data import prepare_data
from src.models.logistic_elastic_net_weights import dev_logistic_elastic_net_weights
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    try:
        logging.info("Starting the data visualization...\n")

        # Data processed
        prepare_data("data/raw/results-OAT_20casos_V3.xlsx")
        # # Models
        logging.info("Developing the models...")
        dev_logistic_elastic_net_weights()

        # # Visualizations
        logging.info("Starting the data visualization...\n")
        run_composition_pipeline()
        run_evaluation_pipeline()
        run_exploration_pipeline()

        logging.info("--- Data Analytics Finished Successfully! ---")

    except Exception as e:
        logging.error(f"Error encontrado: {e}")