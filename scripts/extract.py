import os
import sys
import click

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data import Dataset
from config import load_configs

@click.command()
@click.option('--config', '-c', type=click.Path(exists=True, dir_okay=False), required=True, help='Path to the YAML configuration file.')
def extract(config):
    config_set = load_configs(config, False, False)

    dataset = Dataset(config_set.base_config.dataset)
    
    dataset.extract_doc_pages_as_images()

if __name__ == "__main__":
    extract()