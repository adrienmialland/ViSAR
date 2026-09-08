
import sys
import click
from pathlib import Path 

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data import Dataset
from models.encoders import get_encoder
from config import load_configs

@click.command()
@click.option('--config', '-c', type=click.Path(exists=True, dir_okay=False), required=True, help='Path to the YAML configuration file.')
def encode(config):
    config_set = load_configs(config, False, False)
    
    dataset  = Dataset(config_set.base_config.dataset)
    enc_name = config_set.base_config.retrieval.encoder

    encoder = get_encoder(enc_name)
    encoder.encode_dataset(dataset)
    
if __name__ == "__main__":
    encode()