
import sys
import click
from pathlib import Path 

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data import Dataset
from config import load_configs

from pipeline import ImageRetriever
from pipeline import AnswerGenerator

@click.command()
@click.option('--config', '-c', type=click.Path(exists=True, dir_okay=False), required=True, help='Path to the YAML configuration file.')
@click.option('--sweep', '-s', is_flag=True, required=False, help='Activate parameter sweep mode.')
def main(config, sweep):
    config_set = load_configs(config, sweep)

    dataset = Dataset(config_set.base_config.dataset, True)

    for conf in config_set.configs:
        retriever = ImageRetriever(conf.retrieval)
        generator = AnswerGenerator(conf.generator)

        retriever.retrieve_images(dataset)
        generator.generate_answer(
            dataset, retriever.get_state()
        )

if __name__ == "__main__":
    main()
