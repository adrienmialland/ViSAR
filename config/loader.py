
from typing import Generator
from itertools import product
from dataclasses import dataclass
from omegaconf import OmegaConf, MISSING



@dataclass
class DatasetConfig:
    name: str = MISSING
    qas_data: str = MISSING
    qas_data_results: str = MISSING

@dataclass
class RetrievalConfig:
    encoder: str = MISSING
    mode: str = MISSING
    max_k: int = MISSING

@dataclass
class GenConfig:
    name: str = MISSING
    gen_id: str = MISSING

@dataclass
class Config:
    dataset: DatasetConfig = MISSING
    retrieval: RetrievalConfig = MISSING
    generator: GenConfig = MISSING

@dataclass
class ConfigSetSummary:
    encoders: list[str]
    modes: list[str]
    max_ks: list[int]

@dataclass
class ConfigSet:
    base_config: Config
    configs: Generator[Config, None, None]
    summary: ConfigSetSummary

def validate_config(config: Config):
    from models.encoders.ENC_factory import AVAILABLE_ENCODER
    from models.generators.GEN_factory import AVAILABLE_LVLM
    AVAILABLE_DATASETS = ['MMLongBench', 'LongDocURL']

    if config.dataset.name not in AVAILABLE_DATASETS:
        raise ValueError(f"Unkown dataset '{config.dataset.name}'. Use {', '.join(AVAILABLE_DATASETS)}")

    if config.retrieval.encoder not in AVAILABLE_ENCODER:
        raise ValueError(f"Unkown encoder '{config.retrieval.encoder}'. Use {', '.join(AVAILABLE_ENCODER)}")

    if config.retrieval.mode not in ['ViSAR', 'LateInteraction', 'LargestGap', 'ScoreCluster']:
        raise ValueError(f"Unkown mode '{config.retrieval.mode}'. Available: ViSAR, LateInteraction, LargestGap, ScoreCluster")
    
    if not isinstance(config.retrieval.max_k, int) or config.retrieval.max_k < 1:
        raise ValueError(f"Max_k of '{config.retrieval.max_k}' is forbiden. It must be positive integers")
    
    if config.generator.name not in AVAILABLE_LVLM:
        raise ValueError(f"Unkown generator '{config.generator.name}'. Use {', '.join(AVAILABLE_LVLM)}")
    
    return config

def load_configs(config_path, sweep, print_config=True) -> ConfigSet:
    cfg_dict = OmegaConf.load(config_path)

    if sweep:
        values = [
            cfg_dict['sweep']['retrieval']['max_k'],
            cfg_dict['sweep']['retrieval']['mode'],
            cfg_dict['sweep']['retrieval']['encoder'],
        ]
    else:
        values = [
            [cfg_dict['retrieval']['max_k']],
            [cfg_dict['retrieval']['mode']],
            [cfg_dict['retrieval']['encoder']]
        ]
    
    cfg_dict.pop("sweep", None)
    cfg_strc = OmegaConf.structured(Config)
    base_config: Config = OmegaConf.merge(cfg_dict, cfg_strc)

    summary = ConfigSetSummary(
        encoders = list(dict.fromkeys(encoder for encoder in values[2])),
        modes    = list(dict.fromkeys(mode    for mode    in values[1])),
        max_ks   = list(dict.fromkeys(max_k   for max_k   in values[0])),
    ) 

    # Ouputs sweep configs one at a time while printing the current configuration.
    def generator():    
        configs: list[Config] = []

        for max_k, mode, encoder in product(*values):
            new_config: Config = OmegaConf.merge({}, base_config)

            new_config.retrieval.encoder = encoder
            new_config.retrieval.mode = mode
            new_config.retrieval.max_k = max_k
            new_config.generator.gen_id = (
                f'{new_config.generator.name}__max_{max_k}'
            )

            configs.append(
                validate_config(new_config)
            )

        for i, conf in enumerate(configs):
            if print_config:
                dn = conf.dataset.name
                rm = conf.retrieval.mode
                re = conf.retrieval.encoder
                mk = conf.retrieval.max_k
                gn = conf.generator.name
                print(f'\nconfig ({i+1}/{len(configs)}): {dn} - {rm} - {re} - max_{mk} - {gn}\n')
            yield conf   

    return ConfigSet(
        base_config=base_config,
        configs=generator(),
        summary=summary
    )