import os
import sys
from omegaconf import OmegaConf

class Tee(object):
    def __init__(self, name, mode="a"):
        self.file = open(name, mode, buffering=1)
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

def setup_logging(log_dir, log_filename="log.txt"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    
    sys.stdout = Tee(log_path, "w")
    sys.stderr = sys.stdout
    return log_path

def print_configurations(model_config, data_config, diffusion_config,
                         optim_config, lr_scheduler_config, train_config, device):
    """
    Pretty-print all configurations at the start of training.
    Supports OmegaConf objects and normal dicts.
    """
    def _print_block(title, cfg):
        print("\n" + "=" * 60)
        print(f"{title}")
        print("=" * 60)
        try:
            # OmegaConf structured config / dict printing
            print(OmegaConf.to_yaml(cfg))
        except Exception:
            print(cfg)

    print("\n" + "#" * 60)
    print("#                TRAINING CONFIGURATIONS")
    print("#" * 60)

    print(f"Device: {device}")

    _print_block("Model Config", model_config)
    _print_block("Data Config", data_config)
    _print_block("Diffusion Config", diffusion_config)
    _print_block("Optimizer Config", optim_config)
    _print_block("LR Scheduler Config", lr_scheduler_config)
    _print_block("Train Config", train_config)

    print("#" * 60 + "\n")