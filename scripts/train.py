import os
import time
import argparse
from omegaconf import OmegaConf
from Trainer.engine import Train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument("--mode", type=str, default="lesion",
                        help="Mode of operation: lesion, brain.")

    parser.add_argument("--config_path", type=str, default="cfgs/trainer/train/train.yaml")
    parser.add_argument("--data_file", type=str, default="experiment_data/train_healthy.txt")

    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--model_lesion_path", type=str, default=None)
    
    args = parser.parse_args()

    if args.mode == "brain" and (args.model_lesion_path is None):
        raise ValueError("In 'brain' mode, --model_lesion_path must be provided.")

    config = OmegaConf.load(args.config_path)
    config.train.mode = args.mode
    config.train.desc = f"model_{args.mode}"
    config.train.resume_path = args.resume_path
    config.train.model_lesion_path = args.model_lesion_path
    config.train.log_dir = os.path.join('.', 'log',
                      f"{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}_{config.train.desc}")
    
    config.data.data_file = args.data_file

    train_utils = Train(config)
    
    train_utils.train()


