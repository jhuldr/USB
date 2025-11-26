import argparse
from omegaconf import OmegaConf
from utils.test_utils import Test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument("--mode", type=str, default="uncond_gen",
                        help="Mode of operation: uncond_gen, cond_gen, p2h_edit, h2p_edit.")

    parser.add_argument("--config_path", type=str, default="cfgs/trainor/test/test.yaml")

    args = parser.parse_args()


    config = OmegaConf.load(args.config_path)
    config.mode = args.mode
    
    test_utils = Test(config)

    if config.mode == "uncond_gen":
        test_utils.uncond_gen()
    elif config.mode == "cond_gen":
        test_utils.cond_gen()
    elif config.mode == "p2h_edit":
        test_utils.p2h_edit()
    elif config.mode == "h2p_edit":
        test_utils.h2p_edit()
    else:
        raise ValueError(f"Unknown mode: {config.mode}")

