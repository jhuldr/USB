# constants file
MODEL_AEKL = 'AutoencoderKL'
MODEL_AEKLAD = 'AutoencoderKLAD'
MODEL_VAEBASE = 'vaebase'
MODEL_CVAEBASE = 'cvaebase'
MODELS = [
            MODEL_AEKL,
            MODEL_AEKLAD,
            MODEL_VAEBASE,
            MODEL_CVAEBASE
        ]

VARIATIONAL_MODELS = []
CONFIG_KEYS = [
            "model",
            "datamodule",
            "encoder",
            "decoder",
            "generator", #added for brainpass code
            "prior",
            "trainer",
            "callbacks",
            "logger",
        ]

KLAD_CONFIG_KEYS = [
            "model",
            "datamodule",
            "encoder",
            "decoder",
            "prior",
            "generator", #added for brainpass code
            "trainer",
            "callbacks",
            "logger",
            "n_discriminator",
        ]   

LDM_CONFIG_KEYS = [
            "model",    
            "datamodule",
            "trainer",
            "callbacks",
            "logger"
            ]