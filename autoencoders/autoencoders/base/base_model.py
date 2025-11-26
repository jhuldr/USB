# implementation of autoencoders based on 

import os
import numpy as np
import hydra
import re
import collections.abc

import torch
import pytorch_lightning as pl

from os.path import join, isdir, exists
from datetime import datetime
from abc import ABC, abstractmethod
from hydra import compose, initialize, initialize_config_dir
from schema import Schema, SchemaError, And, Or

from torch.utils.data.dataloader import DataLoader
from omegaconf import OmegaConf, open_dict
from .constants import *
from .validation import config_schema
from .exceptions import *
import gc
from torch.cuda.amp import GradScaler

def update_dict(d, u, l):
    for k, v in u.items():
        if k in l:
            if isinstance(v, collections.abc.Mapping):
                d[k] = update_dict(d.get(k, {}), v, l=v.keys())
            else:
                d[k] = v
    return d

class BaseModelAE(ABC, pl.LightningModule):
    """Base class for autoencoder models.
    Args:
        model_name (str): Type of autoencoder model.
        cfg (str): Path to configuration file.
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions.
    """
    is_variational = False

    @abstractmethod
    def __init__(
        self,
        model_name = None,
        cfg = None,
        input_dim = None,
        z_dim = None,
    ):

        assert (model_name is not None and model_name in MODELS), \
        "Model name is invalid"  # have to choose which model always

        try:
            # check input_dim
            Schema(And(list, lambda x: len(x) > 0), error="input_dim should not be empty").validate(input_dim)
            Schema([Or(int, tuple)], error="input_dim should be a list of int/tuple").validate(input_dim)
            for d in input_dim:
                if isinstance(d, int):
                    Schema(lambda x: x > 0, error="each dim should be > 0").validate(d)
                else:
                    Schema(lambda x: all(a > 0 for a in x), error="each dim should be > 0").validate(d)
                    # Schema(lambda x: len(x) in [1,2,3], error="each dim should be 1D or 2D").validate(d)

            # check z_dim
            if z_dim is not None:
                Schema(And(int, lambda x: x > 0), error="z_dim must be > 0").validate(z_dim)
        except SchemaError as se:
            raise ConfigError(se) from None

        super().__init__()
        self.model_name = model_name
        self.input_dim = input_dim
        self.n_views = len(self.input_dim)
        self.z_dim = z_dim
        
        with initialize(version_base=None, config_path="../configs"):
            def_cfg = compose(
                            config_name="default",
                            return_hydra_config=True,
                            overrides=[f"model_type={self.model_name}.yaml"]
                        )

        user_cfg = None
        if cfg is not None: # user overrides default config
            if os.path.isabs(cfg):
                cfgdir, cfg_file = os.path.split(cfg)
                with initialize_config_dir(version_base=None, config_dir=cfgdir):
                    user_cfg = compose(
                                config_name=cfg_file,
                                return_hydra_config=True
                            )
            else:
                workdir = os.getcwd()
                with initialize_config_dir(version_base=None, config_dir=workdir):
                    user_cfg = compose(
                                config_name=cfg,
                                return_hydra_config=True
                            )

    
        def_cfg = self.__updateconfig(def_cfg, user_cfg)

        if self.z_dim is not None:   # overrides hydra config... passed arg has precedence
            def_cfg.model.z_dim = self.z_dim


        # validate model configuration
        self.cfg = self.__checkconfig(def_cfg)

        self.__dict__.update(self.cfg.model)

        print("MODEL: ", self.model_name)

        if all(k in self.cfg.model for k in ["seed_everything", "seed"]):
            pl.seed_everything(self.cfg.model.seed, workers=True)

        self._setencoders()
        self._setdecoders()
        self._setprior()

        self.save_hyperparameters()
        self.automatic_optimization = False
        self._training = False
    ################################            public methods
    def fit(self, *data, labels=None, max_epochs=None, batch_size=None, val_data=None):

        self.create_folder(self.cfg.out_dir)
        self.save_config()
        data = list(data)

        self._training = True
        if max_epochs is not None:
            self.max_epochs = max_epochs
            self.cfg.trainer.max_epochs = max_epochs
        else:
            self.max_epochs = self.cfg.trainer.max_epochs

        if batch_size is not None:
            self.batch_size = batch_size
            self.cfg.datamodule.batch_size = batch_size
        else:
            self.batch_size = self.cfg.datamodule.batch_size

        callbacks = []
        if self.cfg.datamodule.is_validate:
            for _, cb_conf in self.cfg.callbacks.items():
                callbacks.append(hydra.utils.instantiate(cb_conf))

        logger = hydra.utils.instantiate(self.cfg.logger)

        # NOTE: have to check file exists otherwise error raised
        if (self.cfg.model.ckpt_path is None) or \
            (not exists(self.cfg.model.ckpt_path)):
            self.cfg.model.ckpt_path = None

        else:
            print('resuming training from checkpoint: ', self.cfg.model.ckpt_path)

        py_trainer = hydra.utils.instantiate(
        self.cfg.trainer, callbacks=callbacks, logger=logger,)
            
        datamodule = hydra.utils.instantiate(
           self.cfg.datamodule, data=data, labels=labels, val_data=val_data, generator=self.cfg.generator, _recursive_=False 
        )
        py_trainer.fit(self, datamodule, ckpt_path=self.cfg.model.ckpt_path)

    def predict_latents(self, *data, labels=None, batch_size=None, path=None):
        if path is not None:
            self.cfg.datamodule.dataset.data_dir = path
        return self.__predict(*data, labels=labels, batch_size=batch_size)

    def predict_reconstruction(self, *data, labels=None, batch_size=None, path=None):
        if path is not None:
            self.cfg.datamodule.dataset.data_dir = path
        return self.__predict(*data, labels=labels, batch_size=batch_size, is_recon=True)

    def print_config(self, cfg=None, keys=None):
        if cfg is None:
            cfg = self.cfg

        if keys is not None:
            print(f"{'model_name'}:\n  {cfg['model_name']}")
            for k in keys:
                if k in CONFIG_KEYS:
                    if cfg[k] is not None:
                        str = (OmegaConf.to_yaml(cfg[k])).replace("\n", "\n  ")
                    else:
                        str = "null\n"
                    print(f"{k}:\n  {str}")
        else:
            self.print_config(cfg=cfg, keys=CONFIG_KEYS)

    def save_config(self, keys=None):
        run_time = datetime.now().strftime("%Y-%m-%d_%H%M")
        save_cfg = {}
        if keys is not None:
            for k in keys:
                if k in CONFIG_KEYS:
                    save_cfg[k] = self.cfg[k]
            OmegaConf.save(save_cfg, join(self.cfg.out_dir, 'config_{0}.yaml'.format(run_time)))
        else:
            self.save_config(keys=CONFIG_KEYS)

    def create_folder(self, dir_path):
        check_folder = isdir(dir_path)
        if not check_folder:
            os.makedirs(dir_path, exist_ok=True)

    ################################            abstract methods

    @abstractmethod
    def encode(self, x):
        raise NotImplementedError()

    @abstractmethod
    def decode(self, z):
        raise NotImplementedError()

    @abstractmethod
    def loss_function(self, x, fwd_rtn):
        raise NotImplementedError()

    ################################            LightningModule methods
    @abstractmethod
    def forward(self, x):
         raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx, stage="val")

    def on_train_epoch_end(self):
        self.trainer.save_checkpoint(join(self.cfg.out_dir, "last.ckpt"))
        print("current device: ", self.device)
        print("current time: ", datetime.now())
        gc.collect()
        
    def on_train_end(self):
        self.trainer.save_checkpoint(join(self.cfg.out_dir, "model.ckpt"))
        torch.save(self, join(self.cfg.out_dir, "model.pkl"))

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        return self.optimizer

    ################################            protected methods, can be overwritten by child
    def _setencoders(self):
        self.encoders = torch.nn.ModuleList(
            [
                hydra.utils.instantiate(
                    eval(f"self.cfg.encoder.enc{i}"),
                    input_dim=d,
                    z_dim=self.z_dim,
                    _recursive_=False,
                    _convert_ = "all"
                )
                for i, d in enumerate(self.input_dim)
            ]
        )

    def _setdecoders(self):
        if hasattr(self, "num_res_blocks") and isinstance(self.cfg.decoder.default.num_res_blocks, int): #TODO: tidy this up
            self.cfg.decoder.default.num_res_blocks = (self.num_res_blocks) * len(self.num_channels)
            self.cfg.decoder.dec0.num_res_blocks = (self.num_res_blocks) * len(self.num_channels)

        self.decoders = torch.nn.ModuleList(
            [
                hydra.utils.instantiate(
                    eval(f"self.cfg.decoder.dec{i}"),
                    input_dim=d,
                    z_dim=self.z_dim,
                    _recursive_=False,
                    _convert_ = "all"
                )
                for i, d in enumerate(self.input_dim)
            ]
        )

    def _setprior(self):
        if self.model_name not in VARIATIONAL_MODELS or \
            (self.model_name in VARIATIONAL_MODELS and not self.sparse):
            self.prior = hydra.utils.instantiate(self.cfg.prior)

    def _unpack_batch(self, batch): # dataset returned other vars than x, need to unpack
        if isinstance(batch[0], list): 
            batch_x, batch_y, *other = batch
        else: 
            batch_x, batch_y, other = batch, None, None
        return batch_x, batch_y, other 

    def _set_batch_labels(self, labels): # for passing labels to encoder/decoder
        if hasattr(self, "conditional"): 
            if self.conditional: 
                self.labels = labels

        for i in range(len(self.encoders)):
            if hasattr(self, "conditional") and self.conditional:
                self.encoders[i].set_labels(labels)

        for i in range(len(self.decoders)):
            if hasattr(self, "conditional") and self.conditional:
                self.decoders[i].set_labels(labels)



    ################################            private methods
    def __updateconfig(self, orig, update):
        print("model name: ", self.model_name)
        if self.model_name in ["AutoencoderKLAD"]:
            config_keys = KLAD_CONFIG_KEYS
        else:
            config_keys = CONFIG_KEYS
        OmegaConf.set_struct(orig, True)
        with open_dict(orig):
            # update default cfg with user config
            if update is not None:
                update_keys = list(set(update.keys()) & set(config_keys))
      
                orig = update_dict(orig, update, l=update_keys)
             

            # update encoder/decoder config
            for i, d in enumerate(self.input_dim):
                enc_key = f"enc{i}"
                if enc_key not in orig.encoder.keys():
                    if update is not None and "encoder" in update.keys() and \
                        enc_key in update.encoder.keys(): # use user-defined
                        orig.encoder[enc_key] = update.encoder[enc_key].copy()
                    else: # use default
                        orig.encoder[enc_key] = orig.encoder.default.copy()

                dec_key = f"dec{i}"
                if dec_key not in orig.decoder.keys():
                    if update is not None and "decoder" in update.keys() and \
                        dec_key in update.decoder.keys(): # use user-defined
                        orig.decoder[dec_key] = update.decoder[dec_key].copy()
                    else: # use default
                        orig.decoder[dec_key] = orig.decoder.default.copy()
        if update is not None and update.get('out_dir'):
            orig['out_dir'] = update['out_dir']
        return orig

    def __checkconfig(self, cfg):

        cfg_dict = OmegaConf.to_container(cfg)

        try:
            cfg_dict = config_schema.validate(cfg_dict)
        except SchemaError as se:
            raise ConfigError(se) from None
            

        pattern = re.compile(r'autoencoders\.architectures\..*\.Decoder')


        if self.model_name in VARIATIONAL_MODELS:
            self.is_variational = True

            pattern = re.compile(r'autoencoders\.architectures\..*\.VariationalEncoder')
            for k in cfg.encoder.keys():
                if not bool(pattern.match(eval(f"cfg.encoder.{k}._target_"))):
                    raise ConfigError(f"{k}: must use variational encoder for variational models")

                if cfg.prior._target_ != eval(f"cfg.encoder.{k}.enc_dist._target_"):
                    raise ConfigError('Encoder and prior must have the same distribution for variational models')

        if cfg.prior._target_ == "autoencoders.base.distributions.Normal":
            if not isinstance(cfg.prior.loc, (int, float)):
                raise ConfigError("loc must be int/float for Normal prior dist")

            if not isinstance(cfg.prior.scale, (int, float)):
                raise ConfigError("scale must be int/float for Normal prior dist")

        else:   # MultivariateNormal
            if isinstance(cfg.prior.loc, (int, float)):
                cfg.prior.loc = [cfg.prior.loc] * cfg.model.z_dim

            if isinstance(cfg.prior.scale, (int, float)):
                cfg.prior.scale = [cfg.prior.scale] * cfg.model.z_dim

            if  len(cfg.prior.loc) != len(cfg.prior.scale):
                raise ConfigError("loc and scale must have the same length for MultivariateNormal prior dist")

            if len(cfg.prior.loc) != cfg.model.z_dim:
                raise ConfigError("loc and scale must have the same length as z_dim for MultivariateNormal prior dist")

        return cfg

    def __step(self, batch, batch_idx, stage):
        batch_x, batch_y, other = self._unpack_batch(batch)
        
        self._set_batch_labels(batch_y)
        if stage == "train":
            fwd_return = self.forward(batch_x)
            loss = self.loss_function(batch_x, fwd_return)
            self.optimizer.zero_grad()
            loss["loss"].backward()
            # Update weights
            self.optimizer.step()
        elif stage == "val":
            with torch.no_grad():
                self.eval()
                fwd_return = self.forward(batch_x)
                loss = self.loss_function(batch_x, fwd_return)
        for loss_n, loss_val in loss.items():
            self.log(
                f"{stage}_{loss_n}", loss_val, on_epoch=True, prog_bar=True, logger=True
            )
        torch.cuda.empty_cache() 
        return loss["loss"]


    def __predict(self, *data, labels=None, batch_size=None, is_recon=False):
        self._training = False

        data = list(data)

        dataset = hydra.utils.instantiate(self.cfg.datamodule.dataset, data, labels=None)
        
        if batch_size is None:
         #   batch_size = data[0].shape[0]v
            batch_size = len(data[0])

        generator = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            z_ = None
            for batch_idx, local_batch in enumerate(generator):
                local_batch = [
                    local_batch_.to(self.device) for local_batch_ in local_batch
                ]
                  
                z = self.encode(local_batch)
                if self.sparse:
                    z = self.apply_threshold(z)
                if is_recon:
                    z = self.decode(z)

                z = [
                        [ d__._sample().cpu().detach().numpy() for d__ in d_ ]
                        if isinstance(d_, (list))
                        else
                        (d_.cpu().detach().numpy() if isinstance(d_, torch.Tensor)
                        else d_._sample().cpu().detach().numpy())
                        for d_ in z
                    ]

                if z_ is not None:
                    z_ = [
                            [ np.append(d_, d, axis=0) for d_, d in zip(p_,p) ]
                            if isinstance(p_, list) else np.append(p_, p, axis=0)
                            for p_, p in zip(z_, z)
                         ]
                else:
                    z_ = z
        return z_



################################################################################
class BaseModelVAE(BaseModelAE):
    """Base class for variational autoencoder models. Inherits from BaseModelAE.
    Args:
        model_name (str): Type of autoencoder model.
        cfg (str): Path to configuration file.
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions.
    """
    @abstractmethod
    def __init__(
        self,
        model_name = None,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):

        super().__init__(model_name=model_name,
                cfg=cfg,
                input_dim=input_dim,
                z_dim=z_dim)
    

    ################################            class methods
    def apply_threshold(self, z):
        """
        Implementation from: https://github.com/ggbioing/mcvae
        """

        keep = (self.__dropout() < self.threshold).squeeze().cpu()
        z_keep = []

        for _ in z:
            _ = _._sample()
            _[:, ~keep] = 0
            d = hydra.utils.instantiate(   
                self.cfg.encoder.default.enc_dist, loc=_, scale=1
            )
            z_keep.append(d)
            del _

        return z_keep

    ################################            protected methods
    def _setencoders(self):

        if self.sparse and self.threshold != 0.:
            self.log_alpha = torch.nn.Parameter(
                torch.FloatTensor(1, self.z_dim).normal_(0, 0.01)
            )
        else:
            self.sparse = False
            self.log_alpha = None
        # if self.num_res_blocks exists and is int, convert to list of length num_channels
        if hasattr(self, "num_res_blocks") and isinstance(self.num_res_blocks, int): #TODO: tidy this up
            self.num_res_blocks = [self.num_res_blocks] * len(self.input_dim)
            self.cfg.encoder.default.num_res_blocks = (self.num_res_blocks) * len(self.num_channels)
            self.cfg.encoder.enc0.num_res_blocks = (self.num_res_blocks) * len(self.num_channels)

        self.encoders = torch.nn.ModuleList(
            [
                hydra.utils.instantiate(
                    eval(f"self.cfg.encoder.enc{i}"),
                    input_dim=d,
                    z_dim=self.z_dim,
                    sparse=self.sparse,
                    log_alpha=self.log_alpha,
                    _recursive_=False,
                    _convert_="all"
                )
                for i, d in enumerate(self.input_dim)
            ]
        )

    ################################            private methods
    def __dropout(self):
        """
        Implementation from: https://github.com/ggbioing/mcvae
        """
        alpha = torch.exp(self.log_alpha.detach())
        return alpha / (alpha + 1)


class BaseModelVAEAD(BaseModelVAE):
    """Base class for convolutional VAE with adversarial loss. Inherits from BaseModelVAE.
    Args:
        model_name (str): Type of autoencoder model.
        cfg (str): Path to configuration file.
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions.
    """
    @abstractmethod
    def __init__(
        self,
        model_name = None,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):
        #if input_dim is list of lists, convert to list of tuples
        if isinstance(input_dim[0], list):
            input_dim = [tuple(d) for d in input_dim]
        
        super().__init__(model_name=model_name,
                        cfg=cfg,
                        input_dim=input_dim,
                        z_dim=z_dim)
        #clear torch cache
        #torch.cuda.empty_cache()
        self.automatic_optimization = False
        self.scaler_g = GradScaler()
        self.scaler_d = GradScaler()

    ################################        unused abstract methods
    def loss_function(self, x, fwd_rtn):
        pass

    def forward(self, x):
        pass

    ################################            LightningModule methods
    def training_step(self, batch, batch_idx):
        loss = self.__optimise_batch(batch)
        for loss_n, loss_val in loss.items():
            self.log(
                f"train_{loss_n}", loss_val, on_epoch=True, prog_bar=True, logger=True
            )
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        loss = self.__validate_batch(batch)
        for loss_n, loss_val in loss.items():
            self.log(
                f"val_{loss_n}", loss_val, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
            )
        return loss["loss"]

    def configure_optimizers(self):
        # Combine parameters from all encoders and decoders
        encoder_decoder_params = []
        for i in range(self.n_views):
            encoder_decoder_params += list(self.encoders[i].parameters())
            encoder_decoder_params += list(self.decoders[i].parameters())

        # Create one optimizer for all encoders and decoders
        encoder_decoder_optimizer = torch.optim.Adam(
            encoder_decoder_params, lr=self.learning_rate
        )

        # Create another optimizer for the discriminator
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.disc_learning_rate  # Custom learning rate for discriminator
        )

        # Return the two optimizers
        return [encoder_decoder_optimizer, discriminator_optimizer]

    ################################            private methods
    def __optimise_batch(self, local_batch):
        opts = self.optimizers()
        gen_opt = opts[0]
        disc_opt = opts[-1] 

        gen_opt.zero_grad()

        fwd_return = self.forward_recon(local_batch)
        recon_losses = self.recon_loss(local_batch, fwd_return)
        loss_gen = self.generator_loss(fwd_return["px_z"])
        recon_total = recon_losses["kl"] - recon_losses["ll"] + recon_losses["p_loss"] + loss_gen

        
        #self.manual_backward(recon_total) 
        self.scaler_g.scale(recon_total).backward()
        self.scaler_g.unscale_(gen_opt)
        encoder_decoder_params = []
        for i in range(self.n_views):
            encoder_decoder_params += list(self.encoders[i].parameters())
            encoder_decoder_params += list(self.decoders[i].parameters())
        torch.nn.utils.clip_grad_norm_(encoder_decoder_params, 1)
        self.scaler_g.step(gen_opt)
        self.scaler_g.update()

        disc_opt.zero_grad()
        loss_disc = self.discriminator_loss(local_batch, fwd_return["px_z"])

       #self.manual_backward(loss_disc)
       #disc_opt.step()
        self.scaler_d.scale(loss_disc).backward()
        self.scaler_d.unscale_(disc_opt)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1)
        self.scaler_d.step(disc_opt)
        self.scaler_d.update()

        loss_total = recon_total + loss_disc 
        loss = {
            "loss": loss_total,
            "recon": recon_total,
            "kl": recon_losses["kl"], 
            "ll": recon_losses["ll"], 
            "perceptual": recon_losses["p_loss"], 
            "disc": loss_disc,
            "gen": loss_gen,
        }
        return loss

    def __validate_batch(self, local_batch):
        with torch.no_grad():
            self.eval()
            fwd_return = self.forward_recon(local_batch)
            recon_losses = self.recon_loss(local_batch, fwd_return)  
            loss_gen = self.generator_loss(fwd_return["px_z"])
            recon_total = recon_losses["kl"] - recon_losses["ll"] + recon_losses["p_loss"] + loss_gen
            loss_disc = self.discriminator_loss(local_batch, fwd_return["px_z"])
            loss_total = recon_total + loss_disc
            loss = {
                "loss": loss_total,
                "recon": recon_total,
                "kl": recon_losses["kl"], 
                "ll": recon_losses["ll"], 
                "perceptual": recon_losses["p_loss"], 
                "disc": loss_disc,
                "gen": loss_gen,
            }
        return loss
   
class BaseModelVAEBASE(BaseModelVAE):
    """Base class for vae base model"""
    @abstractmethod
    def __init__(
        self,
        model_name = None,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):
        #if input_dim is list of lists, convert to list of tuples
        if isinstance(input_dim[0], list):
            input_dim = [tuple(d) for d in input_dim]
        
        super().__init__(model_name=model_name,
                        cfg=cfg,
                        input_dim=input_dim,
                        z_dim=z_dim)
        #clear torch cache
        #torch.cuda.empty_cache()
        self.automatic_optimization = False
    