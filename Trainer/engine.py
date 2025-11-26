import os
import time
import sys
import torch


from torch.optim import lr_scheduler
from torch.utils import data
from tqdm import tqdm

from datasets import USBData
from autoencoders import AutoencoderKLAD
from Trainer.models.ldm_3d import LDM_3D
from Trainer.visualizer import visualize_diffusion_batch
from utils.logging import setup_logging, print_configurations
from utils.get_betas import get_betas
from utils.get_optimizer import get_optimizer
from utils.loss import noise_estimation_loss
from utils.denoise import one_step_denoise



class Train():
    def __init__(self, args):
        # load configs
        
        self.model_config = args.model
        self.diffusion_config = args.diffusion
        self.optim_config = args.optim
        self.lr_scheduler_config = args.lr_scheduler

        self.train_config = args.train
        self.data_config = args.data

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.betas = get_betas(self.diffusion_config).to(self.device)
        self.num_diffusion_timesteps = self.diffusion_config.num_diffusion_timesteps


        # load VAE
        if not os.path.exists(self.model_config.vae_path):
            raise ValueError(f"Model not found at {self.model_config.vae_path}")
        self.vae = AutoencoderKLAD.load_from_checkpoint(self.model_config.vae_path, cfg=self.model_config.vae_config_path, input_dim=[(160, 160, 160)])
        self.vae.eval() 

        # Freeze VAE
        self.vae.requires_grad_(False)

        
    def train(self):
        """
        Trains the model based on the provided configuration.
        """
        data_config = self.data_config
        model_config = self.model_config
        optim_config = self.optim_config
        lr_scheduler_config = self.lr_scheduler_config
        train_config = self.train_config
        device = self.device
        betas = self.betas
        num_diffusion_timesteps = self.num_diffusion_timesteps
        mode = train_config.mode
        desc = train_config.desc
        min_loss = 1000

        # ============ preparing data ... ============
        dataset = USBData(data_config, training_=True, device=device)
        
        train_dataloader = data.DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True,
                                           num_workers=train_config.num_workers)
        
        # ============ building model ... ============
        
        model = LDM_3D(in_channels=model_config.in_channels,
                        out_channels=model_config.in_channels,
                        num_channels=model_config.num_channels,
                        attention_levels=model_config.attention_levels,
                        num_res_blocks=model_config.num_res_blocks,
                        num_head_channels=model_config.num_head_channels, 
                        with_conditioning=False,
                        learn_sigma=model_config.learn_sigma).to(device)
        
        model = torch.nn.DataParallel(model)


        # ============ preparing optimizer ... ============   
        optimizer = get_optimizer(optim_config, model.parameters())
        # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
        
        # ============ init schedulers ... ============ 
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_scheduler_config.factor,
                                                   patience=lr_scheduler_config.patience,
                                                   verbose=True, min_lr=1e-6) if lr_scheduler_config.type == 'ReduceLROnPlateau' else None

        start_epoch = 0
        if train_config.resume_path:
            y_states = torch.load(train_config.resume_path)
            model.load_state_dict(y_states[0])
            optimizer.load_state_dict(y_states[1])
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-6
            start_epoch = y_states[2]+1

        log_dir = os.path.join(train_config.log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = setup_logging(log_dir, "train_log.txt")
        print_configurations(model_config, data_config, self.diffusion_config, optim_config, lr_scheduler_config, train_config, device)

    
        if train_config.mode == 'brain':
            x_model_path = train_config.model_lesion_path
            
            x_model = LDM_3D(in_channels=model_config.in_channels,
                            out_channels=model_config.in_channels,
                            num_channels=model_config.num_channels,
                            attention_levels=model_config.attention_levels,
                            num_res_blocks=model_config.num_res_blocks,
                            num_head_channels=model_config.num_head_channels, 
                            with_conditioning=False,
                            learn_sigma=model_config.learn_sigma).to(device)
            x_model = torch.nn.DataParallel(x_model)
            x_states = torch.load(x_model_path)
            x_model.load_state_dict(x_states[0])
            x_model.eval()
            torch.save(x_states, os.path.join(log_dir, f"x_{train_config.x}_y_{train_config.y}_x.pth"))

        # ============ start training ... ============
        for epoch in range(start_epoch, start_epoch + train_config.n_epochs):
            epoch_loss = 0
            epoch_start_time = time.time()
            with tqdm(total=len(train_dataloader),
                    desc=f'training_epoch_{epoch}/{start_epoch + train_config.n_epochs}',
                    file=sys.__stdout__) as pbar:

                for i, batch in enumerate(train_dataloader):
                    x0 = batch[train_config.x]
                    x0_num = x0.shape[0]
                    y0 = batch[train_config.y]

                    x0 = x0.to(device)
                    y0 = y0.to(device)

                    self.vae = self.vae.to(device)

                    x0 = self.vae.encode([x0])[0]._sample().mul_(0.18215)
                    x0 = x0.to(device)

                    y0 = self.vae.encode([y0])[0]._sample().mul_(0.18215)
                    y0 = y0.to(device)

                    # latent injection
                    alpha = - 0.05
                    y0 = y0 + alpha * x0

                    model.train()

                    e_x = torch.randn_like(x0)
                    e_y = torch.randn_like(y0)

                    t = torch.randint(low=0, high=num_diffusion_timesteps, size=(x0_num // 2 + 1,)).to(device)
                    t = torch.cat([t, num_diffusion_timesteps - t - 1], dim=0)[:x0_num]
                    
                    a = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1, 1)
                    yt = y0 * a.sqrt() + e_y * (1.0 - a).sqrt()

                    if mode == 'lesion':
                        loss = noise_estimation_loss(model, x0, yt, t, e_x, betas)
                        x0_pred = None
                    elif mode == 'brain':
                        loss_weight_flag = train_config.loss_weight_flag
                        xt = x0 * a.sqrt() + e_x * (1.0 - a).sqrt()

                        x0_pred = one_step_denoise(x_model, xt, yt, t, betas)
                        
                        loss = noise_estimation_loss(model, y0, x0_pred, t, e_y, betas, loss_weight_flag=loss_weight_flag)

                    epoch_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                    pbar.update(1)

            epoch_time = time.time() - epoch_start_time
            print(
                f'epoch:{epoch},lr:{optimizer.param_groups[0]["lr"]},loss:{epoch_loss},time:{epoch_time}')
            
            if scheduler is not None:
                scheduler.step(epoch_loss)

            model_states = [model.state_dict(), optimizer.state_dict(), epoch]
            
            train_epoch = epoch - start_epoch
            
            if epoch_loss <= min_loss:
                best_states = model_states       
                min_loss = epoch_loss

                torch.save(best_states,
                            os.path.join(log_dir, f"best_epoch{best_states[2]}_loss{min_loss:.4f}_{desc}.pth"))

            if train_epoch % train_config.save_freq == 0 or epoch_loss <= min_loss:
                torch.save(model_states,
                            os.path.join(log_dir, f"ckpt_epoch{epoch}_loss{epoch_loss:.4f}_{desc}.pth"))
            
                        
            if train_config.visualize:
                print('======= visualizing ========')
                visualize_diffusion_batch(
                    x0, y0, a, e_x, e_y, t, betas,
                    self.vae, batch,
                    epoch, log_dir, train_config, device,
                    model, x0_pred
                )
                           
