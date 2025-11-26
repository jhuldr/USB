import torch
from tqdm import tqdm

def compute_alpha(betas, t):
    assert betas.device == t.device, print(betas.device, t.device)
    betas = torch.cat([torch.zeros(1).to(betas.device), betas], dim=0)
    alphat = (1 - betas).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1, 1)
    return alphat


def one_step_denoise(model, xt, cond, t, betas, t_next=None, eta=0.0):
    assert xt.shape[0] == t.shape[0]
    model.eval()
    with torch.no_grad():
        device = xt.device
        t, betas, t_next, cond = t.to(device), betas.to(device), t_next.to(
            device) if t_next is not None else None, cond.to(device) if cond is not None else None
        et_pred = model(xt, cond, t)

        alpha_t = compute_alpha(betas, t.long())
        x0_pred = (xt - et_pred * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

        if t_next is not None:
            alphat_next = compute_alpha(betas, t_next.long())
            c1 = eta * ((1 - alpha_t / alphat_next) * (1 - alphat_next) / (1 - alpha_t)).sqrt()
            c2 = ((1 - alphat_next) - c1 ** 2).sqrt()
            xt_next_pred = alphat_next.sqrt() * x0_pred + c1 * torch.randn_like(xt) + c2 * et_pred # sde
          
    if t_next is not None:
        return x0_pred, xt_next_pred
    else:
        return x0_pred


def denoise_uncond(x_model, y_model, xt, yt, t_list, betas, eta=0.0):
    assert xt.shape == yt.shape
    assert xt.device == yt.device
    x_model.eval()
    y_model.eval()
    with torch.no_grad():
        device = xt.device
        betas = betas.to(device)
        img_num = xt.shape[0]
        t_next_list = [-1] + list(t_list[:-1])
        x0s = [] 
        y0s = []
        xts = [xt] 
        yts = [yt]
        for i, j in tqdm(zip(reversed(t_list), reversed(t_next_list)), desc='denoising', total=len(t_list)):
            t = (torch.ones(img_num) * i).to(device)
            t_next = (torch.ones(img_num) * j).to(device)
            xt = xts[-1].to(device)
            yt = yts[-1].to(device)
            x0, xt = one_step_denoise(x_model, xt, yt, t, betas, t_next=t_next, eta=eta)

            x0s.append(x0)
            xts.append(xt)
    
            y0, yt = one_step_denoise(y_model, yt, x0, t, betas, t_next=t_next, eta=eta)

            y0s.append(y0)
            yts.append(yt)
    return xts, yts, x0s, y0s

def denoise_cond(y_model, x0, yt, t_list, betas, eta=0.0):
    assert x0.shape == yt.shape
    assert x0.device == yt.device

    y_model.eval()
    with torch.no_grad():
        device = x0.device
        betas = betas.to(device)
        img_num = x0.shape[0]
        t_next_list = [-1] + list(t_list[:-1])
 
        y0s = []
        yts = [yt]
        for i, j in tqdm(zip(reversed(t_list), reversed(t_next_list)), desc='denoising', total=len(t_list)):
            t = (torch.ones(img_num) * i).to(device)
            t_next = (torch.ones(img_num) * j).to(device)

            yt = yts[-1].to(device)
    
            y0, yt = one_step_denoise(y_model, yt, x0, t, betas, t_next=t_next, eta=eta)

            y0s.append(y0)
            yts.append(yt)
    return yts, y0s

def denoise_p2h(y_model, xt, yt, y0_ori, t_list, betas, eta=0.0,
                  alpha0=20.0, decay=0.5):

    assert xt.shape == yt.shape
    assert xt.device == yt.device

    y_model.eval()
    with torch.no_grad():
        device = xt.device
        betas = betas.to(device)
        img_num = xt.shape[0]
        t_next_list = [-1] + list(t_list[:-1])
        x0s = []
        xts = [xt]
        y0s = []
        yts = [yt]
        for i, j in tqdm(zip(reversed(t_list), reversed(t_next_list)), desc=f'p2h_denoising', total=len(t_list)):
            t = (torch.ones(img_num) * i).to(device)
            t_next = (torch.ones(img_num) * j).to(device)
            xt = xts[-1].to(device)
            yt = yts[-1].to(device)
            x0, xt = xt, xt

            x0s.append(x0)
            xts.append(xt)

            y0, yt = one_step_denoise(y_model, yt, x0, t, betas, t_next=t_next, eta=eta)

            alpha = alpha0 * torch.exp(-decay * t / len(t_list))
            lambda_map = torch.exp(-alpha * torch.abs(y0 - y0_ori))
            lambda_final = lambda_map

            y0 = y0 + lambda_final * (y0_ori - y0)
            yt = yt + lambda_final * (y0_ori - yt)

            y0s.append(y0)
            yts.append(yt)

    return xts, yts, x0s, y0s


def denoise_h2p(y_model, xt, yt, y0_ori, t_list, betas, eta=0.0,
                  alpha0=20.0, decay=0.5, lesion_free_scale=1.0):

    assert xt.shape == yt.shape
    assert xt.device == yt.device

    y_model.eval()
    with torch.no_grad():
        device = xt.device
        betas = betas.to(device)
        img_num = xt.shape[0]
        t_next_list = [-1] + list(t_list[:-1])
        x0s = []
        xts = [xt]
        y0s = []
        yts = [yt]
        for i, j in tqdm(zip(reversed(t_list), reversed(t_next_list)), desc=f'h2p_denoising', total=len(t_list)):
            t = (torch.ones(img_num) * i).to(device)
            t_next = (torch.ones(img_num) * j).to(device)
            xt = xts[-1].to(device)
            yt = yts[-1].to(device)
            x0, xt = xt, xt

            x0s.append(x0)
            xts.append(xt)

            y0, yt = one_step_denoise(y_model, yt, x0, t, betas, t_next=t_next, eta=eta)

            alpha = alpha0 * torch.exp(-decay * t / len(t_list))
            lambda_map = torch.exp(-alpha * torch.abs(y0 - y0_ori))

            mask_soft = xt 
            mask_soft = torch.clamp(mask_soft, 0.0, 1.0)

            mask_factor = 1 - lesion_free_scale * mask_soft

            lambda_final = lambda_map * mask_factor

            y0 = y0 + lambda_final * (y0_ori - y0)
            yt = yt + lambda_final * (y0_ori - yt)
            
            y0s.append(y0)
            yts.append(yt)

    return xts, yts, x0s, y0s