import torch


def noise_estimation_loss(model, x0: torch.Tensor, cond, t: torch.LongTensor, e: torch.Tensor, betas: torch.Tensor,
                          keepdim=False, loss_weight_flag=False):
    
    B, C, F = x0.shape[:3]
    alphas = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1, 1)
    xt = x0 * alphas.sqrt() + e * (1.0 - alphas).sqrt()

    model_output = model(xt, cond, t.float())

    if model_output.shape[1] == C:
        e_pred = model_output
        var_pred = None
    else:
        e_pred, var_pred = torch.split(model_output, C, dim=1)
        frozen_out = torch.cat([model_output.detach(), var_pred], dim=1)
    if keepdim:
        return (e - e_pred).square().mean(dim=(1, 2, 3, 4)) if not loss_weight_flag else (
                (e - e_pred).square() * cond.sigmoid()).mean(dim=(1, 2, 3, 4))
    else:
        return (e - e_pred).square().mean() if not loss_weight_flag else ((e - e_pred).square() * cond.sigmoid()).mean()
