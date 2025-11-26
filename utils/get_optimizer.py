from torch import optim


def get_optimizer(optim_config, parameters):
    if optim_config.optimizer == 'Adam':
        return optim.Adam(parameters, lr=optim_config.lr, weight_decay=optim_config.weight_decay,
                          betas=(optim_config.beta1, 0.999), amsgrad=optim_config.amsgrad, eps=optim_config.eps)
    elif optim_config.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=optim_config.lr, weight_decay=optim_config.weight_decay)
    elif optim_config.optimizer == 'SGD':
        return optim.SGD(parameters, lr=optim_config.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(optim_config.optimizer))

