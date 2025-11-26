from .una_datasets import UNAGen
from .usb_datasets import USBData

def get_dataset(args, device='cpu'):

    if args.dataset == 'una':
        return UNAGen(args.data_config_path, training_=args.training, device=device)
    elif args.dataset == 'usb':
        return USBData(args.data_config_path, training_=args.training)
    else:
        raise NotImplementedError(args.dataset)