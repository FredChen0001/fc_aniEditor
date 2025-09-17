import argparse, os, sys, datetime
import inspect
import shutil
from omegaconf import OmegaConf
import pprint
from torch.utils.data import DataLoader
def gen_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-g",
        "--cfg",
        type=str,
    )
    return parser

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    parser = gen_parser()
    opt, _ = parser.parse_known_args()
    cfg_file = os.path.split(opt.cfg)[-1]
    cfg_name = os.path.splitext(cfg_file)[0]
    tag = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + cfg_name
    logdir = os.path.join('logs', tag)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.join(logdir,'train'), exist_ok=True)
    os.makedirs(os.path.join(logdir,'val'), exist_ok=True)
    OmegaConf.register_new_resolver('logdir', lambda x: os.path.join(logdir, x))
    config = OmegaConf.load(opt.cfg)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.devices
    if int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]:
        os.makedirs(os.path.join(logdir, 'config'), exist_ok=True)
        shutil.copy(opt.cfg, os.path.join(logdir, 'config', cfg_file))
        print("\n" + "="*60)
        print("CONFIGURATION DETAILS")
        print("="*60)
        pprint.pprint(config,
                      width=80,
                      indent=2,
                      depth=3,  # Adjust depth as needed for your config structure
                      compact=False)  # Set to False for more readable multi-line output
        print("="*60 + "\n")
    from src.utils.train_utils import instantiate_from_config, set_seeds
    set_seeds(42)
    dataset_train = instantiate_from_config(config.datasets.training)
    data_loader_train = DataLoader(dataset_train, batch_size=config.datasets.training.batch_size, shuffle=True,
                                   num_workers=config.datasets.training.workers, drop_last=True)
    dataset_val = instantiate_from_config(config.datasets.validation)
    data_loader_val = DataLoader(dataset_val, batch_size=config.datasets.validation.batch_size, shuffle=False,
                                 num_workers=config.datasets.validation.workers, drop_last=True)
    if int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]:
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"Training set: {len(dataset_train):,} samples")
        print(f"Validation set: {len(dataset_val):,} samples")
        print("="*50 + "\n")
    trainer = instantiate_from_config(config.model_trainer)
    module_file = inspect.getfile(trainer.__class__)
    shutil.copy(module_file, os.path.join(logdir, os.path.basename(module_file)))
    trainer.execute_training(data_loader_train, data_loader_val, config.training_parameters, logdir)