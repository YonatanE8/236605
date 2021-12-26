import os

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from datetime import datetime
from AdvancedDL import LOGS_DIR
from torch.utils.data import DataLoader
from AdvancedDL.src.models.moco import MoCoV2
from AdvancedDL.src.training.logger import Logger
from AdvancedDL.src.models.resnet import resnet18, resnet50
from AdvancedDL.src.training.trainer import MoCoTrainer
from AdvancedDL.src.training.optimizer import OptimizerInitializer
from AdvancedDL.src.losses.losses import CrossEntropy, TopKAccuracy
from AdvancedDL.src.utils.defaults import Predictions, Labels, Targets
from AdvancedDL.src.data.datasets import imagenette_train_ds, imagenette_val_ds, imagenette_self_train_ds

import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Define the log dir
date = str(datetime.today()).split()[0]
experiment_name = f"MoCoV2_{date}"
log_dir = LOGS_DIR
# log_dir = '/mnt/walkure_pub/yonatane/logs/'
logs_dir = os.path.join(log_dir, experiment_name)
os.makedirs(logs_dir, exist_ok=True)

# Define the Datasets & Data loaders
num_workers = 4
pin_memory = True
batch_size = 4
self_train_dl = DataLoader(
    dataset=imagenette_self_train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
train_dl = DataLoader(
    dataset=imagenette_train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
val_dl = DataLoader(
    dataset=imagenette_val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

# Define the model
in_channels = 3
# encoder_builder = resnet50
encoder_builder = resnet18
queue_size = 65536
momentum = 0.999
temperature = 0.2
resnet_kwargs = {}
n_classes = 10
n_layers = 2
units_grow_rate = 1
l0_units = 2048
bias = True
activation = 'relu'
mlp_params = {
    'in_channels': 10,
    'n_layers': n_layers,
    'out_channels': 1,
    'l0_units': l0_units,
    'units_grow_rate': units_grow_rate,
    'bias': bias,
    'activation': activation,
}
self_training = True
model_params = {
    'in_channels': in_channels,
    'encoder_builder': encoder_builder,
    'queue_size': queue_size,
    'momentum': momentum,
    'temperature': temperature,
    'resnet_kwargs': resnet_kwargs,
    'mlp_params': mlp_params,
    'self_training': self_training,
}
model = MoCoV2(
    **model_params
)
model.to(device)

# Define the optimizer
optimizers_types = (torch.optim.AdamW,)
optimizers_params = (
    {
        'lr': 0.001,
        'weight_decay': 1e-4,
    },
)

schedulers_types = (
    (
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    ),
)
schedulers_params = (
    (
        {
            'mode': 'min',
            'factor': 0.1,
            'patience': 20,
            'threshold': 1e-4,
            'threshold_mode': 'rel',
            'cooldown': 0,
            'min_lr': 1e-6,
            'eps': 1e-8,
            'verbose': True,
        },
    ),
)

optimizer_init_params = {
    'optimizers_types': optimizers_types,
    'optimizers_params': optimizers_params,
    'schedulers_types': schedulers_types,
    'schedulers_params': schedulers_params,
}

optimizer = OptimizerInitializer(
    **optimizer_init_params
)

# Instantiate the model for the current fold optimizer
model_parameters = (model.parameters(),)
optimizer = optimizer.initialize(model_parameters)

# Define the loss & evaluation functions
loss_fn = CrossEntropy(
    predictions_key=Predictions,
    target_key=Labels,
)
evaluation_metric = CrossEntropy(
    predictions_key=Predictions,
    target_key=Labels,
)

# Define the logger
max_elements = 100
logger = Logger(
    log_dir=logs_dir,
    experiment_name="SelfTraining",
    max_elements=max_elements,
)

# Define the trainer
trainer = MoCoTrainer(
    model=model,
    loss_fn=loss_fn,
    evaluation_metric=evaluation_metric,
    optimizer=optimizer,
    logger=logger,
    device=device,
    self_training=True,
)

if __name__ == '__main__':
    print("Pre-training the model")
    num_epochs = 5
    checkpoints = True
    early_stopping = None
    checkpoints_mode = 'min'
    trainer.fit(
        dl_train=self_train_dl,
        dl_val=self_train_dl,
        num_epochs=num_epochs,
        checkpoints=checkpoints,
        checkpoints_mode=checkpoints_mode,
        early_stopping=early_stopping,
    )

    print("Fine-tuning the model")
    loss_fn = CrossEntropy(
        predictions_key=Predictions,
        target_key=Targets,
    )
    evaluation_metric = TopKAccuracy(
        k=1,
        num_classes=10,
    )
    trainer.set_loss_and_eval_criterions(
        loss_fn=loss_fn,
        eval_fn=evaluation_metric,
    )
    trainer.set_self_training(False)
    trainer.freeze_model()
    checkpoints_mode = 'max'
    trainer.fit(
        dl_train=train_dl,
        dl_val=val_dl,
        num_epochs=num_epochs,
        checkpoints=checkpoints,
        checkpoints_mode=checkpoints_mode,
        early_stopping=early_stopping,
    )

    print("Evaluating over the test set")
    model_params['self_training'] = False
    model = MoCoV2(
        **model_params
    )
    model_ckpt_path = f"{logs_dir}/BestModel.PyTorchModule"
    model_ckp = torch.load(model_ckpt_path)
    model.load_state_dict(model_ckp['model'])
    model.to(device)

    logger = Logger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        max_elements=max_elements,
    )
    trainer = MoCoTrainer(
        model=model,
        loss_fn=loss_fn,
        evaluation_metric=evaluation_metric,
        optimizer=optimizer,
        logger=logger,
        device=device,
    )
    trainer.evaluate(
        dl_test=val_dl,
        ignore_cap=True,
    )
