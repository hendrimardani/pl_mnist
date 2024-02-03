# import torch
# from torch import nn
# # lightning.pytorch as pl is latest versions
# import lightning.pytorch as pl
# from lightning.pytorch.callbacks import Callback
# from torch.utils.data import DataLoader, random_split
# from torch.nn import functional as F
# from torchvision.datasets import MNIST
# from torchvision import datasets, transforms
# import os
# import torchvision
# from torchmetrics.functional import accuracy
# from lightning.pytorch.loggers import CSVLogger


# INPUT_FEATURES = 28*28
# OUTPUT_CLASS = 10

# class Model(pl.LightningModule):
#   def __init__(self, lr=1e-6):
#     super(Model, self).__init__()
  
#     # Declarate for training and validation value
#     self.lr = lr
#     self.train_step_outputs = []
#     self.validation_step_outputs = []
#     self.train_step_epochs = []
#     self.validation_step_epochs = []
#     self.train_step_mean = 0
#     self.validation_step_mean = 0

#     self.layer_1 = nn.Linear(INPUT_FEATURES, 128)
#     self.layer_2 = nn.Linear(128, 256)
#     self.layer_3 = nn.Linear(256, OUTPUT_CLASS)

#   def forward(self, x):
#     """ This is input model """
#     # e.g torch.Size(32, 1, 28, 28)
#     batch, channels, width, height = x.size()
#     # (b, 1, 28, 28) -> (b, 1*28*28)
#     # result torch.size(784, 1)
#     x = x.view(batch, -1)
#     # layer 1 (b, 1*28*28) -> (b, 128)
#     x = F.relu(self.layer_1(x))
#     x = F.relu(self.layer_2(x))
#     x = self.layer_3(x)

#     return x

#   def cross_entropy_loss(self, logits, labels):
#     return F.cross_entropy(logits, labels)

#   def training_step(self, train_batch, batch_idx):
#     x, y = train_batch
#     logits = self(x)
#     loss = self.cross_entropy_loss(logits, y)
#     self.train_step_outputs.append(loss)
#     self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
#     return loss

#   def validation_step(self, val_batch, batch_idx):
#     x, y = val_batch
#     loss, acc = self._shared_eval_step(val_batch, batch_idx)
#     # sample_imgs = x[:6]
#     # grid = torchvision.utils.make_grid(sample_imgs)
#     # self.logger.experiment.add_image('example_images', grid, 0)
#     metrics = {"val_acc": acc, "val_loss": loss}
#     self.validation_step_outputs.append(loss)
#     self.log_dict(metrics, sync_dist=True, enable_graph=True)
#     return metrics

#   def _shared_eval_step(self, batch, batch_idx):
#     x, y = batch
#     y_hat = self(x)
#     loss = F.cross_entropy(y_hat, y)
#     acc = accuracy(y_hat, y, task="multiclass", num_classes=10)
#     return loss, acc

#   def test_step(self, batch, batch_idx):
#     loss, acc = self._shared_eval_step(batch, batch_idx)
#     metrics = {"test_acc": acc, "test_loss": loss}
#     self.log_dict(metrics)
#     return metrics

#   def configure_optimizers(self):
#     optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
#     return optimizer

#   def setup(self, stage):
#     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081))])

#     # prepare transforms standard to MNIST
#     self.mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
#     self.mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)

#   def train_dataloader(self):
#     return DataLoader(self.mnist_train, batch_size=16, num_workers=4)

#   def test_dataloader(self):
#     return DataLoader(self.mnist_test, batch_size=16, num_workers=4)

#   def val_dataloader(self):
#     return DataLoader(self.mnist_test, batch_size=16, num_workers=4)



# '''
#   Link Documentations
#   https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/callbacks/callback.html#Callback.on_train_batch_end

#   Examples:
#   class MyCallbacks(Callback):
#   def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
#         """Called when the train epoch begins."""
# '''
# class MyCallbacks(Callback):
#   def on_train_epoch_end(self, trainer, pl_module):
#     all_preds = torch.stack(pl_module.train_step_outputs).mean()
#     pl_module.train_step_mean = all_preds
#     pl_module.train_step_epochs.append(all_preds)

#   def on_validation_epoch_end(self, trainer, pl_module):
#     all_preds = torch.stack(pl_module.validation_step_outputs).mean()
#     pl_module.validation_step_mean = all_preds
#     pl_module.validation_step_epochs.append(all_preds)


# model = Model()
# callbacks = MyCallbacks()

# trainer = pl.Trainer(
#   callbacks=[callbacks],
#   precision='bf16-mixed',
#   benchmark=True,
#   accelerator="auto",
#   devices=7,
#   # Write the result training to CSV file
#   logger=CSVLogger("logs", "model"),
#   max_epochs=50,
#   enable_progress_bar=True,
#   fast_dev_run=False
# )

# trainer.fit(model)
# trainer.validate(model)
# trainer.test(model)





import lightning.pytorch as pl
import torch
from torch import nn
from torchvision.datasets import MNIST
import os
from sklearn.metrics import f1_score
import torch.nn.functional as F
from torchvision import transforms
INPUT_FEATURES = 28*28
OUTPUT_CLASS = 10
class Model(pl.LightningModule):
  def __init__(self, lr=1e-6):
    super(Model, self).__init__()
  
    # Declarate for training and validation value
    self.lr = lr
    self.train_step_outputs = []
    self.validation_step_outputs = []
    self.train_step_epochs = []
    self.validation_step_epochs = []
    self.train_step_mean = 0
    self.validation_step_mean = 0

    self.layer_1 = nn.Linear(INPUT_FEATURES, 128)
    self.layer_2 = nn.Linear(128, 256)
    self.layer_3 = nn.Linear(256, OUTPUT_CLASS)

  def forward(self, x):
    """" This is input the model """
    # e.g torch.Size(32, 1, 28, 28)
    batch, channels, width, height = x.size()
    # (b, 1, 28, 28) -> (b, 1*28*28)
    # result torch.size(784, 1)
    x = x.view(batch, -1)
    # layer 1 (b, 1*28*28) -> (b, 128)
    x = F.relu(self.layer_1(x))
    x = F.relu(self.layer_2(x))
    x = self.layer_3(x)

    return x

  def cross_entropy_loss(self, logits, labels):
    return F.cross_entropy(logits, labels)

  def training_step(self, train_batch, batch_idx):
    x, y = train_batch
    logits = self(x)
    loss = self.cross_entropy_loss(logits, y)
    self.train_step_outputs.append(loss)
    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
    return loss

  def validation_step(self, val_batch, batch_idx):
    x, y = val_batch
    loss, acc = self._shared_eval_step(val_batch, batch_idx)
    # sample_imgs = x[:6]
    # grid = torchvision.utils.make_grid(sample_imgs)
    # self.logger.experiment.add_image('example_images', grid, 0)
    metrics = {"val_acc": acc, "val_loss": loss}
    self.validation_step_outputs.append(loss)
    self.log_dict(metrics, sync_dist=True, enable_graph=True)
    return metrics

  def _shared_eval_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = F.cross_entropy(y_hat, y)
    acc = accuracy(y_hat, y, task="multiclass", num_classes=10)
    return loss, acc

  def test_step(self, batch, batch_idx):
    loss, acc = self._shared_eval_step(batch, batch_idx)
    metrics = {"test_acc": acc, "test_loss": loss}
    self.log_dict(metrics)
    return metrics

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    return optimizer

  def setup(self, stage):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081))])

    # prepare transforms standard to MNIST
    self.mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
    self.mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)

  def train_dataloader(self):
    return DataLoader(self.mnist_train, batch_size=16, num_workers=4)

  def test_dataloader(self):
    return DataLoader(self.mnist_test, batch_size=16, num_workers=4)

  def val_dataloader(self):
    return DataLoader(self.mnist_test, batch_size=16, num_workers=4)

model = Model.load_from_checkpoint("/home/henz/Desktop/latihan/Pytorch Lightning/logs/model/version_0/checkpoints/epoch=49-step=26800.ckpt")
model.eval()
print(model)
mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
mnist_test = [mnist_test[i] for i in range(3000,5000)] 
x = torch.cat([mnist_test[x][0] for x in range(20)], 0) # 20 images
print(x.size()) # torch.Size([20, 28, 28])
x = x.unsqueeze(1) # torch.Size([20, 1, 28, 28])
y = [mnist_test[x][1] for x in range(20)] # [6, 9, 8, 1, 2, 9, 9, 5, 9, 7, 3, 7, 8, 0, 1, 3, 0, 4, 6, 1]
print(y)
print("================================ Prediction =================================================================")
yhat = model(x) # logits
yhat = torch.argmax(yhat, 1) # [6, 9, 8, 8, 2, 8, 9, 5, 9, 7, 3, 7, 8, 0, 1, 3, 0, 4, 6, 1]
results = [x.item() for x in yhat]
print(results)
print("================================ Evaluation =================================================================")
f1_score = f1_score(yhat, y, average="macro")
print(f"F1 Score: {f1_score}")





# import pandas as pd
# import matplotlib.pyplot as plt
# def evaluate_train_loss_val_loss_visualize(path):
#   metrics = pd.read_csv(path)
#   agg_metrics = []
#   agg_col = "epoch"
#   for i, y in metrics.groupby(agg_col):
#     agg = dict(y.mean())
#     agg[agg_col] = i
#     agg_metrics.append(agg)

#   df_metrics = pd.DataFrame(agg_metrics)
#   df_metrics[["train_loss_step", "val_loss"]].plot(grid=True, legend=True, xlabel="Epoch", ylabel="Loss")
#   plt.show()

# PATH = "/home/henz/Desktop/latihan/Pytorch Lightning/logs/model/version_0/metrics.csv"
# evaluate_train_loss_val_loss_visualize(PATH)