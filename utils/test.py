from Focal_Loss import focal_loss
import torch


pred = torch.randn((3,5))
print("pred:",pred)

label = torch.tensor([2,3,4])
print("label:",label)

loss_fn = focal_loss(alpha=0.25, gamma=2, num_classes=5)
loss = loss_fn(pred, label)
print(loss)