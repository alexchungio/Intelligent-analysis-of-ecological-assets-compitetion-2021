
from torch.optim.lr_scheduler import _LRScheduler


class FindLR(_LRScheduler):
    """
    inspired by fast.ai @https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """
    def __init__(self, optimizer, max_steps, max_lr=10):
        self.max_steps = max_steps
        self.max_lr = max_lr
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * ((self.max_lr / base_lr) ** (self.last_epoch / (self.max_steps - 1)))
                for base_lr in self.base_lrs]


class NoamLR(_LRScheduler):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.optim as optim
    from torchvision.models import resnet18

    epochs = 60
    base_lr = 0.01
    train_loader_size = 50
    warmup_epochs = 2
    warmup_factor = 1. / 3
    gamma = 0.1
    milestones = [40, 50]

    model = resnet18(num_classes=10)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, nesterov=True)

    scheduler = NoamLR(optimizer=optimizer, warmup_steps=warmup_epochs)
    lrs = []
    step = 1
    for epoch in range(epochs):
        for index, data in enumerate(np.arange(0, train_loader_size)):
            optimizer.zero_grad()
            optimizer.step()
            lrs.append(optimizer.param_groups[0]['lr'])
            print('step: {}'.format(step), 'lr: {}'.format(optimizer.param_groups[0]['lr']))
            scheduler.step(epoch + index / train_loader_size)  # update lr every step
            step += 1
        # scheduler.step()  # update lr every epoch
    plt.plot(lrs, c='g', label='warmup step_lr', linewidth=1)
    plt.legend(loc='best')
    plt.show()



