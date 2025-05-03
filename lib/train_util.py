from tqdm import tqdm
import math

import torch
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AverageMetric:

    def __init__(self,):
        self._running_scores = torch.zeros(1)
        self._count = 0.

    def update_state(self, value, counts):

        if math.isnan(value) or math.isinf(value):
            return
        
        self._count += counts
        self._running_scores += value

    def reset_state(self,):
        self._running_scores = torch.zeros_like(self._running_scores)
        self._count = 0

    def result(self,):
        return self._running_scores/self._count


class Accuracy(AverageMetric):

    def __init__(self, topk=(1,)):
        super().__init__()
        self._maxk = max(topk)
        self._running_scores = torch.zeros(len(topk))
        self._topk = topk

    def update_state(self, outputs, targets):
        with torch.no_grad():
            self._count += targets.size(0)
            _, pred = outputs.topk(self._maxk, 1, True, True)

            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            for i, k in enumerate(self._topk):
                self._running_scores[i] += \
                    correct[:k].reshape(-1).float().sum(0).to('cpu')

def train(
        net,
        trainloader,
        testloader,
        criterion,
        optimizer,
        epochs,
        batch_size,
        n_classes,
        log_every_n=250,
        checkpoint=None,
        summary_writer=None,
        save_only_head=False,
        ):

    accuracy = Accuracy((1, min(5, n_classes)))

    train_loss = AverageMetric()
    test_loss = AverageMetric()
    best_acc = (0., 0.)
    start_epoch = 0

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)

    global_steps = 0

    for epoch in range(start_epoch, epochs):

        print('\nEpoch: %d' % epoch)
        print('\nLearning Rate: %.4f' %
              scheduler.get_last_lr()[0])

        net.train()

        train_loss.reset_state()

        for inputs, targets in tqdm(trainloader, desc=f"Training epoch {epoch}"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            net.zero_grad()
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.update_state(loss.item(), 1)
            global_steps += 1

            if global_steps % log_every_n == 0:
                summary_writer.add_scalar("train/loss", train_loss.result(), global_steps)
                summary_writer.add_scalar("train/epoch", epoch, global_steps)
                summary_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_steps)
                train_loss.reset_state()

        scheduler.step()
        accuracy.reset_state()

        net.eval()
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in tqdm(enumerate(testloader),
                                                     desc=f"Evaluating epoch {epoch}"):
                
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss.update_state(loss.item(), 1)
                accuracy.update_state(outputs, targets)

        val_top1, val_top5 = accuracy.result()

        print(f"Top 1 acc: {val_top1}")
        print(f"Top 5 acc: {val_top5}")

        summary_writer.add_scalar("test/loss", test_loss.result(), epoch)
        summary_writer.add_scalar("test/acc_top1", val_top1, epoch)
        summary_writer.add_scalar("test/acc_top5", val_top5, epoch)
        
        accuracy.reset_state()
        test_loss.reset_state()

        if (checkpoint is not None) and (val_top1 > best_acc[0]):
            best_acc = (val_top1, val_top5)
            print("Saving...")
            if save_only_head:
                torch.save(net.head.state_dict(), checkpoint)
            else:
                torch.save(net.state_dict(), checkpoint)
            
    return best_acc
