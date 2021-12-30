import copy
from collections import OrderedDict

import torch


class CenterServer:
    def __init__(self, model, dataloader, device="cpu"):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def aggregation(self):
        raise NotImplementedError

    def send_model(self):
        return copy.deepcopy(self.model)

    def validation(self):
        raise NotImplementedError


class FedAvgCenterServer(CenterServer):
    def __init__(self, model, dataloader, device="cpu"):
        super().__init__(model, dataloader, device)

    def aggregation(self, clients, aggregation_weights):
        update_state = OrderedDict()

        for k, client in enumerate(clients):
            local_state = client.model.state_dict()
            for key in self.model.state_dict().keys():
                if k == 0:
                    update_state[
                        key] = local_state[key] * aggregation_weights[k]
                else:
                    update_state[
                        key] += local_state[key] * aggregation_weights[k]

        self.model.load_state_dict(update_state)

    def validation(self, loss_fn):
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for img, target in self.dataloader:
                img = img.to(self.device)
                target = target.to(self.device)
                logits = self.model(img)
                test_loss += loss_fn(logits, target).item()
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        self.model.to("cpu")
        test_loss = test_loss / len(self.dataloader)
        accuracy = 100. * correct / len(self.dataloader.dataset)

        return test_loss, accuracy
