import math
import warnings

import numpy as np
import torch.nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset_utils.dataset import Dataset, SiameseNetworkDataset
# from utils.transforms_util import ShoulderAbductionTransform
# from utils_models.physiq import UnitNormClipper
from utils.util import save_paths

# def add_weights(opt, data, label, add_weight_flag=True):
#     weights = [
#         [0, 1, 1, 1, 1, 1],
#         [0, 1, 1, 1, 1.5, 1],
#         [0, 1, 1, 1, 1.5, 1],
#         [0, 1, 1, 1, 1.5, 1],
#         [0, 1, 1, 1.25, 1, 1.25]
#     ]
#     if add_weight_flag:
#         for i, (d, l) in enumerate(zip(data, label)):
#             w = torch.tensor(weights[l]).to(opt.device)
#             data[i] = d * w
#     return data


def generate_dataloader(
    *args,
    custom_dataset,
    shuffle=True,
    batch_size=256,
    collate_fn=None,
    drop_last=False,
    transform=None
):
    if (len(args[0]) < batch_size and drop_last
        ) or (len(args) >= 3 and len(args[2]) < batch_size and drop_last):
        warnings.warn("batch size is too big and drop last is on")
    assert len(args) % 2 == 0, "args are not even number"
    r = tuple()
    start_offset = 0
    for i in range(len(args) // 2):
        ds = custom_dataset(
            args[start_offset], args[start_offset + 1], transform=transform
        )
        dl = DataLoader(
            ds,
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=drop_last
        )
        r = r + (dl, )
        start_offset += 2
        drop_last = False
    return r


class CustomCollate:

    def __init__(self, opt):
        self.device = opt.device

    def custom_collate(self, batch):
        # batch = list of tuples where each tuple is of the form ([i1, i2, i3], [j1, j2, j3], label)
        q1_list = torch.rand(
            (len(batch), len(batch[0][0]), len(batch[0][0][1]))
        ).float()
        q2_list = torch.rand(
            (len(batch), len(batch[0][0]), len(batch[0][0][1]))
        ).float()
        labels = torch.tensor([]).float()
        for i, (s1, s2, lab) in enumerate(batch, 0):
            q1_list[i, :, :] = s1
            q2_list[i, :, :] = s2
            labels = torch.cat([labels, lab.float()], 0)
        return (q1_list.to(self.device), q2_list.to(self.device
                                                    )), labels.to(self.device)

    def __call__(self, batch):
        return self.custom_collate(batch)


def evaluate(
    opt, x_batch, label, model, optimizer, criterion, cre, test=False
):
    device = opt.device
    if opt.siamese:
        inputs = []
        for each in x_batch:
            inp = Variable(each).float().to(device)  # [8, 400, 6]
            inputs.append(inp)
    else:
        inputs = Variable(x_batch).float().to(device)
        # inputs = add_weights(opt, inputs, label, add_weight_flag=True)
    if cre:
        label = Variable(label).long()
    else:
        label = Variable(label).float()
    label = Variable(label).to(device)
    if opt.siamese:
        output = model(*inputs)
    else:
        if opt.explainable_model:
            output, rec, a, b, c = model(inputs)
            rec_loss = torch.nn.MSELoss()(
                inputs.view(inputs.shape[0], -1), rec
            )
        else:
            output = model(inputs)
    if not cre:
        output = output.squeeze()
    # if not opt.siamese:
    if opt.explainable_model:
        loss = criterion(output, label) + rec_loss
    else:
        loss = criterion(output, label)
    # else:
    #     loss = criterion(output, label)
    # if opt.debug: print('rec loss', rec_loss.item())
    # if test:
    #     ax0 = plt.subplot(511)
    #     ax0.plot(inputs.view(*inputs.shape)[0, :, 0].cpu().detach().numpy())
    #     ax0.set_title('input')
    #     ax1 = plt.subplot(512)
    #     ax1.plot(rec.view(*inputs.shape)[0, :, 0].cpu().detach().numpy())
    #     ax1.set_title('reconstructing')
    #     ax2 = plt.subplot(513)
    #     ax2.plot(a.view(*inputs.shape)[0, :, 0].cpu().detach().numpy())
    #     ax2.set_title('trend')
    #
    #     ax3 = plt.subplot(514)
    #     ax3.plot(b.view(*inputs.shape)[0, :, 0].cpu().detach().numpy())
    #     ax3.set_title('seasonality')
    #     ax4 = plt.subplot(515)
    #     ax4.plot(c.view(*inputs.shape)[0, :, 0].cpu().detach().numpy())
    #     ax4.set_title('residual')
    #     plt.show()
    return output, loss


def reference(
    opt, x_batch, label, model, optimizer, criterion, cre, test=False
):
    output, loss = evaluate(
        opt, x_batch, label, model, optimizer, criterion, cre, test=test
    )
    # valid_loss.append(loss.data.cpu())
    if cre:
        # classification
        # print(output.shape)
        preds = F.softmax(output, dim=1).argmax(dim=1)

        # verification of the output values:
        outt = output.cpu().detach().numpy()
        r, c = np.where(((outt < 1e-5) & (outt >= -1e-5)))
        s = np.searchsorted(r, np.arange(1, outt.shape[0]))
        for i in np.split(c, s):
            if len(i) >= opt.output_size:
                warnings.warn('there is softmax is all zero')

    else:
        # regression
        preds = output
        preds = preds.squeeze()
    return preds, loss
    # target.extend(list(label.data.cpu()))
    # predict.extend(list(preds.data.cpu()))


def train(
    opt,
    model,
    data_input,
    criterion,
    optimizer,
    name='',
    specific_directory=None
):
    device = opt.device
    # init model:
    model = model(opt).to(device)
    # if opt.XAI_weights is not None:
    #     model.weights = Parameter(torch.tensor(opt.XAI_weights,
    #                                           requires_grad=True, device=opt.device))
    txtName, pthName = save_paths(
        specific_directory, model.model_name, name=name, classification=True
    )
    f = open(txtName, 'w')

    train_x, train_y, valid_x, valid_y = data_input
    if opt.siamese:
        train_dataloader, valid_dataloader, = generate_dataloader(
            train_x,
            train_y,
            valid_x,
            valid_y,
            custom_dataset=SiameseNetworkDataset,
            shuffle=True,
            batch_size=opt.batch_size,
            collate_fn=CustomCollate(opt),
            drop_last=True,
            transform=opt.transform
        )
    else:
        train_dataloader, valid_dataloader = generate_dataloader(
            train_x,
            train_y,
            valid_x,
            valid_y,
            custom_dataset=Dataset,
            shuffle=True,
            batch_size=opt.batch_size,
            transform=opt.transform
        )
    if opt.debug:
        print(
            "NUMBER OF PARAMETERS FOR MODEL:",
            sum(p.numel() for p in model.parameters())
        )

    # init the criterion, optimizer:
    cre = False
    if criterion in [torch.nn.CrossEntropyLoss]:
        cre = True
    criterion = criterion().to(device)
    optimizer = optimizer(model.parameters(), lr=opt.lr)
    best_loss = math.inf
    patience, trial = int(opt.epochs / .2), 0
    train_loss_sum = []
    train_loss_record = []

    # clipper = UnitNormClipper()
    for epoch in range(opt.epochs):
        running_loss = []
        model.train()
        for i, (x_batch, label) in enumerate(train_dataloader, 0):
            optimizer.zero_grad()
            output, loss = evaluate(
                opt, x_batch, label, model, optimizer, criterion, cre
            )
            # model.weights.retain_grad()
            loss.backward()

            optimizer.step()
            train_loss_sum.append(loss.data.cpu().item())
            running_loss.append(loss.data.cpu().item())
        # model.apply(clipper)
        if opt.debug:
            print(
                'Train Loss at epoch {}: current: {}, avg: {}'.format(
                    epoch, np.mean(running_loss), np.mean(train_loss_sum)
                )
            )
        train_loss_record.append(np.mean(train_loss_sum))

        # VALIDATION PHASE:
        model.eval()
        correct, total = 0, len(valid_x)
        valid_loss = []
        target = []
        predict = []
        for idx, (x_batch, label) in enumerate(valid_dataloader, 0):
            preds, loss = reference(
                opt, x_batch, label, model, optimizer, criterion, cre
            )
            valid_loss.append(loss.data.cpu())
            target.extend(list(label.data.cpu()))
            predict.extend(list(preds.data.cpu()))
            if cre:
                correct += (preds.data.cpu() == label.data.cpu()).sum().item()
            else:
                correct += np.count_nonzero(
                    np.isclose(preds.data.cpu(), label.data.cpu(), atol=.05)
                )
        temp_acc = correct / total
        if not cre:
            temp_acc = r2_score(target, predict)
        temp_loss = np.mean(valid_loss)

        print(
            "{}, {}, {:10.8f}, {:10.8f}, {}".format(
                epoch, np.mean(train_loss_sum), temp_loss, temp_acc, correct
            ),
            file=f
        )
        if opt.debug:
            print(
                'epoch {}: CEL {:10.8f}, acc {:10.8f}, cor {}'.format(
                    epoch, temp_loss, temp_acc, correct
                )
            )
        if temp_loss < best_loss:
            best_loss = temp_loss
            trial = 0
            torch.save(model.state_dict(), pthName)
            # torch.save(mlp.state_dict(), classificationName)
        else:
            trial += 1
            if trial >= patience:
                if opt.debug:
                    print(f'Early stopping on epoch {epoch}')
                break

    f.close()


def test(
    opt,
    model,
    data_input,
    criterion,
    optimizer,
    name='',
    specific_directory=None,
    shuffle=True
):
    device = opt.device
    # init model:
    model = model(opt).to(device)
    txtName, pthName = save_paths(
        specific_directory, model.model_name, name=name, classification=True
    )
    if opt.debug: print(txtName, pthName)
    test_x, test_y = data_input
    if opt.siamese:
        test_dataloader, = generate_dataloader(
            test_x,
            test_y,
            custom_dataset=SiameseNetworkDataset,
            shuffle=shuffle,
            batch_size=opt.batch_size,
            collate_fn=CustomCollate(opt),
            transform=opt.transform
        )
    else:
        test_dataloader, = generate_dataloader(
            test_x,
            test_y,
            custom_dataset=Dataset,
            shuffle=shuffle,
            batch_size=opt.batch_size,
            transform=opt.transform
        )
    if not torch.cuda.is_available():
        model.load_state_dict(
            torch.load(pthName, map_location=torch.device('cpu'))
        )
    else:
        model.load_state_dict(torch.load(pthName))

    # init the criterion, optimizer:
    cre = False
    if criterion in [torch.nn.CrossEntropyLoss]:
        cre = True
    criterion = criterion().to(device)
    model.eval()
    if opt.baseline.lower() in ['lstm_dropout']:

        def enable_dropout(m):
            for each_module in m.modules():
                if each_module.__class__.__name__.startswith('Dropout'):
                    each_module.train()

        enable_dropout(model)

    correct, total = 0, len(test_x)
    test_loss = []
    target = []
    predict = []
    for i, (x_batch, label) in enumerate(test_dataloader, 0):

        preds, loss = reference(
            opt, x_batch, label, model, optimizer, criterion, cre, test=True
        )
        test_loss.append(loss.data.cpu())
        target.extend(list(label.data.cpu()))
        predict.extend(list(preds.data.cpu()))
        if cre:
            correct += (preds.data.cpu() == label.data.cpu()).sum().item()
        else:
            correct += np.count_nonzero(
                np.isclose(preds.data.cpu(), label.data.cpu(), atol=.05)
            )
    temp_acc = correct / total
    if not cre:
        temp_acc = r2_score(target, predict)
    temp_loss = np.mean(test_loss)

    if opt.debug:
        print(
            'CEL {:10.8f}, acc {:10.8f}, cor {}'.format(
                temp_loss, temp_acc,
                (str(correct) + '/' + str(total) if cre else 'N\A')
            )
        )
    return target, predict, temp_loss, temp_acc
