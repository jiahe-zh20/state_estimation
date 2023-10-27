import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def Trainer(model, model_optimizer, train_dl, valid_dl, test_dl,
            device, logger, config, ex_log, training_mode='self_supervised'):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    for epoch in range(1, config.epoch + 1):
        # Train and validate
        train_loss, train_acc = model_train(model, model_optimizer, criterion,
                                            train_dl, config, device, training_mode)
        valid_loss, valid_acc = model_evaluate(model, valid_dl, device, 'valid', training_mode)
        scheduler.step(valid_loss)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:.4f}\n'
                     f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:.4f}')

        # save model after training
        os.makedirs(os.path.join(ex_log, 'save_models'), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ex_log, 'save_models', f'final_model.pth'))

        logger.debug('\n Evaluate on the test set:')
        test_loss, test_acc, preds, poses, masks = model_evaluate(
            model, test_dl, device, 'test', training_mode)
        logger.debug(f'Test loss      :{test_loss:2.4f}\t | Test Accuracy      : {test_acc:2.4f}')

    logger.debug("\n################## Training is Done! #########################")


def model_train(model, model_optimizer, criterion, train_dl, config, device, training_mode):
    total_loss = []
    total_acc = []
    model.train()

    for batch_idx, (data, mask_pos, mask_values, Z) in train_dl:
        # send to device
        data = data.float().to(device)
        mask_pos = mask_pos.float().to(device)
        mask_values = mask_values.float().to(device)
        Z = Z.float().to(device)

        model_optimizer.zero_grad()
        _, mask_values_pred = model(data, mask_pos, Z)
        loss = criterion(mask_values_pred, mask_values)

        mask_values_pred_detach = mask_values_pred.detach().squeeze(-1)
        total_acc.append(torch.abs((mask_values - mask_values_pred_detach) / mask_values).sum(dim=1).mean())
        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()
    total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc


def model_evaluate(model, test_dl, device, mode, training_mode):
    model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.MSELoss()
    if mode == 'test':
        preds = np.array([])
        poses = np.array([])
        masks = np.array([])

    with torch.no_grad():
        for batch_idx, (data, mask_pos, mask_values, Z) in test_dl:
            # send to device
            data = data.float().to(device)
            mask_pos = mask_pos.float().to(device)
            mask_values = mask_values.float().to(device)
            Z = Z.float().to(device)

            _, mask_values_pred = model(data, mask_pos, Z)
            loss = criterion(mask_values_pred, mask_values)

            mask_values_pred_detach = mask_values_pred.detach().squeeze(-1)
            total_acc.append(torch.abs((mask_values - mask_values_pred_detach) / mask_values).sum(dim=1).mean())
            total_loss.append(loss.item())

            if mode == 'test':
                preds = np.append(preds, mask_values_pred.cpu().numpy())
                poses = np.append(poses, mask_pos.data.cpu().numpy())
                masks = np.append(masks, mask_values.data.cpu().numpy())

    total_loss = torch.tensor(total_loss).mean()
    total_acc = torch.tensor(total_acc).mean()
    if mode == 'test':
        return total_loss, total_acc, preds, poses, masks
    else:
        return total_loss, total_acc
