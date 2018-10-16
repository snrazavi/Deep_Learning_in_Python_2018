import torch
from tqdm import tqdm_notebook
from utils import to_var, detach


def train_step(model, train_dl, criterion, optimizer, scheduler):
    model.train()
    scheduler.step()
    
    # init hidden states
    model.hidden = model.init_hidden()

    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    
    for i, (train_inputs, train_labels) in tqdm_notebook(enumerate(train_dl), 
                                                         desc='Training', 
                                                         total=len(train_dl)):

        train_inputs, train_labels = to_var(train_inputs), to_var(train_labels)
        if len(train_labels) < train_dl.batch_size: continue
        
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.hidden = detach(model.hidden)
        model.zero_grad()
        output = model(train_inputs.t())
        
        loss = criterion(output, train_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.3)        
        optimizer.step()

        # calculate training acc and loss
        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == train_labels.data).sum()
        total_loss += loss.data[0]
        total += len(train_labels)
        
    return total_loss / total, total_acc / total


def validate_step(model, valid_dl, criterion):
    model.eval()
    
    # init hidden states
    model.hidden = model.init_hidden()
    
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    
    for i, (test_inputs, test_labels) in tqdm_notebook(enumerate(valid_dl), 
                                                       desc='Validation', 
                                                       total=len(valid_dl)):

        test_inputs, test_labels = to_var(test_inputs, True), to_var(test_labels, True)
        if len(test_labels) < valid_dl.batch_size: continue

        output = model(test_inputs.t())
        loss = criterion(output, test_labels)

        # calculate testing acc and loss
        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == test_labels.data).sum()
        total_loss += loss.data[0]
        total += len(test_labels)
        
        model.hidden = detach(model.hidden)
        
    return total_loss / total, total_acc / total


def train(model, train_dl, valid_dl, criterion, optimizer, scheduler, num_epochs):
    max_len, min_count = train_dl.dataset.max_len, train_dl.dataset.min_count
    
    train_hist, valid_hist = [], []
    best_acc, best_wts = 0.0, None

    for epoch in range(num_epochs):

        ## perform one epoch of training and validation
        trn_loss, trn_acc = train_step(model, train_dl, criterion, optimizer, scheduler)
        val_loss, val_acc = validate_step(model, valid_dl, criterion)

        train_hist += [(trn_loss, trn_acc)]
        valid_hist += [(val_loss, val_acc)]

        # save weights
        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = model.state_dict().copy()
            torch.save(best_wts, 'models/lstm-{}-{}-{}-{}-{}-{}-{:.5f}.pth'.format(
                epoch, max_len, min_count, model.embed_size, model.hidden_size, model.num_layers, best_acc))

        print('[Epoch: %3d/%3d] Training Loss: %.3f, Testing Loss: %.3f, Training Acc: %.3f, Testing Acc: %.3f'
              % (epoch + 1, num_epochs, trn_loss, val_loss, trn_acc, val_acc))

    model.load_state_dict(best_wts)
    return train_hist, valid_hist