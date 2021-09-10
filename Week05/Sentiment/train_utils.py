import torch
import tqdm
from utils import detach
from IPython.display import clear_output


def train_step(model, train_dl, criterion, optimizer, device):
    model.train()
    
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    
    for i, (train_inputs, train_labels) in tqdm.notebook.tqdm(enumerate(train_dl), 
                                                         desc='Training', 
                                                         total=len(train_dl)):

        train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
        if len(train_labels) < train_dl.batch_size: continue
        
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        output = model(train_inputs.t())
        
        loss = criterion(output, train_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)        
        optimizer.step()

        # calculate training acc and loss
        _, predicted = torch.max(output, 1)
        total_acc += (predicted == train_labels).sum()
        total_loss += loss.item()
        total += len(train_labels)
        
    return total_loss / total, total_acc / total


def validate_step(model, valid_dl, criterion, device):
    model.eval()
        
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    
    with torch.no_grad():
        for i, (test_inputs, test_labels) in tqdm.notebook.tqdm(enumerate(valid_dl), 
                                                           desc='Validation', 
                                                           total=len(valid_dl)):

            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

            if len(test_labels) < valid_dl.batch_size: continue

            output = model(test_inputs.t())
            loss = criterion(output, test_labels)

            # calculate testing acc and loss
            _, predicted = torch.max(output, 1)
            total_acc += (predicted == test_labels).sum()
            total_loss += loss.item()
            total += len(test_labels)
        
    return total_loss / total, total_acc / total


def train(model, train_dl, valid_dl, criterion, optimizer, device, scheduler, num_epochs):
    max_len, min_count = train_dl.dataset.max_len, train_dl.dataset.min_count
    
    train_hist, valid_hist = [], []
    best_acc, best_wts = 0.0, None
    
    report = ""

    for epoch in range(num_epochs):
        if epoch > 0: 
            clear_output(wait=True)
            print(report)

        ## perform one epoch of training and validation
        trn_loss, trn_acc = train_step(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc = validate_step(model, valid_dl, criterion, device)
        scheduler.step()

        train_hist += [(trn_loss, trn_acc)]
        valid_hist += [(val_loss, val_acc)]

        # save weights
        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = model.state_dict().copy()
            torch.save(best_wts, 'models/lstm-{}-{}-{}-{}-{}-{}-{:.5f}.pth'.format(
                model.num_layers, max_len, min_count, model.embedding_dim, model.hidden_dim, epoch, best_acc))

        report += f'[Epoch: {epoch + 1:2d}/{num_epochs:2d}] | Training Loss: {trn_loss:.4f} | Testing Loss: {val_loss:.4f} | Training Acc: {trn_acc*100:.2f} | Testing Acc: {val_acc*100:.2f}\n'

    model.load_state_dict(best_wts)
    return train_hist, valid_hist