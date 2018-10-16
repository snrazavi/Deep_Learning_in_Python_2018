import itertools
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from utils import to_var


def imshow(inp, title=None):
    """Imshow for Tensor.
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.axis('off')
    if title is not None:
        plt.title(title)


def visualize_model(model, dataloader, num_images=6):
    """ Visulaize the prediction of the model on a bunch of random data.
    """
    model.train(False)
    
    images_so_far = 0
    fig = plt.figure(figsize=(10., 8.))

    for i, (inputs, labels, _) in enumerate(dataloader):
        inputs, labels = to_var(inputs, volatile=True), to_var(labels, volatile=True)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(dataloader.dataset.classes[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return

def plot_errors(model, dataloader):
    model.train(False)
    
    plt.figure(figsize=(12, 24))
    count = 0
    
    for (inputs, labels, _) in tqdm(dataloader):
        inputs, labels = to_var(inputs, volatile=True), to_var(labels, volatile=True)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        incorrect_idxs = np.flatnonzero(preds.cpu().numpy() != labels.data.cpu().numpy())
        
        for idx in incorrect_idxs:
            count += 1
            if count > 30: break
            ax = plt.subplot(10, 3, count)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(dataloader.dataset.classes[preds[idx]]))
            imshow(inputs.cpu().data[idx])
    plt.show()

    print("{} images out of {} were misclassified.".format(count, len(dataloader.dataset)))


def plot_confusion_matrix(cm, classes, normalize=False, figsize=(12, 12), title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        annot = "%.2f" % cm[i, j] if cm[i, j] > 0 else "" 
        plt.text(j, i, annot, horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plots_raw(ims, figsize=(12,6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, ceildiv(len(ims), rows), i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])

def load_img_id(ds, idx, path): return np.array(PIL.Image.open(os.path.join(path, ds.fnames[idx])))


class ImageModelResults():
    """ Visualize the results of an image model
    
    Arguments:
        ds (dataset): a dataset which contains the images
        log_preds (numpy.ndarray): predictions for the dataset in log scale
        
    Returns:
        ImageModelResults
    """
    def __init__(self, ds, log_preds):
        """Initialize an ImageModelResults class instance"""
        self.ds = ds
        # returns the indices of the maximum value of predictions along axis 1, representing the predicted class
        # log_preds.shape = (number_of_samples, number_of_classes);
        # preds.shape = (number_of_samples,)
        self.preds = np.argmax(log_preds, axis=1)
        # computes the probabilities
        self.probs = np.exp(log_preds)
        # extracts the number of classes
        self.num_classes = log_preds.shape[1]

    def plot_val_with_title(self, idxs, y):
        """ Displays the images and their probabilities of belonging to a certain class
            
            Arguments:
                idxs (numpy.ndarray): indexes of the image samples from the dataset
                y (int): the selected class
                
            Returns:
                Plots the images in n rows [rows = n]
        """
        # if there are any samples to be displayed
        if len(idxs) > 0:
            imgs = np.stack([self.ds[x][0] for x in idxs])
            title_probs = [self.probs[x,y] for x in idxs]

            return plots(self.ds.denorm(imgs), rows=1, titles=title_probs)
        # if idxs is empty return false
        else:
            return False;

    def most_by_mask(self, mask, y, mult):
        """ Extracts the first 4 most correct/incorrect indexes from the ordered list of probabilities
        
            Arguments:
                mask (numpy.ndarray): the mask of probabilities specific to the selected class; a boolean array with shape (num_of_samples,) which contains True where class==selected_class, and False everywhere else
                y (int): the selected class
                mult (int): sets the ordering; -1 descending, 1 ascending
                
            Returns:
                idxs (ndarray): An array of indexes of length 4
        """
        idxs = np.where(mask)[0]
        return idxs[np.argsort(mult * self.probs[idxs,y])[:4]]

    def most_uncertain_by_mask(self, mask, y):
        """ Extracts the first 4 most uncertain indexes from the ordered list of probabilities
            
            Arguments:
                mask (numpy.ndarray): the mask of probabilities specific to the selected class; a boolean array with shape (num_of_samples,) which contains True where class==selected_class, and False everywhere else
                y (int): the selected class
            
            Returns:
                idxs (ndarray): An array of indexes of length 4
        """
        idxs = np.where(mask)[0]
        # the most uncertain samples will have abs(probs-1/num_classes) close to 0;
        return idxs[np.argsort(np.abs(self.probs[idxs,y]-(1/self.num_classes)))[:4]]
    
    def most_by_correct(self, y, is_correct):
        """ Extracts the predicted classes which correspond to the selected class (y) and to the specific case (prediction is correct - is_true=True, prediction is wrong - is_true=False)
            
            Arguments:
                y (int): the selected class
                is_correct (boolean): a boolean flag (True, False) which specify the what to look for. Ex: True - most correct samples, False - most incorrect samples
            
            Returns:
                idxs (numpy.ndarray): An array of indexes (numpy.ndarray)
        """
        # mult=-1 when the is_correct flag is true -> when we want to display the most correct classes we will make a descending sorting (argsort) because we want that the biggest probabilities to be displayed first.
        # When is_correct is false, we want to display the most incorrect classes, so we want an ascending sorting since our interest is in the smallest probabilities.
        mult = -1 if is_correct==True else 1
        return self.most_by_mask(((self.preds == self.ds.y)==is_correct)
                                 & (self.ds.y == y), y, mult)

    def plot_by_correct(self, y, is_correct):
        """ Plots the images which correspond to the selected class (y) and to the specific case (prediction is correct - is_true=True, prediction is wrong - is_true=False)
            
            Arguments:
                y (int): the selected class
                is_correct (boolean): a boolean flag (True, False) which specify the what to look for. Ex: True - most correct samples, False - most incorrect samples
        """    
        return self.plot_val_with_title(self.most_by_correct(y, is_correct), y)

    def most_by_uncertain(self, y):
        """ Extracts the predicted classes which correspond to the selected class (y) and have probabilities nearest to 1/number_of_classes (eg. 0.5 for 2 classes, 0.33 for 3 classes) for the selected class.
            
            Arguments:
                y (int): the selected class
            
            Returns:
                idxs (numpy.ndarray): An array of indexes (numpy.ndarray)
        """
        return self.most_uncertain_by_mask((self.ds.y == y), y)

    def plot_most_correct(self, y):
        """ Plots the images which correspond to the selected class (y) and are most correct.
            
            Arguments:
                y (int): the selected class
        """
        return self.plot_by_correct(y, True)
    def plot_most_incorrect(self, y): 
        """ Plots the images which correspond to the selected class (y) and are most incorrect.
            
            Arguments:
                y (int): the selected class
        """
        return self.plot_by_correct(y, False)
    def plot_most_uncertain(self, y):
        """ Plots the images which correspond to the selected class (y) and are most uncertain i.e have probabilities nearest to 1/number_of_classes.
            
            Arguments:
                y (int): the selected class
        """
        return self.plot_val_with_title(self.most_by_uncertain(y), y)

