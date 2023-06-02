import os.path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from models import Vae
from trainer import train_epoch, test_epoch
from sklearn import svm
import pickle

TRAIN_LR_STOP = 1e-6
HIDDEN_SIZE = 600
Z_SIZE = 50
BATCH_SIZE = 64
PRINT_EVERY = 5
MODELS_DIR = 'models'

torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(dataset):
    training_data = dataset(root="data", train=True, download=True, transform=ToTensor())
    test_data = dataset(root="data", train=False, download=True, transform=ToTensor())
    return training_data, test_data


def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')  # best initialization even though activation is softplus


def construct_svm_data(train_dataloader, test_dataloader, labeled_samples, model):
    # Get balanced classes
    X_train, y_train = next(iter(train_dataloader))
    X_train = X_train.flatten(1)
    num_classes = len(y_train.unique())
    indices_of_balanced_classes = torch.cat([(y_train == i).nonzero()[:(labeled_samples // num_classes)].flatten() for i in range(num_classes)])
    X_train = X_train[indices_of_balanced_classes].to(device)
    y_train = y_train[indices_of_balanced_classes]

    # Get latent representation
    mu, sigma, _ = model(X_train)
    X_train = torch.cat([mu, sigma], dim=1)

    X_test, y_test = next(iter(test_dataloader))
    X_test = X_test.to(device)
    mu, sigma, _ = model(X_test.flatten(1))
    X_test = torch.cat([mu, sigma], dim=1)

    X_train = X_train.cpu()
    X_test = X_test.cpu()

    return (X_train.detach().numpy(), y_train.detach().numpy()), \
           (X_test.detach().numpy(), y_test.detach().numpy())


def train(train_dataset, test_dataset, labeled_samples, use_saved_weights):
    # Train VAE
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = Vae(
        train_dataset[0][0].flatten().size(0),
        HIDDEN_SIZE,
        Z_SIZE,
        train_dataset.__class__.__name__ + '_NN'
    ).to(device)
    model.apply(init_weights)
    nn_save_path = os.path.join(MODELS_DIR, model.name + '.pkl')

    construction_loss_fn = nn.BCELoss(reduction='sum')

    if use_saved_weights and os.path.exists(nn_save_path):
        # Load the NN model
        print('Loading NN old weights from "' + nn_save_path + '"')
        model.load_state_dict(torch.load(nn_save_path, map_location=torch.device(device)))
        test_loss = test_epoch(device, test_dataloader, model, construction_loss_fn)
        print("Loaded NN model - Test Loss: {:>0.3f}".format(test_loss))

    else:
        print('Training NN model...')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=10)
        epoch = 0
        while optimizer.param_groups[0]['lr'] > TRAIN_LR_STOP:
            train_epoch(device, train_dataloader, model, construction_loss_fn, optimizer)
            train_loss = test_epoch(device, train_dataloader, model, construction_loss_fn)
            test_loss = test_epoch(device, test_dataloader, model, construction_loss_fn)
            if epoch % PRINT_EVERY == 0:
                print("Epoch {}, LR: {:8.6f}, Train Loss: {:>0.3f}, Test Loss: {:>0.3f}".format(
                    epoch + 1, optimizer.param_groups[0]['lr'], train_loss, test_loss))
            scheduler.step(test_loss)
            epoch += 1

        train_loss = test_epoch(device, train_dataloader, model, construction_loss_fn)
        test_loss = test_epoch(device, test_dataloader, model, construction_loss_fn)
        print("Epoch {}, LR: {:8.6f}, Train Loss: {:>0.3f}, Test Loss: {:>0.3f}".format(
            epoch + 1, optimizer.param_groups[0]['lr'], train_loss, test_loss))

        # Save the NN model
        print('Saving NN model to "' + nn_save_path + '"')
        torch.save(model.state_dict(), nn_save_path)  # TODO

    # Train SVM
    svm_save_path = os.path.join(MODELS_DIR, train_dataset.__class__.__name__ + '_SVM' + '_' + str(labeled_samples) + '.pkl')
    (X_train, y_train), (X_test, y_test) = construct_svm_data(
        DataLoader(train_dataset, batch_size=len(train_dataset)),
        DataLoader(test_dataset, batch_size=len(test_dataset)),
        labeled_samples,
        model)
    svm_classifier = svm.SVC(kernel='poly', degree=5)

    if use_saved_weights and os.path.exists(svm_save_path):
        # Load the SVM model
        print('Loading SVM old weights from "' + svm_save_path + '"')
        svm_classifier = pickle.load(open(svm_save_path, 'rb'))

    else:
        print('Training SVM model...')
        svm_classifier.fit(X_train, y_train)

        # Save the SVM model
        print('Saving SVM model to "' + svm_save_path + '"')
        pickle.dump(svm_classifier, open(svm_save_path, 'wb'))

    pred = svm_classifier.predict(X_test)
    print('Accuracy Error for {} labeled samples: {:>0.2f}%'.format(labeled_samples, 100 * (1 - (pred == y_test).mean())))
    print()


def main():
    training_data, test_data = load_data(datasets.MNIST)
    for labeled_samples in [100, 600, 1000, 3000]:
        train(training_data, test_data, labeled_samples, use_saved_weights=True)

    print('#' * 50)
    print()

    training_data, test_data = load_data(datasets.FashionMNIST)
    for labeled_samples in [100, 600, 1000, 3000]:
        train(training_data, test_data, labeled_samples, use_saved_weights=True)


if __name__ == '__main__':
    main()
