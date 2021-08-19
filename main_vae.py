import numpy as np
import random
import sys
import matplotlib.pylab as plt
from tqdm import tqdm as tqdm_bar
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class ConvVAE(nn.Module):
    def __init__(self, K, input_dims, filter1, filter2):
        super(ConvVAE, self).__init__()
        self.input_dims = input_dims
        self.filter1 = filter1
        self.filter2 = filter2

        self.conv1 = nn.Conv2d(input_dims[0], self.filter1, 8, stride=4)
        self.conv2 = nn.Conv2d(self.filter1, self.filter2, 4, stride=2)
        self.conv3 = nn.Conv2d(self.filter2, 64, 3, stride=1)

        if 'neurons' not in locals():
            neurons, self.shape_pre_flatten = self.getNeuronNum(torch.zeros(input_dims).unsqueeze(0))

        self.q_fc_mu = nn.Linear(neurons, K)
        self.q_fc_sig = nn.Linear(neurons, K)

        self.p_fc_upsample = nn.Linear(K, neurons)

        self.p_unflatten = nn.Unflatten(-1, self.shape_pre_flatten)

        self.p_deconv_1 = nn.ConvTranspose2d(64, self.filter2, 3, stride=1)
        self.p_deconv_2 = nn.ConvTranspose2d(self.filter2, self.filter1, 4, stride=2, output_padding=1) # TODO: try to get this working without output_padding
        self.p_deconv_3 = nn.ConvTranspose2d(self.filter1, input_dims[0], 8, stride=4)

        # Define a special extra parameter to learn scalar sig_x for all pixels
        self.log_sig_x = nn.Parameter(torch.zeros(()))

    def infer(self, x):
        # Map (batch of) x to (batch of) phi which can then be passed to rsample to get z

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        flat_x = torch.flatten(x, 1)  # x.view(x.size()[0], -1)
        mu = self.q_fc_mu(flat_x)
        sig = self.q_fc_sig(flat_x)

        return mu, sig

    def generate(self, zs):
        # Map [b,n,k] sized samples of z to [b,n,p] sized images
        # Note that for the purposes of passing through the generator, we need
        # to reshape zs to be size [b*n,k]
        b, k = zs.size()
        # print('zs: ', zs.size())
        # s = zs.view(b, -1)

        s = F.relu(self.p_fc_upsample(zs))
        s = s.view([b] + list(self.shape_pre_flatten[1:]))
        s = F.relu(self.p_deconv_1(s))
        s = F.relu(self.p_deconv_2(s))
        s = self.p_deconv_3(s)
        mu_xs = s.view(b, -1)

        return mu_xs

    def elbo(self, x):
        # Run input end to end through the VAE and compute the ELBO using n samples of z

        mu, sig = self.infer(x)
        zs = rsample(mu, sig)
        mu_xs = self.generate(zs)

        return log_p_x(x, mu_xs, self.log_sig_x.exp()) - kl_q_p(zs, mu, sig)

    def get_sample(self, x):
        x = x.reshape([1] + list(x.shape))
        mu, sig = self.infer(x)
        zs = rsample(mu, sig)
        mu_xs = self.generate(zs)

        return mu_xs

    def getNeuronNum(self, x):
        # Pass an arbitrary input x through the network to see how many neurons are needed in the linear layer
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        flat_x = torch.flatten(x, 1)
        return flat_x.numel(), x.shape


def kl_q_p(zs, mu_q, log_sig_q):
    """Given [b,n,k] samples of z drawn from q, compute estimate of KL(q||p).
    phi must be size [b,k+1]
    This uses mu_p = 0 and sigma_p = 1, which simplifies the log(p(zs)) term to
    just -1/2*(zs**2)
    """

    # b, k = zs.size()
    # mu_q, log_sig_q = phi[:,:-1], phi[:,-1]
    log_p = -0.5 * (zs ** 2)
    log_q = -0.5 * (zs - mu_q) ** 2 / log_sig_q.exp() ** 2 - log_sig_q

    # Size of log_q and log_p is [b,n,k]. Sum along [k] but mean along [b,n]
    return (log_q - log_p).sum(dim=1).mean(dim=0)


def log_p_x(x, mu_xs, sig_x):
    """Given [batch, ...] input x and [batch, n, ...] reconstructions, compute
    pixel-wise log Gaussian probability
    Sum over pixel dimensions, but mean over batch and samples.
    """
    b, n = mu_xs.size()[:2]
    # Flatten out pixels and add a singleton dimension [1] so that x will be
    # implicitly expanded when combined with mu_xs
    x = x.reshape(b, -1)
    _, p = x.size()
    squared_error = (x - mu_xs) ** 2 / (2 * sig_x ** 2)

    # Size of squared_error is [b,n,p]. log prob is by definition sum over [p].
    # Expected value requires mean over [n]. Handling different size batches
    # requires mean over [b].
    return -(squared_error + torch.log(sig_x)).sum(dim=1).mean(dim=(0))


def rsample(mu, sig):
    """Sample z ~ q(z;phi)
    Ouput z is size [b,n_samples,K] given phi with shape [b,K+1]. The first K
    entries of each row of phi are the mean of q, and phi[:,-1] is the log
    standard deviation
    """
    b, k = mu.size()  # phi.size() = [b,k] / [256,2]

    # print('u: ', mu.size(), ' sig: ', sig.size())

    eps = torch.randn(b, k, device=mu.device)  # eps.size() = [b,n,k] / [256,1,2]

    return mu + eps * torch.exp(sig)


def train_vae(vae, X_train, X_validate, epochs=10, batch_size=64):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3, weight_decay=0)
    vae.to(device)
    vae.train()
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(X_validate, batch_size=batch_size, shuffle=True, pin_memory=True)
    train_losses = []
    validation_losses = []
    for _ in tqdm_bar(range(epochs)):
        vae.train()
        for im in train_loader:  # tqdm_bar(loader, total=len(dataset) / batch_size):

            im = im.to(device)
            opt.zero_grad()
            train_loss = -vae.elbo(im)
            train_loss.backward()
            opt.step()
            train_losses.append(-train_loss.item())

        vae.eval()
        for im in validation_loader:
            im = im.to(device)
            val_loss = -vae.elbo(im)
            validation_losses.append(-val_loss.item())

    return train_losses, validation_losses


def set_seed(seed=None, seed_torch=True):
    if seed is None:
        seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f'Random seed {seed} has been set.')

def Transform_Image(_img, _n):
    # Highlight what we care about -- lunar lander.
    _x = _img
    _x = _x * (_x < 0.8) * (_x > 0.4) # Mask anything outside 0.4 < _x < 0.8 
    _x[50:52, 32:34] = 0 # remove flags
    _x[50:52, 48:50] = 0
    _y = _n * _x + _img # highlight items between region, add to original image.

    return _y


if __name__ == "__main__":
    set_seed(seed=2021)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    path = 'dataset/dataset.npy'

    LOAD_MODEL = False
    TRAIN = True
    SAVE = True
    PLOT = True

    X = np.load(path, allow_pickle=True)
    for i in range(len(X)):
        for j in range(len(X[0])):
            X[i,j] = Transform_Image(X[i,j], 3)

    X = torch.tensor(X, dtype=torch.float32)
    n_samples = X.shape[0]
    training_samples = round(n_samples * 0.8)

    X_train = X[:training_samples, :, :, :]
    X_validate = X[training_samples:, :, :, :]

    vae = ConvVAE(K=20, input_dims=X_train.shape[1:], filter1=32, filter2=64)
    if LOAD_MODEL:
        model_location = 'VAE_Checkpoint_Lunar.pt'
        vae.load_state_dict(torch.load(model_location))

    if TRAIN:
        train_losses, validation_losses = train_vae(vae, X_train, X_validate, epochs=50, batch_size=64)

        if SAVE:
            torch.save(vae.state_dict(), 'VAE_Checkpoint_Lunar.pt')
            np.save('train_losses.npy', train_losses)
            np.save('validation_losses.npy', validation_losses)

            plt.figure()
            plt.plot(train_losses, label="train")
            plt.plot(np.arange(0, len(validation_losses) * 4, 4), validation_losses, label="validation")
            plt.legend()
            plt.xlabel('Batch #')
            plt.ylabel('ELBO')
            plt.show()


    if PLOT:
      vae.to('cpu')
      for i in range(100,110):
        X_train_sample = X_train[i, :, :, :]
        reconstructed_sample = vae.get_sample(X_train_sample)

        plt.figure()

        plt.subplot(2, 3, 1)
        plt.imshow(X_train_sample[0])

        plt.subplot(2, 3, 2)
        plt.imshow(X_train_sample[1])

        plt.subplot(2, 3, 3)
        plt.imshow(X_train_sample[2])


        img = reconstructed_sample.view(X_train_sample.shape).to('cpu').detach().numpy()

        plt.subplot(2, 3, 4)
        plt.imshow(img[0])

        plt.subplot(2, 3, 5)
        plt.imshow(img[1])

        plt.subplot(2, 3, 6)
        plt.imshow(img[2])


        plt.show()
