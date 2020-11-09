import torch
import torch.nn as nn
import torch.nn.functional as F



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")







class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        hidden_dims = [32, 64, 128, 256, 512, 1024]
        modules = []
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


    def forward(self, x):
        result=x
        for layer in self.encoder:
            result = layer(result)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]




class Decoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        hidden_dims = [32, 64, 128, 256, 512, 1024]
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        padding_pattern = [1,0,0,0,1]

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=padding_pattern[i]),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.Tanh())
            )



        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())

    def forward(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 1024, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result





class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encoder(input)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var, z

    def loss_function(self, recons, input, mu, log_var, kld_weight=1.):
        """
        Computes the VAE loss function.

        """
        
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return loss, recons_loss, -kld_loss


    def sample(self, num_samples, current_device):

        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples



if __name__ == "__main__":
    # simple dimension debugging

    vae = VAE(in_channels=3, latent_dim=64).to(device)
    print(vae.eval())
    #x = torch.randn(1, 3, 100,100).to(device)
    #y, z_mu, z_logsig, z = vae(x)
    #print(y.shape, "ssss")
