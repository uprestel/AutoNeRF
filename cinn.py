"""Flow consisting of ActNorm, DoubleVectorCouplingBlock and Shuffle. Additionally, powerful conditioning encodings are
learned."""
import torch
import torch.nn as nn
import numpy as np


from blocks import ActNorm, ConditionalFlow, FeatureLayer, DenseEncoderLayer


def retrieve(
    list_or_dict, key, splitval="/", default=None, expand=True, pass_success=False
):
    """Given a nested list or dict return the desired value at key expanding
    callable nodes if necessary and :attr:`expand` is ``True``. The expansion
    is done in-place.
    Parameters
    ----------
        list_or_dict : list or dict
            Possibly nested list or dictionary.
        key : str
            key/to/value, path like string describing all keys necessary to
            consider to get to the desired value. List indices can also be
            passed here.
        splitval : str
            String that defines the delimiter between keys of the
            different depth levels in `key`.
        default : obj
            Value returned if :attr:`key` is not found.
        expand : bool
            Whether to expand callable nodes on the path or not.
    Returns
    -------
        The desired value or if :attr:`default` is not ``None`` and the
        :attr:`key` is not found returns ``default``.
    Raises
    ------
        Exception if ``key`` not in ``list_or_dict`` and :attr:`default` is
        ``None``.
    """

    keys = key.split(splitval)

    success = True
    try:
        visited = []
        parent = None
        last_key = None
        for key in keys:
            if callable(list_or_dict):
                if not expand:
                    raise KeyNotFoundError(
                        ValueError(
                            "Trying to get past callable node with expand=False."
                        ),
                        keys=keys,
                        visited=visited,
                    )
                list_or_dict = list_or_dict()
                parent[last_key] = list_or_dict

            last_key = key
            parent = list_or_dict

            try:
                if isinstance(list_or_dict, dict):
                    list_or_dict = list_or_dict[key]
                else:
                    list_or_dict = list_or_dict[int(key)]
            except (KeyError, IndexError, ValueError) as e:
                raise KeyNotFoundError(e, keys=keys, visited=visited)

            visited += [key]
        # final expansion of retrieved value
        if expand and callable(list_or_dict):
            list_or_dict = list_or_dict()
            parent[last_key] = list_or_dict
    except KeyNotFoundError as e:
        if default is None:
            raise e
        else:
            list_or_dict = default
            success = False

    if not pass_success:
        return list_or_dict
    else:
        return list_or_dict, success






class DenseEmbedder(nn.Module):
    """Basically an MLP. Maps vector-like features to some other vector of given dimenionality"""

    def __init__(self, in_dim, up_dim, depth=4, given_dims=None):
        super().__init__()
        self.net = nn.ModuleList()
        if given_dims is not None:
            assert given_dims[0] == in_dim
            assert given_dims[-1] == up_dim
            dims = given_dims
        else:
            dims = np.linspace(in_dim, up_dim, depth).astype(int)
        for l in range(len(dims) - 2):
            self.net.append(nn.Conv2d(dims[l], dims[l + 1], 1))
            self.net.append(ActNorm(dims[l + 1]))
            self.net.append(nn.LeakyReLU(0.2))

        self.net.append(nn.Conv2d(dims[-2], dims[-1], 1))

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x.squeeze(-1).squeeze(-1)


class Embedder(nn.Module):
    """Embeds a 4-dim tensor onto dense latent code."""

    def __init__(self, in_spatial_size, in_channels, emb_dim, n_down=4):
        super().__init__()
        self.feature_layers = nn.ModuleList()
        norm = 'an'  # hard coded
        bottleneck_size = in_spatial_size // 2 ** n_down
        self.feature_layers.append(FeatureLayer(0, in_channels=in_channels, norm=norm))
        for scale in range(1, n_down):
            self.feature_layers.append(FeatureLayer(scale, norm=norm))
        self.dense_encode = DenseEncoderLayer(n_down, bottleneck_size, emb_dim)
        if n_down == 1:
            print(" Warning: Embedder for ConditionalTransformer has only one down-sampling step. You might want to "
                  "increase its capacity.")

    def forward(self, input):
        h = input
        for layer in self.feature_layers:
            h = layer(h)
        h = self.dense_encode(h)
        return h.squeeze(-1).squeeze(-1)


class ConditionalTransformer(nn.Module):
    """
    Conditional Invertible Neural Network.
    Can be conditioned both on input with spatial dimension (i.e. a tensor of shape BxCxHxW) and a flat input
    (i.e. a tensor of shape BxC)
    """
    def __init__(self, config):
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        super().__init__()
        self.config = config
        # get all the hyperparameters
        in_channels = retrieve(config, "Transformer/in_channels")
        mid_channels = retrieve(config, "Transformer/mid_channels")
        hidden_depth = retrieve(config, "Transformer/hidden_depth")
        n_flows = retrieve(config, "Transformer/n_flows")
        conditioning_option = retrieve(config, "Transformer/conditioning_option")
        flowactivation = retrieve(config, "Transformer/activation", default="lrelu")
        embedding_channels = retrieve(config, "Transformer/embedding_channels", default=in_channels)
        n_down = retrieve(config, "Transformer/embedder_down", default=4)

        self.emb_channels = embedding_channels
        self.in_channels = in_channels

        self.flow = ConditionalFlow(in_channels=in_channels, embedding_dim=self.emb_channels, hidden_dim=mid_channels,
                                    hidden_depth=hidden_depth, n_flows=n_flows, conditioning_option=conditioning_option,
                                    activation=flowactivation)
        conditioning_spatial_size = retrieve(config, "Transformer/conditioning_spatial_size")
        conditioning_in_channels = retrieve(config, "Transformer/conditioning_in_channels")
        if conditioning_spatial_size == 1:
            depth = retrieve(config, "Transformer/conditioning_depth",
                             default=4)
            dims = retrieve(config, "Transformer/conditioning_dims",
                            default="none")
            dims = None if dims == "none" else dims
            self.embedder = DenseEmbedder(conditioning_in_channels,
                                          in_channels,
                                          depth=depth,
                                          given_dims=dims)
        else:
            self.embedder = Embedder(conditioning_spatial_size, conditioning_in_channels, in_channels, n_down=n_down)

    def embed(self, conditioning):
        # embed it via embedding layer
        embedding = self.embedder(conditioning)
        return embedding

    def sample(self, shape, conditioning):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde, self.embed(conditioning))
        return sample

    def forward(self, input, conditioning, train=False):
        embedding = self.embed(conditioning)
        out, logdet = self.flow(input, embedding)
        if train:
            return self.flow.last_outs, self.flow.last_logdets
        return out, logdet

    def reverse(self, out, conditioning):
        embedding = self.embed(conditioning)
        return self.flow(out, embedding, reverse=True)

    def get_last_layer(self):
        return getattr(self.flow.sub_layers[-1].coupling.t[-1].main[-1], 'weight')

    # @classmethod
    # def from_pretrained(cls, name, config=None):
    #     if name not in URL_MAP:
    #         raise NotImplementedError(name)
    #     if config is None:
    #         config = CONFIG_MAP[name]
    #
    #     model = cls(config)
    #     ckpt = get_ckpt_path(name)
    #     model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")))
    #     model.eval()
    #     return model
    
    
CONFIG_MAP = {
    "cinn_alexnet_aae_conv5":
        {"Transformer": {
              "activation": "none",
              "conditioning_option": "none",
              "hidden_depth": 2,
              "in_channels": 128,
              "mid_channels": 1024,
              "n_flows": 20,
              "conditioning_in_channels": 256,
              "conditioning_spatial_size": 13,
              "embedder_down": 2,
            }
        },
    "cinn_alexnet_aae_fc6":
        {"Transformer": {
                      "activation": "none",
                      "conditioning_option": "none",
                      "hidden_depth": 2,
                      "in_channels": 128,
                      "mid_channels": 1024,
                      "n_flows": 20,
                      "conditioning_in_channels": 4096,
                      "conditioning_spatial_size": 1,
                      "embedder_down": 3,
                    }
                },
    "cinn_alexnet_aae_fc7":
        {"Transformer": {
                      "activation": "none",
                      "conditioning_option": "none",
                      "hidden_depth": 2,
                      "in_channels": 128,
                      "mid_channels": 1024,
                      "n_flows": 20,
                      "conditioning_in_channels": 4096,
                      "conditioning_spatial_size": 1,
                      "embedder_down": 3,
                    }
                },
    "cinn_alexnet_aae_fc8":
        {"Transformer": {
                      "activation": "none",
                      "conditioning_option": "none",
                      "hidden_depth": 2,
                      "in_channels": 128,
                      "mid_channels": 1024,
                      "n_flows": 20,
                      "conditioning_in_channels": 1000,
                      "conditioning_spatial_size": 1,
                      "embedder_down": 3,
                    }
                },
    "cinn_alexnet_aae_softmax":
        {"Transformer": {
            "activation": "none",
            "conditioning_option": "none",
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 1000,
            "conditioning_spatial_size": 1,
            "embedder_down": 3,
            }
        },
    "cinn_stylizedresnet_avgpool":
        {"Transformer": {
            "activation": "none",
            "conditioning_option": "none",
            "hidden_depth": 2,
            "in_channels": 268,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 2048,
            "conditioning_spatial_size": 1,
            "embedder_down": 3,
            }
        },
    "cinn_resnet_avgpool":
        {"Transformer": {
            "activation": "none",
            "conditioning_option": "none",
            "hidden_depth": 2,
            "in_channels": 268,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 2048,
            "conditioning_spatial_size": 1,
            "embedder_down": 3,
            }
        },
    "resnet101_animalfaces_shared":
        {"Model": {
            "n_classes": 149,
            "type": "resnet101"
            }
        },

    "resnet101_animalfaces_10":
        {"Model": {
                "n_classes": 10,
                "type": "resnet101"
                }
        },
    "cinn_resnet_animalfaces10_ae_maxpool":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 64,
            "conditioning_spatial_size": 56,
            "embedder_down": 4,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
    "cinn_resnet_animalfaces10_ae_input":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 3,
            "conditioning_spatial_size": 224,
            "embedder_down": 5,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
    "cinn_resnet_animalfaces10_ae_layer1":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 256,
            "conditioning_spatial_size": 56,
            "embedder_down": 4,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
    "cinn_resnet_animalfaces10_ae_layer2":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 512,
            "conditioning_spatial_size": 28,
            "embedder_down": 3,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
    "cinn_resnet_animalfaces10_ae_layer3":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 1024,
            "conditioning_spatial_size": 14,
            "embedder_down": 2,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
    "cinn_resnet_animalfaces10_ae_layer4":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 2048,
            "conditioning_spatial_size": 7,
            "embedder_down": 1,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
    "cinn_resnet_animalfaces10_ae_avgpool":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 2048,
            "conditioning_spatial_size": 1,
            "conditioning_depth": 6,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
    "cinn_resnet_animalfaces10_ae_fc":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 10,
            "conditioning_spatial_size": 1,
            "conditioning_depth": 4,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
    "cinn_resnet_animalfaces10_ae_softmax":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 10,
            "conditioning_spatial_size": 1,
            "conditioning_depth": 4,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
}
