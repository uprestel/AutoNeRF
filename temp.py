class AutoencoderConcatTrainer(Trainer):
    """
    Concatenates a given neural network (e.g. a classifier) and an autoencoder.
    Example: see configs/alexnet
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = kwargs["model"]
        self.autoencoder = kwargs["autoencoder"]

	def eval_op_samples(self, xin, zae, zrep):
		"""
			Used for generating a forward sample of the cINN.
			xin: the imput image of 
			zae: the latent code in the domain Dy
			zrep: the latent code in the domain Dx
		"""
		# First we go from domain Dy to the sample space
		zz, _ = self.model(zae, zrep)

		# Using this sample, we instantiate new samples v ~ N(0,1)
		zz_sample = torch.randn_like(zz)
		zae_sample = self.model.reverse(zz_sample, zrep)
		xae_sample = self.autoencoder.decode(zae_sample)
		