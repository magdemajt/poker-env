from ray.rllib.utils.typing import ModelConfigDict


MODEL_DEFAULTS: ModelConfigDict = {
    # Experimental flag.
    # If True, user specified no preprocessor to be created
    # (via config._disable_preprocessor_api=True). If True, observations
    # will arrive in model as they are returned by the env.
    "_disable_preprocessor_api": False,
    # Experimental flag.
    # If True, RLlib will no longer flatten the policy-computed actions into
    # a single tensor (for storage in SampleCollectors/output files/etc..),
    # but leave (possibly nested) actions as-is. Disabling flattening affects:
    # - SampleCollectors: Have to store possibly nested action structs.
    # - Models that have the previous action(s) as part of their input.
    # - Algorithms reading from offline files (incl. action information).
    "_disable_action_flattening": False,

    # === Built-in options ===
    # FullyConnectedNetwork (tf and torch): rllib.models.tf|torch.fcnet.py
    # These are used if no custom model is specified and the input space is 1D.
    # Number of hidden layers to be used.
    "fcnet_hiddens": [256, 256],
    # Activation function descriptor.
    # Supported values are: "tanh", "relu", "swish" (or "silu", which is the same),
    # "linear" (or None).
    "fcnet_activation": "tanh",
    # Initializer function or class descriptor for encoder weigths.
    # Supported values are the initializer names (str), classes or functions listed
    # by the frameworks (`tf2``, `torch`). See
    # https://pytorch.org/docs/stable/nn.init.html for `torch` and
    # https://www.tensorflow.org/api_docs/python/tf/keras/initializers for `tf2`.
    # Note, if `None`, the default initializer defined by `torch` or `tf2` is used.
    "fcnet_weights_initializer": None,
    # Initializer configuration for encoder weights.
    # This configuration is passed to the initializer defined in
    # `fcnet_weights_initializer`.
    "fcnet_weights_initializer_config": None,
    # Initializer function or class descriptor for encoder bias.
    # Supported values are the initializer names (str), classes or functions listed
    # by the frameworks (`tf2``, `torch`). See
    # https://pytorch.org/docs/stable/nn.init.html for `torch` and
    # https://www.tensorflow.org/api_docs/python/tf/keras/initializers for `tf2`.
    # Note, if `None`, the default initializer defined by `torch` or `tf2` is used.
    "fcnet_bias_initializer": None,
    # Initializer configuration for encoder bias.
    # This configuration is passed to the initializer defined in
    # `fcnet_bias_initializer`.
    "fcnet_bias_initializer_config": None,

    # VisionNetwork (tf and torch): rllib.models.tf|torch.visionnet.py
    # These are used if no custom model is specified and the input space is 2D.
    # Filter config: List of [out_channels, kernel, stride] for each filter.
    # Example:
    # Use None for making RLlib try to find a default filter setup given the
    # observation space.
    "conv_filters": None,
    # Activation function descriptor.
    # Supported values are: "tanh", "relu", "swish" (or "silu", which is the same),
    # "linear" (or None).
    "conv_activation": "relu",
    # Initializer function or class descriptor for CNN encoder kernel.
    # Supported values are the initializer names (str), classes or functions listed
    # by the frameworks (`tf2``, `torch`). See
    # https://pytorch.org/docs/stable/nn.init.html for `torch` and
    # https://www.tensorflow.org/api_docs/python/tf/keras/initializers for `tf2`.
    # Note, if `None`, the default initializer defined by `torch` or `tf2` is used.
    "conv_kernel_initializer": None,
    # Initializer configuration for CNN encoder kernel.
    # This configuration is passed to the initializer defined in
    # `conv_weights_initializer`.
    "conv_kernel_initializer_config": None,
    # Initializer function or class descriptor for CNN encoder bias.
    # Supported values are the initializer names (str), classes or functions listed
    # by the frameworks (`tf2``, `torch`). See
    # https://pytorch.org/docs/stable/nn.init.html for `torch` and
    # https://www.tensorflow.org/api_docs/python/tf/keras/initializers for `tf2`.
    # Note, if `None`, the default initializer defined by `torch` or `tf2` is used.
    "conv_bias_initializer": None,
    # Initializer configuration for CNN encoder bias.
    # This configuration is passed to the initializer defined in
    # `conv_bias_initializer`.
    "conv_bias_initializer_config": None,
    # Initializer function or class descriptor for CNN head (pi, Q, V) kernel.
    # Supported values are the initializer names (str), classes or functions listed
    # by the frameworks (`tf2``, `torch`). See
    # https://pytorch.org/docs/stable/nn.init.html for `torch` and
    # https://www.tensorflow.org/api_docs/python/tf/keras/initializers for `tf2`.
    # Note, if `None`, the default initializer defined by `torch` or `tf2` is used.
    "conv_transpose_kernel_initializer": None,
    # Initializer configuration for CNN head (pi, Q, V) kernel.
    # This configuration is passed to the initializer defined in
    # `conv_transpose_weights_initializer`.
    "conv_transpose_kernel_initializer_config": None,
    # Initializer function or class descriptor for CNN head (pi, Q, V) bias.
    # Supported values are the initializer names (str), classes or functions listed
    # by the frameworks (`tf2``, `torch`). See
    # https://pytorch.org/docs/stable/nn.init.html for `torch` and
    # https://www.tensorflow.org/api_docs/python/tf/keras/initializers for `tf2`.
    # Note, if `None`, the default initializer defined by `torch` or `tf2` is used.
    "conv_transpose_bias_initializer": None,
    # Initializer configuration for CNN head (pi, Q, V) bias.
    # This configuration is passed to the initializer defined in
    # `conv_transpose_bias_initializer`.
    "conv_transpose_bias_initializer_config": None,

    # Some default models support a final FC stack of n Dense layers with given
    # activation:
    # - Complex observation spaces: Image components are fed through
    #   VisionNets, flat Boxes are left as-is, Discrete are one-hot'd, then
    #   everything is concated and pushed through this final FC stack.
    # - VisionNets (CNNs), e.g. after the CNN stack, there may be
    #   additional Dense layers.
    # - FullyConnectedNetworks will have this additional FCStack as well
    # (that's why it's empty by default).
    "post_fcnet_hiddens": [],
    "post_fcnet_activation": "relu",
    # Initializer function or class descriptor for head (pi, Q, V) weights.
    # Supported values are the initializer names (str), classes or functions listed
    # by the frameworks (`tf2``, `torch`). See
    # https://pytorch.org/docs/stable/nn.init.html for `torch` and
    # https://www.tensorflow.org/api_docs/python/tf/keras/initializers for `tf2`.
    # Note, if `None`, the default initializer defined by `torch` or `tf2` is used.
    "post_fcnet_weights_initializer": None,
    # Initializer configuration for head (pi, Q, V) weights.
    # This configuration is passed to the initializer defined in
    # `post_fcnet_weights_initializer`.
    "post_fcnet_weights_initializer_config": None,
    # Initializer function or class descriptor for head (pi, Q, V) bias.
    # Supported values are the initializer names (str), classes or functions listed
    # by the frameworks (`tf2``, `torch`). See
    # https://pytorch.org/docs/stable/nn.init.html for `torch` and
    # https://www.tensorflow.org/api_docs/python/tf/keras/initializers for `tf2`.
    # Note, if `None`, the default initializer defined by `torch` or `tf2` is used.
    "post_fcnet_bias_initializer": None,
    # Initializer configuration for head (pi, Q, V) bias.
    # This configuration is passed to the initializer defined in
    # `post_fcnet_bias_initializer`.
    "post_fcnet_bias_initializer_config": None,

    # For DiagGaussian action distributions, make the second half of the model
    # outputs floating bias variables instead of state-dependent. This only
    # has an effect is using the default fully connected net.
    "free_log_std": False,
    # Whether to skip the final linear layer used to resize the hidden layer
    # outputs to size `num_outputs`. If True, then the last hidden layer
    # should already match num_outputs.
    "no_final_linear": False,
    # Whether layers should be shared for the value function.
    "vf_share_layers": True,

    # == LSTM ==
    # Whether to wrap the model with an LSTM.
    "use_lstm": False,
    # Max seq len for training the LSTM, defaults to 20.
    "max_seq_len": 20,
    # Size of the LSTM cell.
    "lstm_cell_size": 256,
    # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
    "lstm_use_prev_action": False,
    # Whether to feed r_{t-1} to LSTM.
    "lstm_use_prev_reward": False,
    # Initializer function or class descriptor for LSTM weights.
    # Supported values are the initializer names (str), classes or functions listed
    # by the frameworks (`tf2``, `torch`). See
    # https://pytorch.org/docs/stable/nn.init.html for `torch` and
    # https://www.tensorflow.org/api_docs/python/tf/keras/initializers for `tf2`.
    # Note, if `None`, the default initializer defined by `torch` or `tf2` is used.
    "lstm_weights_initializer": None,
    # Initializer configuration for LSTM weights.
    # This configuration is passed to the initializer defined in
    # `lstm_weights_initializer`.
    "lstm_weights_initializer_config": None,
    # Initializer function or class descriptor for LSTM bias.
    # Supported values are the initializer names (str), classes or functions listed
    # by the frameworks (`tf2``, `torch`). See
    # https://pytorch.org/docs/stable/nn.init.html for `torch` and
    # https://www.tensorflow.org/api_docs/python/tf/keras/initializers for `tf2`.
    # Note, if `None`, the default initializer defined by `torch` or `tf2` is used.
    "lstm_bias_initializer": None,
    # Initializer configuration for LSTM bias.
    # This configuration is passed to the initializer defined in
    # `lstm_bias_initializer`.
    "lstm_bias_initializer_config": None,
    # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
    "_time_major": False,

    # == Attention Nets (experimental: torch-version is untested) ==
    # Whether to use a GTrXL ("Gru transformer XL"; attention net) as the
    # wrapper Model around the default Model.
    "use_attention": False,
    # The number of transformer units within GTrXL.
    # A transformer unit in GTrXL consists of a) MultiHeadAttention module and
    # b) a position-wise MLP.
    "attention_num_transformer_units": 1,
    # The input and output size of each transformer unit.
    "attention_dim": 64,
    # The number of attention heads within the MultiHeadAttention units.
    "attention_num_heads": 1,
    # The dim of a single head (within the MultiHeadAttention units).
    "attention_head_dim": 32,
    # The memory sizes for inference and training.
    "attention_memory_inference": 50,
    "attention_memory_training": 50,
    # The output dim of the position-wise MLP.
    "attention_position_wise_mlp_dim": 32,
    # The initial bias values for the 2 GRU gates within a transformer unit.
    "attention_init_gru_gate_bias": 2.0,
    # Whether to feed a_{t-n:t-1} to GTrXL (one-hot encoded if discrete).
    "attention_use_n_prev_actions": 0,
    # Whether to feed r_{t-n:t-1} to GTrXL.
    "attention_use_n_prev_rewards": 0,

    # == Atari ==
    # Set to True to enable 4x stacking behavior.
    "framestack": True,
    # Final resized frame dimension
    "dim": 84,
    # (deprecated) Converts ATARI frame to 1 Channel Grayscale image
    "grayscale": False,
    # (deprecated) Changes frame to range from [-1, 1] if true
    "zero_mean": True,

    # === Options for custom models ===
    # Name of a custom model to use
    "custom_model": None,
    # Extra options to pass to the custom classes. These will be available to
    # the Model's constructor in the model_config field. Also, they will be
    # attempted to be passed as **kwargs to ModelV2 models. For an example,
    # see rllib/models/[tf|torch]/attention_net.py.
    "custom_model_config": {},
    # Name of a custom action distribution to use.
    "custom_action_dist": None,
    # Custom preprocessors are deprecated. Please use a wrapper class around
    # your environment instead to preprocess observations.
    "custom_preprocessor": None,

    # === Options for ModelConfigs in RLModules ===
    # The latent dimension to encode into.
    # Since most RLModules have an encoder and heads, this establishes an agreement
    # on the dimensionality of the latent space they share.
    # This has no effect for models outside RLModule.
    # If None, model_config['fcnet_hiddens'][-1] value will be used to guarantee
    # backward compatibility to old configs. This yields different models than past
    # versions of RLlib.
    "encoder_latent_dim": None,
    # Whether to always check the inputs and outputs of RLlib's default models for
    # their specifications. Input specifications are checked on failed forward passes
    # of the models regardless of this flag. If this flag is set to `True`, inputs and
    # outputs are checked on every call. This leads to a slow-down and should only be
    # used for debugging. Note that this flag is only relevant for instances of
    # RLlib's Model class. These are commonly generated from ModelConfigs in RLModules.
    "always_check_shapes": False,

}