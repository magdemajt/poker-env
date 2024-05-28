from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

class CustomModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.fc1 = nn.LSTM(obs_space.shape[0], 64, 2)
        self.fc2 = nn.LSTM(64, 64, 2)
        self.fc3 = nn.Linear(64, num_outputs)

    def forward(self, input_dict, state, seq_lens):
        x = torch.relu(self.fc1(input_dict["obs"].float()))
        x = torch.relu(self.fc2(x))
        output = self.fc3(x)
        return output, state

# Register the custom model in ModelCatalog
ModelCatalog.register_custom_model("my_model", CustomModel)