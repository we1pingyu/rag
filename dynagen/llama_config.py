"""
The Llama model configurations and weight downloading utilities.

adopted from opt_config.py
"""

import dataclasses
import glob
import os
import numpy as np
from tqdm import tqdm


@dataclasses.dataclass(frozen=False)
class LlamaConfig:
    name: str = "Llama-2-7b-hf"
    org: str = "meta-llama"
    hf_token: str = ""
    hidden_act: str = "silu"
    input_dim: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    max_position_embeddings: int = 4096
    n_head: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: int = 32
    rms_norm_eps: float = 1e-05
    dtype: type = np.float16
    pad_token_id: int = 2
    vocab_size: int = 32000

    def model_bytes(self):
        h = self.input_dim
        intermediate = self.intermediate_size
        n_head = self.n_head
        head_dim = h // n_head
        return 2 * (
            self.vocab_size * h
            + self.num_hidden_layers
            * (
                # self-attention
                3 * h * h
                + h * h
                + head_dim // 2
                +
                # mlp
                3 * h * intermediate
                +
                # layer norm
                2 * h
            )
            +
            # head
            h
            + self.vocab_size * h
        )

    def cache_bytes(self, batch_size, seq_len):
        return 2 * batch_size * seq_len * self.num_hidden_layers * self.input_dim * 2

    def hidden_bytes(self, batch_size, seq_len):
        return batch_size * seq_len * self.input_dim * 2


def get_llama_config(name, **kwargs):
    if "/" in name:
        org = name.split("/")[0]
        name = name.split("/")[1]

    if "-chat" in name:
        arch_name = name.replace("-chat", "")
    else:
        arch_name = name

    if arch_name == "Llama-2-7b-hf":
        config = LlamaConfig(
            name=name,
            org=org,
            hf_token=kwargs.get("hf_token"),
            input_dim=4096,
            intermediate_size=11008,
            n_head=32,
            num_hidden_layers=32,
            num_key_value_heads=32,
        )
    elif arch_name == "Llama-2-13b-hf":
        config = LlamaConfig(
            name=name,
            org=org,
            hf_token=kwargs.get("hf_token"),
            input_dim=5120,
            intermediate_size=13824,
            n_head=40,
            num_hidden_layers=40,
            num_key_value_heads=40,
        )
    elif arch_name == "Llama-2-70b-hf":
        config = LlamaConfig(
            name=name,
            org=org,
            hf_token=kwargs.get("hf_token"),
            input_dim=8192,
            intermediate_size=28672,
            n_head=64,
            num_hidden_layers=80,
            num_key_value_heads=8,
        )
    elif "8B" in arch_name or "8b" in arch_name:
        config = LlamaConfig(
            name=name,
            org=org,
            hf_token=kwargs.get("hf_token"),
            input_dim=4096,
            intermediate_size=14336,
            n_head=32,
            num_hidden_layers=32,
            num_key_value_heads=8,
            vocab_size=128256,
        )
    elif "70B" in arch_name:
        config = LlamaConfig(
            name=name,
            org=org,
            hf_token=kwargs.get("hf_token"),
            input_dim=8192,
            intermediate_size=28672,
            n_head=64,
            num_hidden_layers=80,
            num_key_value_heads=8,
            vocab_size=128256,
        )
    elif "1B" in arch_name:
        config = LlamaConfig(
            name=name,
            org=org,
            hf_token=kwargs.get("hf_token"),
            input_dim=2048,
            intermediate_size=8192,
            n_head=32,
            num_hidden_layers=16,
            num_key_value_heads=8,
            vocab_size=128256,
        )
    elif "34b" in arch_name:
        config = LlamaConfig(
            name=name,
            org=org,
            hf_token=kwargs.get("hf_token"),
            input_dim=8192,
            intermediate_size=22016,
            n_head=64,
            num_hidden_layers=48,
            num_key_value_heads=8,
            vocab_size=32000,
        )
    else:
        raise ValueError(f"Invalid model name: {name}")

    return dataclasses.replace(config, **kwargs)


import numpy as np
import torch


def _download_llama_weights(model_name, org_name, path, hf_token):
    from huggingface_hub import snapshot_download
    import os
    import glob
    from safetensors import safe_open  # 导入 safetensors 库
    from tqdm import tqdm

    hf_model_name = org_name + "/" + model_name

    folder = snapshot_download(hf_model_name, allow_patterns=["*.bin", "*.safetensors"], token=hf_token)
    bin_files = glob.glob(os.path.join(folder, "*.bin"))
    safetensors_files = glob.glob(os.path.join(folder, "*.safetensors"))

    if "/" in model_name:
        model_name = model_name.split("/")[1]
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)

    for bin_file in tqdm(bin_files, desc="Convert .bin format"):
        state = torch.load(bin_file, map_location="cuda:0")
        for name, param in tqdm(state.items(), leave=False):
            name = name.replace("model.", "")
            param_path = os.path.join(path, name)
            if param.dtype == torch.bfloat16:
                param = param.to(torch.float16)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())

    for safe_file in tqdm(safetensors_files, desc="Convert .safetensors format"):
        with safe_open(safe_file, framework="pt", device="cuda:0") as f:
            for name in tqdm(f.keys(), leave=False):
                param = f.get_tensor(name)
                name = name.replace("model.", "")
                param_path = os.path.join(path, name)
                if param.dtype == torch.bfloat16:
                    param = param.to(torch.float16)
                with open(param_path, "wb") as f_out:
                    np.save(f_out, param.cpu().detach().numpy())


global torch_linear_init_backup
global torch_layer_norm_init_backup


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    global torch_linear_init_backup
    global torch_layer_norm_init_backup

    torch_linear_init_backup = torch.nn.Linear.reset_parameters
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)

    torch_layer_norm_init_backup = torch.nn.LayerNorm.reset_parameters
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def restore_torch_init():
    """Rollback the change made by disable_torch_init."""
    import torch

    setattr(torch.nn.Linear, "reset_parameters", torch_linear_init_backup)
    setattr(torch.nn.LayerNorm, "reset_parameters", torch_layer_norm_init_backup)


def download_llama_weights(model_name, org_name, path, hf_token):
    """Download weights from huggingface."""
    import torch
    from transformers import AutoModelForCausalLM, LlamaForCausalLM

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))

    hf_model_name = org_name + "/" + model_name
    model_class = LlamaForCausalLM

    print(
        f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
        f"The downloading and cpu loading can take dozens of minutes. "
        f"If it seems to get stuck, you can monitor the progress by "
        f"checking the memory usage of this process."
    )

    disable_torch_init()
    model = model_class.from_pretrained(hf_model_name, torch_dtype=torch.float16, trust_remote_code=True)
    restore_torch_init()

    os.makedirs(path, exist_ok=True)

    print(f"Convert the weights to numpy format under {path} ...")
    for name, param in tqdm(list(model.model.named_parameters())):
        param_path = os.path.join(path, name)
        with open(param_path, "wb") as f:
            np.save(f, param.cpu().detach().numpy())

    param_path = os.path.join(path, "lm_head.weight")
    with open(param_path, "wb") as f:
        np.save(f, model.lm_head.weight.detach().cpu().numpy())

    for idx, layer in enumerate(model.model.layers):
        rotary_emb = layer.self_attn.rotary_emb
        inv_freq_tensor = rotary_emb.inv_freq
        rotary_emb_path = os.path.join(path, f"layers.{idx}.self_attn.rotary_emb.inv_freq")
        with open(rotary_emb_path, "wb") as f:
            np.save(f, inv_freq_tensor.cpu().detach().numpy())
