import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from dt.finetuning_core import FTViT
from models.transformer_eeg import TransformerEncoder


def get_lora_config(
    model: FTViT,
    rank: int,
    patch=False,
    mlp4d=False,
    attention=True,
    ffn=True,
):
    """
    Only works for FTViT models.
    Goal: fine-tune the wrapped TransformerEncoder model.
    """

    target_modules = []
    encoder = model.encoder

    if any([patch, mlp4d, attention, ffn]) is False:
        raise ValueError("At least one of the flags must be True.")

    # Patch Embedding
    if patch:
        target_modules.append("to_patch_embedding.0")

    # MLP4D
    if mlp4d:
        target_modules.append("mlp4d.0")

    # Transformer
    for i in range(len(encoder.transformer.layers)):
        if attention:
            target_modules.extend([f"transformer.layers.{i}.0.to_qkv", f"layers.{i}.0.to_out"])
        if ffn:
            target_modules.extend([f"transformer.layers.{i}.1.net.1", f"layers.{i}.1.net.3"])

    return LoraConfig(r=rank, target_modules=target_modules)


class CustomGetLora:
    def __init__(self, config, train_all=False):
        """
        Initializes the Lora class with the given configuration.
        Args:
            config (LoraConfig): The configuration for the Lora model.
            train_all (bool, optional): A flag indicating whether to train all parameters. Defaults to False.
        If train_all is False, only the Lora parameters are trained.
        Else, all (except the base layer) parameters are trained.
        """

        self.config = config
        self.train_all = train_all

    def get_model(self, model: nn.Module):
        ret = get_peft_model(model, self.config)

        if self.train_all:
            for name, param in ret.named_parameters():
                if "base_layer" not in name:
                    param.requires_grad = True

        return ret

    def get_opt_params(self, model, verbose=False):
        ret = []
        for param in model.linear_head.parameters():
            param.requires_grad = True
        model.cls_query_token.requires_grad = True
        for name, param in model.named_parameters():
            if param.requires_grad:
                ret.append(param)
            elif verbose:
                print(f"Skipping {name}")

        return ret


def test_ftvit():
    model = FTViT(encoder=TransformerEncoder(), n_classes=10)

    config = get_lora_config(
        model=model,
        rank=4,
        patch=True,
        mlp4d=True,
        attention=True,
        ffn=True,
    )
    lora = CustomGetLora(config=config, train_all=False)

    lora_model = lora.get_model(model)
    lora_model.print_trainable_parameters()

    params = lora.get_opt_params(lora_model)
    opt = torch.optim.Adam(params, lr=1e-3)
    print(opt)

