import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# load model from HuggingFace
def load_model_from_HF(
    model_name: str, tokenizer_only: bool=False
) -> AutoTokenizer | tuple[AutoTokenizer, AutoModelForMaskedLM]:
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if tokenizer_only:
        return tokenizer

    # load model (eval-only)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        deterministic_eval=True,
        trust_remote_code=True
    )
    for param in model.parameters():
        param.requires_grad = False

    return tokenizer, model

# load model from a torch weights file
def load_model_from_file(
    model: torch.nn.Module,
    file_pth: str,
    from_lightning: bool = False,
    mode: str="eval"
) -> torch.nn.Module:
    ws = torch.load(file_pth, map_location="cpu", weights_only=True)

    # check if the output is from torch-lightning
    if from_lightning:
        ws = {k.split(".", 1)[1]: v for k, v in ws["state_dict"].items()}

    # load weights
    mis, une = model.load_state_dict(ws, strict=False)
    if mis:
        print(f"Warning: Missing keys: {mis}.")
    if une:
        print(f"Warning: Unexpected keys: {une}.")

    if mode == "eval":
        model.eval()
    return model
