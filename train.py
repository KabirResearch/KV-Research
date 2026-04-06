import torch
import torch.nn as nn
from models import VoCModel
from utils import load_model
from data import load_dataset_part
import json
import wandb

with open('config.json') as f:
    config = json.load(f)

wandb.init(project="layer-skipping", config=config)

def train_voc():
    model, tokenizer = load_model()
    dataset = load_dataset_part()
    features, targets = [], []

    def hook_fn(module, input, output):
        layer_id = module.layer_id
        h_in = input[0].detach().float()
        h_out = output[0].detach().float()
        delta = (h_out - h_in).abs().mean(dim=[1,2])
        delta = torch.clamp(delta, 0, 10)
        feat = torch.stack([
            h_in.norm(dim=-1).mean(dim=1),
            h_in.abs().mean(dim=[1,2]),
            torch.full_like(delta, layer_id / 24.0)
        ], dim=1)
        features.append(feat.cpu())
        targets.append(delta.cpu())

    hooks = []
    for layer in model.gpt_neox.layers:
        layer.layer_id = id(layer)  # or something, but in notebook it's i
        hooks.append(layer.register_forward_hook(hook_fn))

    for i, batch in enumerate(dataset):
        if i > config['voc_collect_samples']:
            break
        input_ids = batch["input_ids"].unsqueeze(0).cuda()
        with torch.no_grad():
            _ = model(input_ids)

    for h in hooks:
        h.remove()

    X = torch.cat(features)
    Y = torch.cat(targets)
    mask = torch.isfinite(X).all(dim=1) & torch.isfinite(Y)
    X = X[mask]
    Y = Y[mask]
    X_mean, X_std = X.mean(0), X.std(0) + 1e-6
    Y_mean, Y_std = Y.mean(), Y.std() + 1e-6
    X = (X - X_mean) / X_std
    Y = (Y - Y_mean) / Y_std
    dataset_voc = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset_voc, batch_size=config['batch_size'], shuffle=True)

    voc_model = VoCModel().cuda().float()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(voc_model.parameters(), lr=config['learning_rate'])

    for epoch in range(config['epochs']):
        total_loss = 0
        for x, y in dataloader:
            x = x.float().cuda()
            y = y.float().cuda().unsqueeze(1)
            pred = voc_model(x)
            loss = criterion(pred, y)
            if torch.isnan(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(voc_model.parameters(), config['clip_grad_norm'])
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        wandb.log({"epoch": epoch, "loss": avg_loss})
        print(f"epoch {epoch} loss {avg_loss:.4f}")

    return voc_model

def train_router(router, records):
    router.train()
    opt = torch.optim.Adam(router.parameters(), lr=config['learning_rate'])
    for epoch in range(config['epochs']):
        for r in records:
            x = r["feat"].cuda()
            y = torch.tensor([r["label"]]).cuda()
            pred = router(x).squeeze()
            loss = (pred - y) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
    return router