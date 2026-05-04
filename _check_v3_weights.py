import os, torch, glob
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import yaml
from modules.commons import recursive_munch, build_model

CONFIG = r'E:\AIscene\AISVCs\YingMusic-SVC\configs\YingMusic-SVC.yml'
PRE = r'E:\AIscene\AISVCs\YingMusic-SVC\YingMusic-SVC-full.pt'

def load_weights(path):
    config = yaml.safe_load(open(CONFIG))
    mp = recursive_munch(config['model_params'])
    model = build_model(mp, stage='DiT')
    state = torch.load(path, map_location='cpu')
    weights = state.get('net', state)
    return {k: weights[k] for k in model if k in weights}

pre_w = load_weights(PRE)

checkpoints = [
    ('v3_5000',        'output_models/spkemb_v3_lr3e-5_20k/DiT_epoch_00000_step_05000.pth'),
    ('v3_10000',       'output_models/spkemb_v3_lr3e-5_20k/DiT_epoch_00000_step_10000.pth'),
    ('v3_15000',       'output_models/spkemb_v3_lr3e-5_20k/DiT_epoch_00001_step_15000.pth'),
    ('v3_20000',       'output_models/spkemb_v3_lr3e-5_20k/ft_model.pth'),
    ('exp1_12000',     'output_models/yingmusic_exp1/ft_model.pth'),
]

prompt_style = torch.load('output_models/hanamaru_prompt_style.pt', map_location='cpu')

print(f"{'Model':<18s} {'CFM_cos':>9s} {'CFM_L2':>9s} {'LR_cos':>9s} {'LR_L2':>9s} {'spk_cos':>9s}")
print('-' * 70)

for tag, ckpt_path in checkpoints:
    try:
        ckpt_w = load_weights(ckpt_path)
        for m in ['cfm', 'length_regulator']:
            cos_sum, l2_sum, n = 0.0, 0.0, 0
            for k in ckpt_w[m]:
                if k in pre_w[m] and pre_w[m][k].shape == ckpt_w[m][k].shape:
                    cos_sum += torch.nn.functional.cosine_similarity(
                        pre_w[m][k].float().view(-1), ckpt_w[m][k].float().view(-1), dim=0).item()
                    l2_sum += (pre_w[m][k] - ckpt_w[m][k]).float().norm(p=2).item()
                    n += 1
            if m == 'cfm':
                cfm_cos, cfm_l2 = cos_sum/n if n else 1.0, l2_sum
            else:
                lr_cos, lr_l2 = cos_sum/n if n else 1.0, l2_sum

        state = torch.load(ckpt_path, map_location='cpu')
        spk_cos = 1.0
        if 'spk_embedding' in state:
            spk_w = state['spk_embedding']['weight'].squeeze(0)
            spk_cos = torch.nn.functional.cosine_similarity(prompt_style, spk_w, dim=0).item()

        print(f"{tag:<18s} {cfm_cos:9.6f} {cfm_l2:9.1f} {lr_cos:9.6f} {lr_l2:9.1f} {spk_cos:9.6f}")

    except Exception as e:
        print(f"{tag:<18s} ERROR: {e}")
