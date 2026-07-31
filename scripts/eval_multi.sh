#!/bin/bash
#SBATCH --job-name=eval_multi
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_multi_%j.out
#SBATCH --error=logs/eval_multi_%j.err

source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate qwen9B

echo "=== Multi-SFT Per-Dataset Evaluation ==="
nvidia-smi -L

cd /home/wushaohua/data/risk-control-posttraining
python3 -c "
import json, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import roc_auc_score, average_precision_score

MODEL_ID='/data/share/model/Qwen3.5-4B'
ADAPTER='outputs/sft/german_multi/best_adapter'
SFT_DIR='data/processed/multi/combined/sft'

base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
model = PeftModel.from_pretrained(base, ADAPTER)
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model.eval()
print('Model loaded.')

def find_thresh(scores, gts):
    best_t, best_c = 0.5, float('inf')
    for t in np.arange(0.05, 0.96, 0.05):
        preds = (np.array(scores) >= t).astype(int)
        c = sum(5 if g==1 and p==0 else (1 if g==0 and p==1 else 0) for g,p in zip(gts,preds))
        if c < best_c: best_c = c; best_t = t
    return best_t, best_c

for split in ('valid', 'test'):
    with open(f'{SFT_DIR}/{split}.jsonl') as f:
        data = [json.loads(l) for l in f if l.strip()]

    all_gts, all_scores, all_ids = [], [], []
    per_ds = {}
    for r in data:
        ds = r['metadata']['dataset']
        msgs = r['messages']
        prompt = tok.apply_chat_template([{'role':'system','content':msgs[0]['content']},
                                           {'role':'user','content':msgs[1]['content']}],
                                          tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            lp = torch.nn.functional.log_softmax(model(**ids).logits[0,-1,:], dim=-1)
        s_low = lp[tok.encode('low',add_special_tokens=False)[0]].item()
        s_high = lp[tok.encode('high',add_special_tokens=False)[0]].item()
        p_high = np.exp(s_high)/(np.exp(s_low)+np.exp(s_high))
        gt = r['metadata']['risk_label']
        all_gts.append(gt); all_scores.append(p_high); all_ids.append(r['metadata']['sample_id'])
        if ds not in per_ds: per_ds[ds] = {'gt':[], 'score':[], 'id':[]}
        per_ds[ds]['gt'].append(gt); per_ds[ds]['score'].append(p_high); per_ds[ds]['id'].append(r['metadata']['sample_id'])

    print(f'\n{split} (N={len(all_gts)}):')
    for ds in sorted(per_ds):
        gts = per_ds[ds]['gt']; scs = per_ds[ds]['score']
        auc = roc_auc_score(gts, scs) if len(set(gts))>1 else 0.5
        pr = average_precision_score(gts, scs) if len(set(gts))>1 else 0
        best_t, best_c = find_thresh(scs, gts)
        hr_pct = sum(1 for s in scs if s>=best_t)/len(scs)*100
        print(f'  {ds:12s}: N={len(gts):3d} ROC-AUC={auc:.4f} PR-AUC={pr:.4f} thresh={best_t:.2f} cost={best_c} HR%={hr_pct:.1f}%')

    # Overall
    auc = roc_auc_score(all_gts, all_scores)
    pr = average_precision_score(all_gts, all_scores)
    best_t, best_c = find_thresh(all_scores, all_gts)
    print(f'  {\"OVERALL\":12s}: N={len(all_gts):3d} ROC-AUC={auc:.4f} PR-AUC={pr:.4f} thresh={best_t:.2f} cost={best_c}')

print('\nDone.')
"
echo "--- Done ---"
