import os
import pandas as pd
from collections import Counter
from configs.config import cfg

def ensemble_analysis_and_voting():
    files = {
        'model_224': os.path.join(cfg.output_dir, "submission_224.csv"), # 0.92 acc
        'model_448': os.path.join(cfg.output_dir, "submission_448.csv"), # 0.924 acc
        'model_tta': os.path.join(cfg.output_dir, "submission_448_tta.csv") # 0.932 acc
    }
    
    dfs = {k: pd.read_csv(v) for k, v in files.items()}
    ref_ids = dfs['model_448']['filename'].values
    for k, df in dfs.items():
        assert (df['filename'].values == ref_ids).all(), f"Orden incorrecto en {k}"

    final_preds = []
    conflicts = 0
    
    print("\n--- ANALIZANDO CONFLICTOS ---")
    
    for i in range(len(ref_ids)):
        votes = [dfs[k].iloc[i]['cls'] for k in dfs]
        counts = Counter(votes)
        most_common = counts.most_common()
        winner_cls, winner_votes = most_common[0]

        if winner_votes == 1:
            final_cls = dfs['model_tta'].iloc[i]['cls']
            print(f"Img {ref_ids[i]}: Desacuerdo total {votes}. Escogiendo TTA: {final_cls}")
            conflicts += 1
        elif winner_votes < 3:
            final_cls = winner_cls
            print(f"Img {ref_ids[i]}: Desacuerdo parcial {votes}. Escogiendo: {final_cls}")
            conflicts += 1
        else:
            final_cls = winner_cls
            
        final_preds.append(final_cls)

    print(f"\nTotal imágenes con desacuerdo: {conflicts} de {len(ref_ids)}")
    print(f"Porcentaje de incertidumbre: {conflicts/len(ref_ids)*100:.2f}%")

    sub = dfs['model_448'].copy()
    sub['cls'] = final_preds
    sub.to_csv(os.path.join(cfg.output_dir, "submission_VOTING.csv"), index=False)
    print("Submission VOTING guardada.")

ensemble_analysis_and_voting()