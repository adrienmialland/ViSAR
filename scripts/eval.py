
import re
import sys
import ast
import json
import click
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path 
from tabulate import tabulate
import matplotlib.pylab as plt
from collections import Counter
from dataclasses import dataclass
from matplotlib.ticker import FormatStrFormatter

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data import Dataset
from config import load_configs

from pipeline.generate import smart_resize_images
from pipeline.retrievers import ViSARSelector

from models.generators.qwen import QwenBaseModel
from models.generators.qwen import Qwen2_5_7B_Instruct, Qwen2_5_14B_Instruct

eval_system_prompt = """
You are a strict evaluation agent. Your task is to decide whether a model's answer 
is semantically correct with respect to the question and the set of valid reference answers.

You MUST output ONLY a single JSON object exactly following this schema:

{
  "correctness": "CORRECT" or "INCORRECT"
}

=== FEW-SHOT EXAMPLES ===

Example 1:
Question: What is the capital of France?
Reference Answers: ["Paris"]
Model Answer: Paris is the capital.
Output:
{"correctness": "CORRECT"}

Example 2:
Question: What is the total revenue in 2022?
Reference Answers: ["1.2 billion USD", "1.2B"]
Model Answer: The company earned about 900M that year.
Output:
{"correctness": "INCORRECT"}

=== END OF EXAMPLES ===

RULES:
1. Do NOT output explanations, reasoning, or any additional text.
2. Mark an answer as CORRECT if it is semantically equivalent to ANY of the reference answers.
3. Mark an answer as INCORRECT if the answer contradicts the reference, includes hallucinated information, or misses essential meaning.
4. Allow differences in phrasing, word order, synonyms, or formatting.
5. Partial answers that miss crucial information are INCORRECT.
6. Output must be a valid JSON object ONLY.
"""

eval_user_prompt = """
Evaluate the model's answer.

Question:
{question}

Reference Answer:
{ref_answer}

Model Answer:
{gen_answer}

Return the evaluation as JSON according to the schema.
"""

def get_eval_generator(name: str):
    if name == "Qwen2.5-7B-Instruct":
        return Qwen2_5_7B_Instruct()
    elif name == "Qwen2.5-14B-Instruct":
        return Qwen2_5_14B_Instruct()
    else:
        raise ValueError('unknown generation model')

def compute_ranking_metrics(ev_pages: list, indexes: list):
        pages_hit = [p in ev_pages for p in indexes]

        if any(pages_hit):
            dcg = lambda hits: sum(int(rel) / np.log2(i + 2) for i, rel in enumerate(hits))
            ideal_hit = sorted(pages_hit, reverse=True)
            ndcg = dcg(pages_hit) / dcg(ideal_hit)
            r_rank = 1 / (pages_hit.index(True) + 1)
            recall = sum(pages_hit) / len(ev_pages)
            precision = sum(pages_hit) / len(indexes)
            f1 = 2 * recall * precision / (recall + precision)
            hits = 1
        else:
            ndcg, r_rank, recall, precision, f1, hits = 0, 0, 0, 0, 0, 0

        return {
            'NDCG': ndcg, 'Recall': recall, 'Precision': precision, 'F1-score': f1, 'MRR': r_rank, 'Hits': hits
        }

def get_McNemar_p_value(late, visar):
    from statsmodels.stats.contingency_tables import mcnemar

    late = np.array(late)
    visar = np.array(visar)

    a = np.sum((late == True)  & (visar == True))
    b = np.sum((late == True)  & (visar == False))
    c = np.sum((late == False) & (visar == True))
    d = np.sum((late == False) & (visar == False))

    table = [[a, b], [c, d]]
    result = mcnemar(table, exact=True)

    return result.pvalue

def bootstrap_ci(labels, n_boot=10000, ci=95, seed=0):
    """
    labels: array-like of 0/1 per-query correctness (same input you already average)
    Returns: (lower_bound, upper_bound)
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    n = len(labels)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = labels[idx].mean(axis=1)
    alpha = (100 - ci) / 2
    lower, upper = np.percentile(boot_means, [alpha, 100 - alpha])
    se = boot_means.std(ddof=1)
    half_width = (upper - lower) / 2
    return se, half_width, lower, upper

@dataclass
class EvalState():
    mode: str
    encoder: str
    gen_id: str

class DatasetEval(Dataset):
    def __init__(self, dataset):
        super().__init__(dataset, True)

        self.answer_keys = ["answer", "answer_2", "answer_3"]

    def get_evidence_pages(self, sample: dict[str, dict]):
        if self.name == 'MMLongBench':
            indexes = sample['LateInteraction']['colpali-v1.2']['indexes']
            ev_pags = ast.literal_eval(sample['evidence_pages'])
            ev_pags = [max([0, p-1]) for p in ev_pags if max([0, p-1]) in indexes]
        else:
            ev_pags = sample['evidence_pages']

        return ev_pags

    def get_ref_answers(self, sample: dict):
        keys = [k for k in self.answer_keys if k in sample]
        ref_answer = [sample[k] for k in keys if sample[k] not in ["", " "]]
        return [x for item in ref_answer for x in (item if isinstance(item, list) else [item])]

class SensitivityAnalyser():
    def __init__(self, dataset: DatasetEval):
        self.dataset = dataset
        self.data = self.dataset.load_data(from_results=True)

    def run_sensitivity_analyses(self, encoder_names: list[str]):
        from models.encoders import get_encoder

        ViSAR = ViSARSelector()

        top_Ts = [1, 5, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
        gammas = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]

        not_in_top_T = lambda T: str(T) not in sample['ViSAR'][encoder.name]['SensitivityAnalysis']['top_T']
        not_in_gamma = lambda g: str(g) not in sample['ViSAR'][encoder.name]['SensitivityAnalysis']['gamma']

        print('\n### Evaluation of ViSAR sensitivity to T (Equaion 11) and gamma (Equation 14) :\n')
        for encoder_name in encoder_names:
            encoder = get_encoder(encoder_name)

            for sample in tqdm(self.data, desc='sensitivity '):
                if 'SensitivityAnalysis' not in sample['ViSAR'][encoder.name]:
                    sample['ViSAR'][encoder.name]['SensitivityAnalysis'] = {
                        'top_T': {}, 'gamma': {}
                    }

                Ts = [T for T in top_Ts if not_in_top_T(T)]
                gs = [g for g in gammas if not_in_gamma(g)]

                if Ts == [] and gs == []:
                    continue
                encoder.from_pretrained()            
                
                embed_query = encoder.encode_query(sample['question'])
                embed_pages = encoder.load_embed_pages(sample['doc_id'], self.dataset.embedding_dir)

                for T in Ts:
                    ranking = ViSAR.score_pages_and_find_k_star(
                        embed_query[0], embed_pages, top_T=T
                    )

                    ev_pages = self.dataset.get_evidence_pages(sample)
                    metrics_at_5  = compute_ranking_metrics(ev_pages, ranking['indexes'][:5 ])
                    metrics_at_10 = compute_ranking_metrics(ev_pages, ranking['indexes'][:10])

                    sample['ViSAR'][encoder.name]['SensitivityAnalysis']['top_T'][str(T)] = [
                        metrics_at_5['NDCG'], metrics_at_10['NDCG'], metrics_at_5['Recall'], metrics_at_10['Recall']
                    ]

                for g in gs:
                    ranking = ViSAR.score_pages_and_find_k_star(
                        embed_query[0], embed_pages, gamma=g
                    )

                    sample['ViSAR'][encoder.name]['SensitivityAnalysis']['gamma'][str(g)] = ranking['k_star']

                self.dataset.new_data = True
                self.dataset.save_results(self.data, 60)
            encoder.empty_memory()
        self.dataset.save_results(self.data)

    def results_sensitivity_analysis_to_T(self, encoder_names: list[str]):
        colors_dist_k = {
            'colpali-v1.2': 'xkcd:bright red',
            'colqwen2.5-v0.1': 'xkcd:kelly green',
            'colmodernvbert': 'xkcd:electric blue'
        }

        print('\n### Sensitivity analysis to T (Equation 11) :')
        for encoder_name in encoder_names:
            ndcg_at_5_visar, ndcg_at_10_visar = [], []
            recall_at_5_visar, recall_at_10_visar = [], []
            ndcg_at_5_lateinteraction, ndcg_at_10_lateinteraction = [], []
            recall_at_5_lateinteraction, recall_at_10_lateinteraction = [], []
            
            for sample in self.data:
                if sample['ViSAR'][encoder_name]['indexes'] == []:
                    continue

                ev_pages = self.dataset.get_evidence_pages(sample)
                if len(ev_pages) == 0:
                    continue

                metrics_visar = list(sample['ViSAR'][encoder_name]['SensitivityAnalysis']['top_T'].values())
                ndcg_at_5_visar.append([v[0] for v in metrics_visar])
                ndcg_at_10_visar.append([v[1] for v in metrics_visar])
                recall_at_5_visar.append([v[2] for v in metrics_visar])
                recall_at_10_visar.append([v[3] for v in metrics_visar])

                LateI_indexes = sample['LateInteraction'][encoder_name]['indexes']

                metrics_at_5_lateinteraction = [
                    compute_ranking_metrics(ev_pages, LateI_indexes[:5])[k] for k in ['NDCG', 'Recall']
                ]
                ndcg_at_5_lateinteraction.append(metrics_at_5_lateinteraction[0])
                recall_at_5_lateinteraction.append(metrics_at_5_lateinteraction[1])

                metrics_at_10_lateinteraction = [
                    compute_ranking_metrics(ev_pages, LateI_indexes[:10])[k] for k in ['NDCG', 'Recall']
                ]
                ndcg_at_10_lateinteraction.append(metrics_at_10_lateinteraction[0])
                recall_at_10_lateinteraction.append(metrics_at_10_lateinteraction[1])
                
            mean_ndcg_at_5_visar = np.mean(ndcg_at_5_visar, axis=0) - np.mean(ndcg_at_5_lateinteraction)
            mean_ndcg_at_10_visar = np.mean(ndcg_at_10_visar, axis=0) - np.mean(ndcg_at_10_lateinteraction)

            mean_recall_at_5_visar = 100 * ((np.mean(recall_at_5_visar, axis=0) - np.mean(recall_at_5_lateinteraction)))
            mean_recall_at_10_visar = 100 * ((np.mean(recall_at_10_visar, axis=0) - np.mean(recall_at_10_lateinteraction)))

            ndcg_at_5_ymax    = (-0.001, 0.012) if self.dataset.name == 'MMLongBench' else (-0.001, 0.018)
            ndcg_at_10_ymax   = (-0.001, 0.015) if self.dataset.name == 'MMLongBench' else (-0.001, 0.015)
            recall_at_5_ymax  = (-0.1  , 1.9  ) if self.dataset.name == 'MMLongBench' else (-0.1  , 1.9  )
            recall_at_10_ymax = (-0.1  , 2.50 ) if self.dataset.name == 'MMLongBench' else (-0.1  , 1.55 )

            top_Ts = [int(k) for k in sample['ViSAR'][encoder_name]['SensitivityAnalysis']['top_T'].keys()]

            for metric, y_label, y_lim, fig_name in [
                [mean_ndcg_at_5_visar   , 'Gains in NDCG@5'   , ndcg_at_5_ymax   , 'sensitivity_top_T_ndcg_at_5'   ], 
                [mean_ndcg_at_10_visar  , 'Gains in NDCG@10'  , ndcg_at_10_ymax  , 'sensitivity_top_T_ndcg_at_10'  ], 
                [mean_recall_at_5_visar , 'Gains in Recall@5' , recall_at_5_ymax , 'sensitivity_top_T_recall_at_5' ], 
                [mean_recall_at_10_visar, 'Gains in Recall@10', recall_at_10_ymax, 'sensitivity_top_T_recall_at_10']
            ]:
                plt.plot(metric, marker='.', label=encoder_name, color=colors_dist_k[encoder_name], linestyle='solid', linewidth=0.75)
                plt.axhline(0, color='seagreen', linestyle='dashed', label='Late-Interaction')
                plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
                plt.xlabel("top-$T$", fontsize=12); plt.ylabel(y_label, fontsize=12)
                plt.xticks(ticks=list(range(len(metric))), labels=top_Ts)
                plt.ylim(*y_lim)
                plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                plt.legend(fontsize=9); plt.tight_layout()
                to_save = f"{self.dataset.results_dir}/{encoder_name}/{fig_name}.svg"
                plt.savefig(to_save, dpi=300)
                print(f'- figure saved: {to_save}')
                plt.close()

    def results_sensitivity_analysis_to_gamma(self, encoder_names: list[str]):
        colors_dist_k = {
            'colpali-v1.2': 'xkcd:bright red',
            'colqwen2.5-v0.1': 'xkcd:kelly green',
            'colmodernvbert': 'xkcd:electric blue'
        }

        gammas = [str(k) for k in self.data[0]['ViSAR'][encoder_names[0]]['SensitivityAnalysis']['gamma'].keys()]

        print('\n### Sensitivity analysis to gamma (Equation 14) :')
        for encoder_name in encoder_names:
            k_stars = []
            for sample in self.data:
                k_stars.append(
                    [v for v in sample['ViSAR'][encoder_name]['SensitivityAnalysis']['gamma'].values()]
                )

            mean_k  = np.mean(k_stars, axis=0)
            median  = np.median(k_stars, axis=0)
            perc_25 = np.percentile(k_stars, 25, axis=0)
            perc_75 = np.percentile(k_stars, 75, axis=0)
            delta_k = np.mean(np.abs(np.diff(k_stars, axis=1)), axis=0)

            plt.plot(gammas, mean_k, marker='o', label=encoder_name, color=colors_dist_k[encoder_name])
            plt.plot(gammas, median, marker='.', label='median', color='xkcd:crimson', linestyle='dashed', linewidth=0.75)
            plt.plot(gammas, perc_25, marker='.', label='Q1', color='xkcd:grey blue', linestyle='dotted', linewidth=0.75)
            plt.plot(gammas, perc_75, marker='.', label='Q3', color='xkcd:grey blue', linestyle='dotted', linewidth=0.75)
            
            plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            plt.xlabel("$\\gamma$", fontsize=12); plt.ylabel("Mean $k^*$", fontsize=12)
            plt.legend(fontsize=9, loc='upper right'); plt.tight_layout()
            to_save = f"{self.dataset.results_dir}/{encoder_name}/sensitivity_gamma_mean_k_star.svg"
            plt.savefig(to_save, dpi=300)
            print(f'- figure saved: {to_save}')            
            plt.close()

            plt.plot(delta_k, marker='o', label=encoder_name, color=colors_dist_k[encoder_name])
            plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            plt.xlabel("$\\gamma$", fontsize=12); plt.ylabel("Mean $|\\Delta k^*|$", fontsize=12)
            plt.legend(fontsize=9, loc='upper right'); plt.tight_layout()
            to_save = f"{self.dataset.results_dir}/{encoder_name}/sensitivity_gamma_mean_delta_k_star.svg"
            plt.savefig(to_save, dpi=300)
            print(f'- figure saved: {to_save}')
            plt.close()

class ViSARSelectorAblation(ViSARSelector):
    def __init__(self, dataset: DatasetEval):
        super().__init__()

        self.dataset = dataset
        self.data = self.dataset.load_data(from_results=True)

        self.max_k = 10   

        self.disable_w_i = False
        self.disable_w_p = False

    def compute_weights(self, embed_query: torch.Tensor, embed_pages: torch.Tensor):
        n_pages = len(embed_pages)

        Sim: list[torch.Tensor] = []
        MaxSim = []
        
        # Equation 2
        # get semantic activation A_pi
        for page_emb in embed_pages:              # Equation 2, [np, D] (num v_pj, dim embed)
            Sim.append(embed_query @ page_emb.T)  # Equation 2, [m, np] (num q_i, num v_pj)
            MaxSim.append(Sim[-1].amax(dim=1))    # Equation 2, [m]     (num q_i,)
        A_pi = torch.stack(MaxSim)                # Equation 2, [N, m]  (num pages, num q_i)

        # Equation 3, 4, and
        # highlight strong and spatially localized semantics
        A_i_std = A_pi.std(dim=0)                          # Equation 3 
        sigma_i = A_i_std / A_i_std.mean()                 # Equation 3 
        A_pi_hat = (A_pi / A_pi.mean(dim=0, keepdim=True)) # Equation 3 
        A_pi_tilde = A_pi_hat * sigma_i                    # Equation 4
        # get token-level weights w_i
        a_pi = A_pi_tilde.sum(dim=0)             # Equation 5
        w_i = torch.log(n_pages / (1.0 + a_pi))  # Equation 5

        # Equation 6 and 7
        # Cmpute page-level semantic co-activation
        A_i_bar = A_pi_tilde.mean(dim=0)                                   # Equation 6
        C_pii = A_pi_tilde[:, :, None] * A_pi_tilde[:, None, :]            # Equation 6
        C_pii_hat = (C_pii / (A_i_bar[:, None] * A_i_bar[None, :] + 1e-6)) # Equation 6
        # get page-level weight w_p
        w_p = C_pii_hat.to(torch.bfloat16).mean(dim=1).sum(dim=1) # Equation 7

        # Equation 8
        # min-max normalization of token-level and page-level weights
        w_i_hat = (w_i - w_i.min()) / (w_i.max() - w_i.min() + 1e-6)
        w_p_hat = (w_p - w_p.min()) / (w_p.max() - w_p.min())
        
        if self.disable_w_i:
            w_i_hat = torch.ones_like(w_i_hat)
        if self.disable_w_p:
            w_p_hat = torch.ones_like(w_p_hat)

        A_weighted = A_pi_tilde * w_i_hat[None, :] * w_p_hat[:, None] # Equation 8 subcomponant: Ã_p,i weighted.
        # get patches-level embeddings relevance score r_pj
        r_pj: list[torch.Tensor] = []
        for sim, A_w in zip(Sim, A_weighted):
            r_pj.append(
                (sim * A_w[:, None].pow(2)).amax(dim=0) # Equation 8
            )

        # Equaion 9
        # get patches-level weights
        r_pj_flat = torch.cat(r_pj)  # flatten to compute a global mean
        r_pj_mean = r_pj_flat.mean() # mean patch relevant across pages
        w_pj = (r_pj_flat - r_pj_mean).clamp(min=0) # Equation 9

        n_v_jp = [len(r_j) for r_j in r_pj] # reshape relevances r_pj back to [N, np] (num pages, num v_pj)
        mm, mx = w_pj.min(), w_pj.max()

        # min-max normalization of patches-level weights
        w_pj_hat = (w_pj - mm) / (mx - mm + 1e-6)
        w_pj_hat = list(torch.split(w_pj_hat, n_v_jp))
        
        return w_pj_hat # return patches-level weights

    def run_retrieval_ablation(self, encoder_name: str):
        from models.encoders import get_encoder
        from pipeline.retrievers import HeuristicsSelector

        encoder = get_encoder(encoder_name)
        selector = HeuristicsSelector()

        ablations = [
            (True, False, None, 'w_i'), (False, True, None, 'w_p'), (True, True, None, 'w_i_w_p'), 
            (False, False, 'LargestGap', 'k_star_LargestGap'), (False, False, 'ScoreCluster', 'k_star_ScoreCluster')
        ]

        print('\n### Evluation of Retrieval ablation :\n')
        for sample in tqdm(self.data, desc='ret ablation'):
            if 'Ablation' not in sample:
                sample['Ablation'] = {}

            to_ablate = [a for a in ablations if a[3] not in sample['Ablation']]

            for disable_w_i, disable_w_p, heuristic, ablation_id in to_ablate:
                if ablation_id in ['w_i', 'w_p', 'w_i_w_p']:
                    encoder.from_pretrained()
                    
                    embed_query = encoder.encode_query(sample['question'])
                    embed_pages = encoder.load_embed_pages(sample['doc_id'], self.dataset.embedding_dir)

                    # Adjust the behaviour of self.score_pages_and_find_k_star
                    self.disable_w_i = disable_w_i
                    self.disable_w_p = disable_w_p

                    ranking = self.score_pages_and_find_k_star(
                        embed_query[0], embed_pages
                    )

                    ev_pages = self.dataset.get_evidence_pages(sample)
                    metrics_at_5  = compute_ranking_metrics(ev_pages, ranking['indexes'][:5 ])
                    metrics_at_10 = compute_ranking_metrics(ev_pages, ranking['indexes'][:10])

                    sample['Ablation'][ablation_id] = [
                        metrics_at_5['NDCG'], metrics_at_10['NDCG'], metrics_at_5['Recall'], metrics_at_10['Recall']
                    ]
                else:
                    scores = sample['ViSAR'][encoder_name]['scores']

                    if heuristic == 'LargestGap': 
                        k_star = selector.LargestGap(scores)
                    if heuristic == 'ScoreCluster': 
                        k_star = selector.ScoreCluster(scores)

                    sample['Ablation'][ablation_id] = k_star

                self.dataset.new_data = True
            self.dataset.save_results(self.data, 60)
        
        self.dataset.save_results(self.data)
        encoder.empty_memory()

    def run_generation_from_ablated_retrieval(self, encoder_name: str, LVLM_name: str):
        from models.generators import get_generator

        model = get_generator(LVLM_name)

        ablations = [
            (None, f'answer_top_{self.max_k}'),
            ('k_star_LargestGap', 'answer_LargestGap'), 
            ('k_star_ScoreCluster', 'answer_ScoreCluster')
        ]

        print('\n### Evluation of Generation ablation :\n')
        for sample in tqdm(self.data, desc='gen ablation'):

            to_ablate = [a for a in ablations if a[1] not in sample['Ablation']]

            for k_star_id, ablate_id in to_ablate:
                model.from_pretrained()

                k_star = sample['Ablation'].get(k_star_id, None)
                indexs = sample['ViSAR'][encoder_name]['indexes'][:k_star]
                if k_star == 0:
                    continue

                images = self.dataset.load_sample_images(sample, indexs[:self.max_k])
                images = smart_resize_images(images)

                answer = model.generate(sample['question'], images)

                sample['Ablation'][ablate_id] = {'answer': answer}

                self.dataset.new_data = True
            self.dataset.save_results(self.data, 60)

        self.dataset.save_results(self.data)

    def results_ablation_study(self, encoder_name: str, LVLM_name: str):
        metrics_no_abla = []
        metrics_w_i = []
        metrics_w_p = []
        metrics_w_i_w_p = []
        k_star_LargestGap = []
        k_star_ScoreCluster = []
        acc_noablation = []
        acc_fixedtop10 = []
        acc_LargestGap = []
        acc_ScoreCluster = []

        gen_id = LVLM_name + f'__max_{self.max_k}'

        data = self.dataset.load_data(from_results=True)

        for sample in data:
            ev_pages = self.dataset.get_evidence_pages(sample)
            
            ViSAR_indexes = sample['ViSAR'][encoder_name]['indexes']
            if len(ViSAR_indexes) == 0:
                continue

            k_star_LargestGap.append(sample['Ablation']['k_star_LargestGap'])
            k_star_ScoreCluster.append(sample['Ablation']['k_star_ScoreCluster'])

            acc_noablation.append(sample['ViSAR'][encoder_name][gen_id]['correctness'])
            acc_fixedtop10.append(sample['Ablation'][f'answer_top_{self.max_k}']['correctness'])
            acc_LargestGap.append(sample['Ablation']['answer_LargestGap']['correctness'])
            acc_ScoreCluster.append(sample['Ablation']['answer_ScoreCluster']['correctness'])

            if len(ev_pages) == 0:
                continue

            metrics_no_abla.append(sample['ViSAR'][encoder_name]['SensitivityAnalysis']['top_T']['50'])
            metrics_w_i.append(sample['Ablation']['w_i'])
            metrics_w_p.append(sample['Ablation']['w_p'])
            metrics_w_i_w_p.append(sample['Ablation']['w_i_w_p'])

        round_ = lambda ms: [round((1 if i < 2 else 100) * m, 3) for i, m in enumerate(ms)]

        tabular_ret_ablation = [
            ['ViSAR'       , *round_(np.mean(metrics_no_abla, axis=0))],
            ['w_i=1'       , *round_(np.mean(metrics_w_i, axis=0))],
            ['w_p=1'       , *round_(np.mean(metrics_w_p, axis=0))],
            ['w_i=1, w_p=1', *round_(np.mean(metrics_w_i_w_p, axis=0))]
        ]
        headers = ['Variant', 'NDCG@5', 'NDCG@10', 'Recall@5', 'Recall@10']
        print(tabulate(tabular_ret_ablation, headers=headers, tablefmt="github"))

        tabular_gen_ablation = [
            ['ViSAR'         , ''                                  , round(100 * np.mean(acc_noablation)  , 3)],
            ['Fixed top-k'   , ''                                  , round(100 * np.mean(acc_fixedtop10)  , 3)],
            ['Largest-Gap'   , round(np.mean(k_star_LargestGap), 2), round(100 * np.mean(acc_LargestGap)  , 3)],
            ['Score-Cluster ', round(np.mean(k_star_ScoreCluster), 2), round(100 * np.mean(acc_ScoreCluster), 3)]
        ]
        headers = ['variant', 'Mean k*', f'Accuracy with Max-{self.max_k} budget']
        print(tabulate(tabular_gen_ablation, headers=headers, tablefmt="github"))

class LatencyEvaluator():
    def __init__(self, dataset: DatasetEval):
        self.dataset = dataset
        self.data = self.dataset.load_data(from_results=True)

    def run_ViSAR_Approx_latencies(self, encoder_names: list[str]):
        from models.encoders import get_encoder

        approx_params = [(450, None, None), (None, 75, None), (None, None, 10), (450, 75, None), (450, None, 10), (None, 75, 10), (450, 75, 10)]
        get_id = lambda j, p, s: 'latency' + (f'_top_j_{j}' if j else '') + (f'_top_p_{p}' if p else '') + (f'_top_s_{s}' if s else '')

        ViSAR = ViSARSelector()        

        print('\n### Evaluation of ViSAR-Approx latency :')
        for encoder_name in encoder_names:
            encoder = get_encoder(encoder_name)

            print(f'\n# using {encoder_name}\n')
            for sample in tqdm(self.data, desc=f'Latency Approx '):
                if 'ViSAR_Approx' not in sample:
                    sample['ViSAR_Approx'] = {}
                if encoder.name not in sample['ViSAR_Approx']:
                    sample['ViSAR_Approx'][encoder.name] = {}

                params = [(j, p, s) for j, p, s in approx_params if get_id(j, p, s) not in sample['ViSAR_Approx'][encoder.name]]

                for j, p, s in params:
                    encoder.from_pretrained()
                    
                    embed_query = encoder.encode_query(sample['question'])
                    embed_pages = encoder.load_embed_pages(sample['doc_id'], self.dataset.embedding_dir)

                    ranking = ViSAR.score_pages_and_find_k_star(
                        embed_query[0], embed_pages, top_j=j, top_p=p, top_s=s
                    )

                    sample['ViSAR_Approx'][encoder.name][get_id(j, p, s)] = ranking['latency']

                    self.dataset.new_data = True
                self.dataset.save_results(self.data, 60)
            encoder.empty_memory()
        self.dataset.save_results(self.data)

    def results_latency_ViSAR_vs_lateinteraction(self, encoder_names: list[str], LVLM_name):
        full_approx = 'latency_top_j_450_top_p_75_top_s_10'
        for encoder_name in encoder_names:
            n_pages = []
            latency_retrieval = {
                "LateInteraction": [],
                "ViSAR": {"weights": [], "sim_mat": [], "ranking": [], "total": []},
                "ViSAR_Approx": {
                    'latency_top_j_450': {"weights": [], "sim_mat": [], "ranking": [], "total": []},
                    'latency_top_p_75': {"weights": [], "sim_mat": [], "ranking": [], "total": []},
                    'latency_top_s_10': {"weights": [], "sim_mat": [], "ranking": [], "total": []},
                    'latency_top_j_450_top_p_75': {"weights": [], "sim_mat": [], "ranking": [], "total": []},
                    'latency_top_j_450_top_s_10': {"weights": [], "sim_mat": [], "ranking": [], "total": []},
                    'latency_top_p_75_top_s_10': {"weights": [], "sim_mat": [], "ranking": [], "total": []},
                    'latency_top_j_450_top_p_75_top_s_10': {"weights": [], "sim_mat": [], "ranking": [], "total": []}
                }
            }
            latency_generation = {"LateInteraction": [], "ViSAR": [], "ViSAR_Approx": []}
            latency_end_to_end = {"LateInteraction": [], "ViSAR": [], "ViSAR_Approx": []}

            for sample in self.data:
                n_pages.append(len(sample['LateInteraction'][encoder_name]['indexes']))

                latency_retrieval['LateInteraction'].append(sample['LateInteraction'][encoder_name]['latency']['total'])
                for componant, latency in sample['ViSAR'][encoder_name]['latency'].items():
                    latency_retrieval['ViSAR'][componant].append(latency)
                for k, approx_value in sample['ViSAR_Approx'][encoder_name].items():
                    if k.startswith('latency'):                        
                        for componant, latency in approx_value.items():
                            latency_retrieval['ViSAR_Approx'][k][componant].append(latency)

                gen_ids_sorted = sorted(
                    [k for k in sample['ViSAR'][encoder_name].keys() if k.startswith(LVLM_name)],
                    key=lambda id: int(id.split('_')[-1])
                )

                if latency_generation['ViSAR'] == []:
                    for _ in range(len(gen_ids_sorted)):
                        latency_generation['ViSAR'].append([])
                        latency_end_to_end['ViSAR'].append([])
                        latency_generation['ViSAR_Approx'].append([])
                        latency_end_to_end['ViSAR_Approx'].append([])                        
                        latency_generation['LateInteraction'].append([])
                        latency_end_to_end['LateInteraction'].append([])

                for i, gen_id in enumerate(gen_ids_sorted):
                    latency_gen_visar = sample['ViSAR'][encoder_name][gen_id]['latency']
                    latency_gen_latei = sample['LateInteraction'][encoder_name][gen_id]['latency']
                    
                    latency_generation['ViSAR'][i].append(latency_gen_visar)
                    latency_end_to_end['ViSAR'][i].append(latency_retrieval['ViSAR']['total'][-1] + latency_gen_visar)

                    # assuming ViSAR-Approx parameters: top_j, top_p, and top_s are set so that no
                    # loss in retrieval performances is observed, then ViSAR-Approx uses ViSAR 
                    # generation latency for end-to-end, as input set of image are the same (see main paper).
                    latency_generation['ViSAR_Approx'][i].append(latency_gen_visar)
                    latency_end_to_end['ViSAR_Approx'][i].append(latency_retrieval['ViSAR_Approx'][full_approx]['total'][-1] + latency_gen_visar)

                    latency_generation['LateInteraction'][i].append(latency_gen_latei)
                    latency_end_to_end['LateInteraction'][i].append(latency_retrieval['LateInteraction'][-1] + latency_gen_latei)

            n_pages = np.array(n_pages)

            if self.dataset.name == 'MMLongBench':
                bins = [1, 51, 101, 151, 201, 500]
            else:
                bins = [50, 76, 101, 126, 151]

            print('\n### Retrieval latency: ViSAR vs Late-Interaction vs ViSAR-Approx.')
            for latency_to_plot in [['Late-Interaction', 'ViSAR'], ['Late-Interaction', 'ViSAR', 'ViSAR-Approx.']]:
                for latency, label, color in [
                    (latency_retrieval['LateInteraction']                    , 'Late-Interaction', 'seagreen'  ),
                    (latency_retrieval['ViSAR']['total' ]                    , 'ViSAR'           , 'darkorange'),
                    (latency_retrieval['ViSAR_Approx'][full_approx]['total' ], 'ViSAR-Approx.'   , 'crimson'   )
                ]:
                    if label not in latency_to_plot:
                        continue

                    means, counts, stds, xticks = [], [], [], []

                    for start, end in zip(bins[:-1], bins[1:]):
                        idx = [i for i, n in enumerate(n_pages) if start <= n < end]

                        if len(idx) == 0:
                            continue

                        values = [latency[i] for i in idx]

                        means.append(np.mean(values))
                        counts.append(len(idx))
                        stds.append(np.std(values))
                        xticks.append(f"{start}-{end-1}\n(n={len(idx)})")

                    plt.plot(xticks, means, marker=".", linewidth=1, label=label, color=color)
                    plt.fill_between(xticks, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], color=color, alpha=0.15)                
                
                plt.xticks(fontsize=7)
                plt.xlabel("# Document pages")
                plt.ylabel("Latency (s)")
                plt.title('Retrieval Latency')
                plt.grid(True, alpha=0.3)
                plt.legend()
                if 'ViSAR-Approx.' not in latency_to_plot:
                    to_save = f"{self.dataset.results_dir}/{encoder_name}/latency_retrieval.svg"
                else:
                    to_save = f"{self.dataset.results_dir}/{encoder_name}/latency_retrieval_ViSAR-Approx.svg"
                plt.savefig(to_save, dpi=300)
                print(f'- figure saved: {to_save}')
                plt.close()

            print('\n### Generation latency: ViSAR vs Late-Interaction vs ViSAR-Approx.')
            for latency_to_plot in [['Late-Interaction', 'ViSAR'], ['Late-Interaction', 'ViSAR', 'ViSAR-Approx.']]:
                for mode_latencies, label, color in [
                    (latency_generation['LateInteraction'], 'Late-Interaction' , 'seagreen'  ),
                    (latency_generation['ViSAR']          , 'ViSAR'            , 'darkorange'),
                    (latency_generation['ViSAR_Approx']   , 'ViSAR-Approx.'    , 'crimson'   )
                ]:
                    if label not in latency_to_plot:
                        continue

                    means, stds = [], []
                    for latencies in mode_latencies:
                        means.append(np.mean(latencies))
                        stds.append(np.std(latencies))
                    
                    plt.plot(list(range(1, len(means) + 1)), means, marker=".", linewidth=1, label=label, color=color)
                    plt.fill_between(list(range(1, len(means) + 1)), [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], alpha=0.15, color=color)

                plt.xticks(fontsize=7)
                plt.xlabel("Max_k retrieved pages")
                plt.ylabel("Latency (s)")
                plt.title('Generation Latency')
                plt.grid(True, alpha=0.3)
                plt.legend()
                if 'ViSAR-Approx.' not in latency_to_plot:            
                    to_save = f"{self.dataset.results_dir}/{encoder_name}/latency_generation.svg"
                else:
                    to_save = f"{self.dataset.results_dir}/{encoder_name}/latency_generation_ViSAR-Approx.svg"
                plt.savefig(to_save, dpi=300)
                print(f'- figure saved: {to_save}')        
                plt.close()

            if self.dataset.name == 'MMLongBench':
                document_sizes = [(1, float('inf'), 'All'), (100, 150, '100-150'), (151, 200, '151-200'), (201, float('inf'), '468')]
            else:
                document_sizes = [(1, float('inf'), 'All'), (75, 100, '75-100'), (100, 125, '100-125'), (125, 150, '125-150')]

            print('\n### End-to-end latency: ViSAR vs Late-Interaction vs ViSAR-Approx.')
            for latency_to_plot in [['Late-Interaction', 'ViSAR'], ['Late-Interaction', 'ViSAR', 'ViSAR-Approx.']]:
                for mn, mx, s_name in document_sizes:
                    for mode_latencies, label, color in [
                        (latency_end_to_end['LateInteraction'], 'Late-Interaction' , 'seagreen'  ),
                        (latency_end_to_end['ViSAR']          , 'ViSAR'            , 'darkorange'),
                        (latency_end_to_end['ViSAR_Approx']   , 'ViSAR-Approx.'    , 'crimson')
                    ]:
                        if label not in latency_to_plot:
                            continue

                        mask = (n_pages > mn) & (n_pages <= mx)
                        mode_latencies = [np.array(latencies)[mask] for latencies in mode_latencies]

                        means, stds = [], []
                        for latencies in mode_latencies:
                            means.append(np.mean(latencies))
                            stds.append(np.std(latencies))
                        
                        plt.plot(list(range(1, len(means) + 1)), means, marker=".", linewidth=1, label=label, color=color)
                        plt.fill_between(list(range(1, len(means) + 1)), [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], alpha=0.15, color=color)

                    plt.xticks(fontsize=7)
                    plt.xlabel("Max_k retrieved pages")
                    plt.ylabel("Latency (s)")
                    plt.title(f'End-to-end Latency - Queries on {s_name} pages docs')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    if 'ViSAR-Approx.' not in latency_to_plot:         
                        to_save = f"{self.dataset.results_dir}/{encoder_name}/latency_end-to-end_{s_name}_pages.svg"
                    else:
                        to_save = f"{self.dataset.results_dir}/{encoder_name}/latency_end-to-end_ViSAR-Approx_{s_name}_pages.svg"
                    plt.savefig(to_save, dpi=300)
                    print(f'- figure saved: {to_save}')        
                    plt.close()

            plt.figure(figsize=(10, 4))
            print('\n### Retrieval latency: ViSAR componants vs ViSAR-Approx componants.')
            for latencies, componant, color, plot_idx in [
                [latency_retrieval['ViSAR']['weights']  , 'weights', 'darkgray'  , 1],
                [latency_retrieval['ViSAR']['sim_mat']  , 'sim_mat', 'darkorange', 1],
                [latency_retrieval['ViSAR']['ranking']  , 'ranking', 'orchid'    , 1],
                [latency_retrieval['ViSAR']['total'  ]  , 'total'  , 'crimson'   , 1],
                [latency_retrieval['ViSAR_Approx'][full_approx]['weights'], 'weights', 'darkgray'  , 2],
                [latency_retrieval['ViSAR_Approx'][full_approx]['sim_mat'], 'sim_mat', 'darkorange', 2],
                [latency_retrieval['ViSAR_Approx'][full_approx]['ranking'], 'ranking', 'orchid'    , 2],
                [latency_retrieval['ViSAR_Approx'][full_approx]['total'  ], 'total'  , 'crimson'   , 2]
            ]:
                plt.subplot(1, 2, plot_idx)
                means, counts, stds, xticks = [], [], [], []

                for start, end in zip(bins[:-1], bins[1:]):
                    idx = [i for i, n in enumerate(n_pages) if start <= n < end]

                    if len(idx) == 0:
                        continue

                    values = [latencies[i] for i in idx]

                    means.append(np.mean(values))
                    counts.append(len(idx))
                    stds.append(np.std(values))
                    xticks.append(f"{start}-{end-1}")
                
                plt.plot(xticks, means, marker=".", linewidth=1, label=componant, color=color, alpha=1)
                plt.fill_between(xticks, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], color=color, alpha=0.1)
        
                plt.xticks(fontsize=7)
                plt.xlabel("# Document pages")
                plt.ylabel("Latency (s)")
                plt.grid(True, alpha=0.3)
                plt.legend()

            to_save = f"{self.dataset.results_dir}/{encoder_name}/latency_retrieval_by_componants_ViSAR_vs_ViSAR-Approx.svg"
            plt.savefig(to_save, dpi=300)
            print(f'- figure saved: {to_save}')    
            plt.close()

            print('\n### Retrieval latency: ViSAR-Approx strategies.')
            for latencies, label, color, linestyle, zorder in [
                (latency_retrieval['LateInteraction']                                             , 'Late-Interaction' , 'seagreen'     , 'solid',  3),
                (latency_retrieval['ViSAR']['total']                                              , 'ViSAR        '    , 'darkorange'   , 'solid',  6),
                (latency_retrieval['ViSAR_Approx']['latency_top_p_75']['total']                   , 'max_p'            , 'steelblue'    , 'dashed', 0),
                (latency_retrieval['ViSAR_Approx']['latency_top_s_10']['total']                   , 'max_k'            , 'cadetblue'    , 'dashed', 0),
                (latency_retrieval['ViSAR_Approx']['latency_top_j_450']['total']                  , 'top_j'            , 'lightskyblue' , 'dashed', 0),
                (latency_retrieval['ViSAR_Approx']['latency_top_p_75_top_s_10']['total']          , 'max_p, max_k'     , 'orange'       , 'dashed', 0),
                (latency_retrieval['ViSAR_Approx']['latency_top_j_450_top_p_75']['total']         , 'top_j, max_p'     , 'darkgray'     , 'dashed', 0),
                (latency_retrieval['ViSAR_Approx']['latency_top_j_450_top_s_10']['total']         , 'top_j, max_k'     , 'orchid'       , 'dashed', 0),
                (latency_retrieval['ViSAR_Approx']['latency_top_j_450_top_p_75_top_s_10']['total'], 'ViSAR-Approx.'    , 'crimson'      , 'solid', 10)
            ]:
                means, counts, stds, xticks = [], [], [], []

                for start, end in zip(bins[:-1], bins[1:]):
                    idx = [i for i, n in enumerate(n_pages) if start <= n < end]

                    if len(idx) == 0:
                        continue

                    values = [latencies[i] for i in idx]

                    means.append(np.mean(values))
                    counts.append(len(idx))
                    stds.append(np.std(values))
                    xticks.append(f"{start}-{end-1}\n(n={len(idx)})")

                    linewidth = 0.6 if linestyle == 'dashed' else 1
                    alpha = 0.8 if linestyle == 'dashed' else 1
                
                plt.plot(xticks, means, marker=".", linewidth=linewidth, label=label, color=color, linestyle=linestyle, alpha=alpha, zorder=zorder)
                # plt.fill_between(xticks, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], color=color, alpha=0.1)
            
            plt.xticks(fontsize=7)
            plt.xlabel("Number of pages")
            plt.ylabel("Latency (s)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            to_save = f"{self.dataset.results_dir}/{encoder_name}/latency_retrieval_by_ViSAR-Approx_strategies.svg"
            plt.savefig(to_save, dpi=300)
            print(f'- figure saved: {to_save}')
            plt.close()

    def results_latency_ViSAR_vs_heuristics(self, encoder_names: list[str], max_ks: list[int], LVLM_name: str):
        manuscript_max_ks = [5, 10]
        if manuscript_max_ks[0] in max_ks and manuscript_max_ks[1] in max_ks:
            max_ks_ = manuscript_max_ks
        else:
            max_ks_ = max_ks

        print('\n### Retrieval, Generation, and End-to-end Latencies: ViSAR vs Heristics :')
        for encoder_name in encoder_names:
            mean_ret_latency = []
            mean_gen_latency = []
            mean_ete_latency = []
            
            for mode in ['LateInteraction', 'LargestGap', 'ScoreCluster', 'ViSAR']:
                mean_ret_latency.append([mode])
                mean_gen_latency.append([mode])
                mean_ete_latency.append([mode])

                ret_latency = [sample[mode][encoder_name]['latency']['total'] for sample in self.data]
                mean_ret_latency[-1].append(round(np.mean(ret_latency), 3))

                for k in max_ks_:
                    gen_latency = [sample[mode][encoder_name][f'{LVLM_name}__max_{k}']['latency'] for sample in self.data]
                    mean_gen_latency[-1].append(round(np.mean(gen_latency), 3))
                    mean_ete_latency[-1].append(round(mean_ret_latency[-1][-1] + mean_gen_latency[-1][-1], 3))

            print(f'# Using {encoder_name} and {LVLM_name}')

            headers = ['modes', 'Latency']
            print(tabulate(mean_ret_latency, headers=headers, tablefmt="github"))

            mean_gen_latency = [[''] + l for l in mean_gen_latency]
            mean_gen_latency[0][0] = 'Generation'
            mean_ete_latency = [[''] + l for l in mean_ete_latency]
            mean_ete_latency[0][0] = 'End_to_end'

            mean_latencies = mean_gen_latency + [[]] + mean_ete_latency

            headers = ['modes'] + [f'Latency at Max-{k}' for k in max_ks_]
            print(tabulate(mean_latencies, headers=headers, tablefmt="github"), '\n')

class AdaptiveRetrievalEvaluator():
    def __init__(self, dataset: DatasetEval):
        self.dataset = dataset
        self.data = self.dataset.load_data(from_results=True)

    def results_k_star_distributions(self, encoder_names: list[str]):          
        zorder = {
            'ev_pages': 0,
            'k_oracle': 20,
            'LargestGap': 40,
            'ScoreCluster': 60,
            'ViSAR': 80
        }
        colors = {
            'k_oracle': 'black',
            'ev_pages': 'xkcd:cool grey',
            'colpali-v1.2': 'xkcd:bright red',
            'colqwen2.5-v0.1': 'xkcd:kelly green',
            'colmodernvbert': 'xkcd:electric blue',
            'LargestGap': 'xkcd:vivid purple',
            'ScoreCluster': 'xkcd:golden rod'
        }
        linestyle = {
            'LargestGap': 'dashed',
            'ScoreCluster': 'dashed',
            'ev_pages': 'solid',
            'k_oracle': 'dashed',
            'ViSAR': 'solid'
        }

        max_evidence_pages = {
            'MMLongBench': 24,
            'LongDocURL': 30
        }
        max_ev_pages = max_evidence_pages[self.dataset.name]

        get_plot_args = lambda mode, encoder_name: {
            'order': zorder[mode],
            'line' : linestyle[mode],
            'color': colors[encoder_name if mode == 'ViSAR' else mode],
            'label': mode + (f' {encoder_name}' if encoder_name else '')
        }

        def plot_k_stars_distribution(k_stars, order: str, line: str, color: str, label: str):
            counts = Counter(k_stars)
            if 0 in counts:
                c_zero = counts.pop(0)
            x = list(counts.keys())
            y = list(counts.values())

            a = np.argsort(x)
            x = np.array(x)[a]
            y = np.array(y)[a] / sum(y)

            mask = x <= max_ev_pages
            x, y = x[mask], y[mask]
            plt.plot(
                x, 100 * y, 
                marker='.', label=label,
                zorder=order, color=color, linestyle=line
            )
        
        print(f'\n### K* Distributions - ViSAR vs Heuristics :')
        for encoder_name in encoder_names:
            k_star_per_mode = {
                'k_oracle': [], 'ev_pages': [], 'ViSAR': [], 'LargestGap': [], 'ScoreCluster': []
            }

            for sample in self.data:
                ev_pages = self.dataset.get_evidence_pages(sample)
                pages_idxs = sample['LateInteraction']['colpali-v1.2']['indexes']
                ev_pages_idxs = [i for i, p in enumerate(pages_idxs) if p in ev_pages]
                k_star_per_mode['k_oracle'].append((ev_pages_idxs[-1] + 1) if ev_pages_idxs else 0)
                k_star_per_mode['ev_pages'].append(len(ev_pages))

                if sample['ViSAR'][encoder_name]['indexes'] == []:
                    continue

                k_star_per_mode['ViSAR'].append(sample['ViSAR'][encoder_name]['k_star'])
                k_star_per_mode['LargestGap'].append(sample['LargestGap'][encoder_name]['k_star'])
                k_star_per_mode['ScoreCluster'].append(sample['ScoreCluster'][encoder_name]['k_star'])

            plt.figure(0)
            for mode in ['k_oracle', 'ev_pages', 'ViSAR']:
                if encoder_name != encoder_names[0] and mode != 'ViSAR':
                    continue
                plot_k_stars_distribution(k_star_per_mode[mode], **get_plot_args(mode, encoder_name))

            plt.figure(1)
            for mode in ['ev_pages', 'ViSAR', 'LargestGap', 'ScoreCluster']:
                plot_k_stars_distribution(k_star_per_mode[mode], **get_plot_args(mode, encoder_name))

            plt.figure(1)
            plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            plt.xlabel("# Retrieved Pages", fontsize=12); plt.ylabel("Queries (%)", fontsize=12)
            plt.title("ViSAR's Distribution of k*", fontsize=14, fontweight="bold")
            plt.xticks(list(range(1, max_ev_pages)), fontsize=10); plt.yticks(fontsize=10)
            plt.legend(fontsize=9, loc='upper right'); plt.tight_layout()
            to_save = f"{self.dataset.results_dir}/{encoder_name}/k_star_distributions_ViSAR_vs_heuristics.svg"
            plt.savefig(to_save, dpi=300)
            plt.close()
            print(f'- figure saved: {to_save}')

        plt.figure(0)
        print(f'\n### K* Distributions - ViSAR vs Oracle:')
        plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.xlabel("# Retrieved Pages", fontsize=12); plt.ylabel("Queries (%)", fontsize=12)
        plt.title("ViSAR's Distribution of k*", fontsize=14, fontweight="bold")
        plt.xticks(list(range(1, max_ev_pages)), fontsize=10); plt.yticks(fontsize=10)
        plt.legend(fontsize=9, loc='upper right'); plt.tight_layout()
        to_save = f"{self.dataset.results_dir}/k_star_distributions_ViSAR.svg"
        plt.savefig(to_save, dpi=300)
        plt.close()
        print(f'- figure saved: {to_save}')

    def results_k_star_statistics_and_metrics(self, encoder_names: list[str]):
        tabular_metrics = []
        round_ = lambda ms: [round(100 * m if i in [0, 1, 2] else m, 2) for i, m in enumerate(ms)]

        print('\n### Retrieval statistics and metrics@k* :')
        for encoder_name in encoder_names:
            for mode in ['Oracle', 'ScoreCluster', 'LargestGap', 'ViSAR']:
                metrics = []
                k_stars = []

                for sample in self.data:
                    if sample['ViSAR'][encoder_name]['indexes'] == []:
                        continue

                    ev_pages = self.dataset.get_evidence_pages(sample)
                    if len(ev_pages) == 0:
                        continue

                    if mode == 'Oracle':
                        indexes = sample['LateInteraction'][encoder_name]['indexes']
                        ev_pags_idxs = [i for i, p in enumerate(indexes) if p in ev_pages]
                        k_star = ev_pags_idxs[-1] + 1
                        indexes = sample['LateInteraction'][encoder_name]['indexes']
                    else:
                        k_star  = sample[mode][encoder_name]['k_star']
                        indexes = sample[mode][encoder_name]['indexes']

                    k_stars.append(k_star)

                    m = compute_ranking_metrics(ev_pages, indexes[:k_star])
                    metrics.append([m['Recall'], m['Precision']])

                recall, precision = round_(np.mean(metrics, axis=0))
                F1 = round(2 * recall * precision / ( recall + precision), 2)

                mn_k_star = round(np.mean(k_stars), 1)
                md_k_star = np.median(k_stars)

                tabular_metrics.append([encoder_name, mode, mn_k_star, md_k_star, recall, precision, F1])
            tabular_metrics.append(['-' * 17, '-' * 15, '-' * 10, '-' * 10, '-' * 10, '-' * 10, '-' * 10])

        headers = ['encoder', 'method', 'mean', 'median', 'Recall@k*', 'Precision@k*', 'F1-score@k*']
        print(tabulate(tabular_metrics, headers=headers, tablefmt="github"))

    def results_metrics_as_a_function_of_k_oracle(self, encoder_names: list[str]):
        zorder_dist_k = {
            'ev_pages': 0,
            'k_oracle': 20,
            'LargestGap': 40,
            'ScoreCluster': 60,
            'ViSAR': 80
        }
        colors_dist_k = {
            'k_oracle': 'black',
            'ev_pages': 'xkcd:cool grey',
            'colpali-v1.2': 'xkcd:bright red',
            'colqwen2.5-v0.1': 'xkcd:kelly green',
            'colmodernvbert': 'xkcd:electric blue',
            'LargestGap': 'xkcd:vivid purple',
            'ScoreCluster': 'xkcd:golden rod'
        }

        print(f'\n### Retrieval metrics@k* as a function of k_Oracle :')
        for encoder_name in encoder_names:
            k_stars_per_mode = {'ViSAR': [], 'LargestGap': [], 'ScoreCluster': [], 'k_oracle': [], 'ev_pages': []}
            metrics_per_mode = {'ViSAR': [], 'LargestGap': [], 'ScoreCluster': [], 'k_oracle': []}

            for sample in self.data:
                if sample['ViSAR'][encoder_name]['indexes'] == []:
                    continue

                ev_pages = self.dataset.get_evidence_pages(sample)
                if len(ev_pages) == 0:
                    continue

                indexes = sample['LateInteraction'][encoder_name]['indexes']
                ev_pages_idxs = [i for i, p in enumerate(indexes) if p in ev_pages]
                k_stars_per_mode['k_oracle'].append((ev_pages_idxs[-1] + 1) if ev_pages_idxs else 0)
                k_stars_per_mode['ev_pages'].append(len(ev_pages))

                k_stars_per_mode['ViSAR'].append(sample['ViSAR'][encoder_name]['k_star'])
                k_stars_per_mode['LargestGap'].append(sample['LargestGap'][encoder_name]['k_star'])
                k_stars_per_mode['ScoreCluster'].append(sample['ScoreCluster'][encoder_name]['k_star'])

                ViSAR_indexes = sample['ViSAR'][encoder_name]['indexes']
                LateI_indexes = sample['LateInteraction'][encoder_name]['indexes']

                k_star = sample['ViSAR'][encoder_name]['k_star']
                metrics_per_mode['ViSAR'].append(
                    compute_ranking_metrics(ev_pages, ViSAR_indexes[:k_star])
                )

                k_star = sample['LargestGap'][encoder_name]['k_star']
                metrics_per_mode['LargestGap'].append(
                    compute_ranking_metrics(ev_pages, LateI_indexes[:k_star])
                )

                k_star = sample['ScoreCluster'][encoder_name]['k_star']
                metrics_per_mode['ScoreCluster'].append(
                    compute_ranking_metrics(ev_pages, LateI_indexes[:k_star])
                )

                k_star = k_stars_per_mode['k_oracle'][-1]
                metrics_per_mode['k_oracle'].append(
                    compute_ranking_metrics(ev_pages, LateI_indexes[:k_star])
                )
                
            k_oracle = np.array(k_stars_per_mode['k_oracle'])
            
            for metric in ['Precision', 'Recall', 'F1-score', 'k_diffs']:
                query_percentages = None

                for mode in ['ViSAR', 'LargestGap', 'ScoreCluster', 'k_oracle']:
                    if metric != 'k_diffs':
                        mode_data = np.array([mode_mets[metric] for mode_mets in metrics_per_mode[mode]])
                    else:
                        mode_k = np.array(k_stars_per_mode[mode])
                        mode_data = mode_k - k_oracle
                    
                    x, y, counts = [], [], []

                    xticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 21, 26, 36, 500]
                    for k1, k2 in zip(xticks[:-1], xticks[1:]):
                        mask = (k_oracle >= k1) & (k_oracle < k2)
                        
                        if sum(mask) == 0:
                            continue

                        x.append((k1 + min(k2, xticks[-2]) - 1) / 2)
                        y.append(np.mean(mode_data[mask]))
                        counts.append(np.sum(mask))

                    plt.plot(
                        x, y, marker=".", label=mode,
                        zorder=zorder_dist_k[mode], 
                        color=colors_dist_k[encoder_name if mode == 'ViSAR' else mode]
                    )
                    
                    if query_percentages is None:
                        query_percentages = plt.bar(
                            x, np.array(counts) / sum(counts),
                            color="lightgray", alpha=0.5,
                            edgecolor="none", zorder=-10,
                            label='Query distribution'
                        )
            
                xtick_pos = [
                    (k1 + min(k2, xticks[-2]) - 1) / 2 for k1, k2 in zip(xticks[:-1], xticks[1:])
                ]
                xtick_str = [
                    str(x1) if x1 <= 10 else f'{x1}+' if x1 == xticks[-2] else f"{x1}-{x2-1}" for x1, x2 in zip(xticks[:-1], xticks[1:])
                ]
                plt.xticks(xtick_pos, xtick_str, fontsize=10); plt.yticks(fontsize=10)
                plt.xlabel('$k_{oracle}$', fontsize=12)
                plt.legend(fontsize=9)
                
                if metric != 'k_diffs':
                    plt.ylim(0, 1.05)
                    plt.ylabel(metric, fontsize=12)  
                else:
                    plt.ylabel('$mean(k_{method} - k_{Oracle})$', fontsize=12)  
                plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

                if metric != 'k_diffs':
                    plt.twinx()
                    plt.ylim(0, 1.05)
                    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
                    plt.ylabel('% of Queries', fontsize=12) 
                
                plt.tight_layout()
                to_save = f"{self.dataset.results_dir}/{encoder_name}/Metric_vs_k_oracle_{metric}.svg"
                plt.savefig(to_save, dpi=300)
                print(f'- figure saved: {to_save}')
                plt.close()

    def result_ranking_metrics(self, encoder_names: list[str]):
        tabular_metrics = []
        METRICS = ['Recall', 'NDCG', 'Hits', 'MRR']
        round_ = lambda ms: [round(100 * m if i in [0, 2] else m, 3) for i, m in enumerate(ms)]

        print(f'\n### Ranking metrics at ranks 5 and 10 :')
        for rank in [5, 10]:
            rank_str = f'Metrics@{rank}'

            for encoder_name in encoder_names:
                ViSAR_metrics = []
                LateI_metrics = []

                for sample in self.data:
                    LateI_indexes = sample['LateInteraction'][encoder_name]['indexes']
                    ViSAR_indexes = sample['ViSAR'][encoder_name]['indexes']
                    if ViSAR_indexes == [] or LateI_indexes == []:
                        continue

                    ev_pages = self.dataset.get_evidence_pages(sample)
                    if len(ev_pages) == 0:
                        continue

                    LateI_metrics.append([
                        compute_ranking_metrics(ev_pages, LateI_indexes[:rank])[k] for k in METRICS
                    ])
                    ViSAR_metrics.append([
                        compute_ranking_metrics(ev_pages, ViSAR_indexes[:rank])[k] for k in METRICS
                    ])

                LateI_m = round_(np.mean(LateI_metrics, axis=0))
                ViSAR_m = round_(np.mean(ViSAR_metrics, axis=0))

                tabular_metrics.append([rank_str, encoder_name, 'LateInteraction', *LateI_m])
                tabular_metrics.append([''      , ''          , 'ViSAR'          , *ViSAR_m])
                rank_str = ''

        headers = ['Rank', 'encoder', 'mode', *METRICS]
        print(tabulate(tabular_metrics, headers=headers, tablefmt="github"))

class AccuracyEvaluator():
    def __init__(self, dataset: DatasetEval):
        # We evaluate the generated answers using LLM-as-a-judge evaluation, with 
        # Qwen2.5-14B-Instruct LLM as the evaluator (see main paper 'Experimental setup' section).
        self.model: QwenBaseModel = get_eval_generator('Qwen2.5-14B-Instruct')
        
        self.dataset = dataset
        self.data = self.dataset.load_data(from_results=True)

    def is_evaluated(self, sample: dict[str, dict], state: EvalState):
        return (
            state.gen_id in sample[state.mode][state.encoder] and
            'correctness' in sample[state.mode][state.encoder][state.gen_id]
        ) or sample[state.mode][state.encoder]['indexes'] == []

    def run_answer_evaluation(self, state: EvalState):
        self.model.system_prompt = eval_system_prompt

        for sample in tqdm(self.data, desc='evaluate ans'):
            if self.is_evaluated(sample, state):
                continue
            self.model.from_pretrained()

            ref_answer = self.dataset.get_ref_answers(sample)
            gen_answer = sample[state.mode][state.encoder][state.gen_id]['answer']

            self.model.user_prompt = eval_user_prompt.format(
                question="""{question}""", ref_answer=ref_answer, gen_answer=gen_answer
            )

            correctness: bool = self.safe_parse_output(
                self.model.generate(sample['question'], [], no_parse=True)
            )

            sample[state.mode][state.encoder][state.gen_id]['correctness'] = correctness

            self.dataset.new_data = True
            self.dataset.save_results(self.data, 60)
        self.dataset.save_results(self.data)

    def safe_parse_output(self, model_output: str) -> bool:
        model_output = re.sub(r"<.*?>", "", model_output).strip()

        try:
            data: dict[str, str] = json.loads(model_output)
            correctness = next((v for k, v in data.items() if k.lower() == "correctness"), "")
            correctness = str(correctness).strip().upper()
            if correctness in ["CORRECT", "INCORRECT"]:
                return correctness == "CORRECT"
        except json.JSONDecodeError:
            pass

        text = model_output.upper().replace("-", " ")
        if re.search(r"\bCORRECT\b", text) and not re.search(r"\bNOT CORRECT\b", text):
            return True
        elif re.search(r"\bINCORRECT\b", text):
            return False
        
        return False

    def result_answer_generation_accuracy(self, encoder_names: list[str], LVLM_name: str, max_ks: list[int]):
        tabular_acc = []
        tabular_p_values = []

        print('\n### Accuracy results - ViSAR vs Late-Interaction vs Heuristics :')
        for mode in ['LateInteraction', 'LargestGap', 'ScoreCluster']:
            tabular_acc += [
                [LVLM_name, mode   ], 
                [''       , 'ViSAR']
            ]
            tabular_p_values += [
                [LVLM_name, mode   ]
            ]
            for encoder_name in encoder_names:
                for k in max_ks:
                    gen_id = LVLM_name + f'__max_{k}'
                    Vi_are_correct, LI_are_correct = [], []

                    for sample in self.data:
                        # avoid corrupted samples from ColModernVBERT
                        base_indexes = sample[mode][encoder_name]['indexes']
                        ViSAR_indexes = sample['ViSAR'][encoder_name]['indexes']
                        if ViSAR_indexes == [] or base_indexes == []:
                            continue

                        LI_are_correct.append(sample[mode][encoder_name][gen_id]['correctness'])
                        Vi_are_correct.append(sample['ViSAR'][encoder_name][gen_id]['correctness'])

                    se, hw, lower, upper = [
                        round(100 * v, 2) for v in bootstrap_ci(LI_are_correct)
                    ]
                    tabular_acc[-2].append(
                        str(round(100 * np.mean(LI_are_correct), 2)) + f'  ({lower}-{upper})'
                    )

                    se, hw, lower, upper = [
                        round(100 * v, 2) for v in bootstrap_ci(Vi_are_correct)
                    ]
                    tabular_acc[-1].append(
                        str(round(100 * np.mean(Vi_are_correct), 2)) + f'  ({lower}-{upper})'
                    )

                    tabular_p_values[-1].append(round(get_McNemar_p_value(LI_are_correct, Vi_are_correct), 3))

        encoder_at_k = [e + f'@{k}' for e in encoder_names for k in max_ks]

        print('# Mean accuracies with 95% confidence intervals in parentheses :')
        headers = ['models', 'mode'] + encoder_at_k
        print(tabulate(tabular_acc, headers=headers, tablefmt="github"))
        print('# Mean accuracies difference p-values (McNemar\'s test) :')
        headers = ['models', 'mode'] + encoder_at_k
        print(tabulate(tabular_p_values, headers=headers, tablefmt="github"))

    def results_accuracy_vs_similarity_matrix_sparsity(self, encoder_names: list[str], max_ks: list[int], LVLM_name: str):
        colors = {
            'k_oracle': 'black',
            'ev_pages': 'xkcd:cool grey',
            'colpali-v1.2': 'xkcd:bright red',
            'colqwen2.5-v0.1': 'xkcd:kelly green',
            'colmodernvbert': 'xkcd:electric blue',
            'LargestGap': 'xkcd:vivid purple',
            'ScoreCluster': 'xkcd:golden rod'
        }
        labels = {
            'colpali-v1.2': 'Colpali',
            'colqwen2.5-v0.1': 'ColQwen',
            'colmodernvbert': 'ColModernVBERT'
        }

        round_ = lambda z: [
            round(100 * np.mean(z), 2), 
            round(100 * np.std(z), 2), 
            round(100 * min(z), 2), 
            round(100 * max(z), 2)
        ]

        manuscript_max_ks = [5, 10]
        if manuscript_max_ks[0] in max_ks and manuscript_max_ks[1] in max_ks:
            max_ks_ = manuscript_max_ks
        else:
            max_ks_ = max_ks
        
        headers = ['encoder', 'answer type', 'mean', 'std', 'min', 'max']
        tabular_data = {k: [] for k in max_ks_}

        print('\n### Accuracy vs similarity matrix sparsity :')
        for encoder_name in encoder_names:
            for k in max_ks_:
                sparsity = {'correct': [], 'wrong': [], 'all': []}
                correctness = []
            
                for sample in self.data:
                    scores = np.array(sample['ViSAR'][encoder_name]['scores'])
                    if len(scores) == 0:
                        continue

                    correctness.append(sample['ViSAR'][encoder_name][f'{LVLM_name}__max_{k}']['correctness'])

                    # 0.01: small trailing scores were observed in some cases, especially 
                    # with ColModernVBERT. While epsilon = 0.0 yields comparable results, 
                    # using 0.01 filters these residual similarities, to better capture the 
                    # relationship between similarity matrix sparsity and answer accuracy.
                    epsilon = 0.01
                    p = sum(scores <= epsilon) / len(scores)
                    if correctness[-1]:
                        sparsity['correct'].append(p)
                    else:
                        sparsity['wrong'].append(p)
                    sparsity['all'].append(p)

                tabular_data[k].append([encoder_name, 'correct', *round_(sparsity['correct'])])
                tabular_data[k].append([''          , 'wrong'  , *round_(sparsity['wrong'])])
                tabular_data[k].append([''          , 'all'    , *round_(sparsity['all'])])

                filter_data = np.array(sparsity['all'])
                correctness = np.array(correctness)

                x, y = [], []

                for p in range(0, 100, 5):
                    thr = np.percentile(filter_data, min(p, 100))
                    
                    mask = (filter_data >= thr)
                    if sum(mask) == 0:
                        continue

                    x.append(100*np.mean(filter_data[mask]))
                    y.append(np.mean(correctness[mask]))

                plt.plot(
                    x, y, label=labels[encoder_name] + f' (max-{k})',
                    color=colors[encoder_name],
                    linestyle='dashed' if k == 5 else 'solid'
                )

        plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.xlabel('Similarity Matrix Inactive Entries (%)', fontsize=12); plt.ylabel('Accuracy', fontsize=12)
        plt.legend(fontsize=9); plt.tight_layout()   
        to_save = f"{self.dataset.results_dir}/acc_vs_matrix_sparsity.svg"
        plt.savefig(to_save, dpi=300)
        print(f'- figure saved: {to_save}')

        print('\n### Similarity matrix sparsity (%) vs answer binary correctness:')
        for k in max_ks_:
            print(f'# Using a Max-{k} LVLM input dubget :')
            print(tabulate(tabular_data[k], headers=headers, tablefmt="github"))


USAGE_ERROR_ANALYSIS = "\n\nAt least one analysis should be selected: retrieval, generation, latency, sensitivity, ablation\n"

@click.command()
@click.option('--config', '-co', type=click.Path(exists=True, dir_okay=False), required=True, help='Path to the YAML configuration file.')
@click.option('--sweep', '-sw', is_flag=True, required=False, help='Activate parameter sweep mode.')
@click.option('--retrieval', '-re', is_flag=True, required=False, help='Run retrieval analysis.')
@click.option('--generation', '-ge', is_flag=True, required=False, help='Run generation analysis.')
@click.option('--latency', '-la', is_flag=True, required=False, help='Run latency analysis.')
@click.option('--sensitivity', '-sa', is_flag=True, required=False, help='Run sensitivity analysis.')
@click.option('--ablation', '-ab', is_flag=True, required=False, help='Run ablation study.')
def main(config, sweep, retrieval, generation, latency, sensitivity, ablation):
    if not any([retrieval, generation, latency, sensitivity, ablation]):
        print(click.get_current_context().get_help() + USAGE_ERROR_ANALYSIS)
        return

    config_set = load_configs(config, sweep)

    dataset = DatasetEval(config_set.base_config.dataset)

    max_ks = config_set.summary.max_ks
    encoder_names = config_set.summary.encoders
    current_LVLM = config_set.base_config.generator.name    
    main_LVLM = "Qwen2.5-VL-7B-Instruct"

    if retrieval:
        adaptive_k_evaluator = AdaptiveRetrievalEvaluator(dataset)
        adaptive_k_evaluator.results_k_star_distributions(encoder_names)
        adaptive_k_evaluator.results_k_star_statistics_and_metrics(encoder_names)
        adaptive_k_evaluator.results_metrics_as_a_function_of_k_oracle(encoder_names)
        adaptive_k_evaluator.result_ranking_metrics(encoder_names)

    if generation:
        accuracy_evaluator = AccuracyEvaluator(dataset)

        print('\n### Evaluation of generated answers across configurations :')
        for conf in config_set.configs:
            state = EvalState(
                mode=conf.retrieval.mode, 
                encoder=conf.retrieval.encoder, 
                gen_id=conf.generator.gen_id
            )
            accuracy_evaluator.run_answer_evaluation(state)

        accuracy_evaluator.result_answer_generation_accuracy(encoder_names, current_LVLM, max_ks)
        accuracy_evaluator.results_accuracy_vs_similarity_matrix_sparsity(encoder_names, max_ks, main_LVLM)

    if latency:
        latency_evaluator = LatencyEvaluator(dataset)
        latency_evaluator.run_ViSAR_Approx_latencies(encoder_names)
        latency_evaluator.results_latency_ViSAR_vs_lateinteraction(encoder_names, main_LVLM)
        latency_evaluator.results_latency_ViSAR_vs_heuristics(encoder_names, max_ks, main_LVLM)

    if sensitivity:
        sensitivity_analyzer = SensitivityAnalyser(dataset)
        sensitivity_analyzer.run_sensitivity_analyses(encoder_names)
        sensitivity_analyzer.results_sensitivity_analysis_to_T(encoder_names)
        sensitivity_analyzer.results_sensitivity_analysis_to_gamma(encoder_names)

    if ablation:
        visar_ablation = ViSARSelectorAblation(dataset)
        visar_ablation.run_retrieval_ablation('colpali-v1.2')
        visar_ablation.run_generation_from_ablated_retrieval('colpali-v1.2', main_LVLM)
        visar_ablation.results_ablation_study('colpali-v1.2', main_LVLM)

if __name__ == "__main__":
    main()

