

import time
import torch
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from sklearn.metrics import silhouette_score
from typing import List, Tuple, Generator
from sklearn.cluster import HDBSCAN, KMeans, AgglomerativeClustering, BisectingKMeans

from data import Dataset
from config import RetrievalConfig
from models.encoders import BaseEncoder, get_encoder

@dataclass
class RetrievalState():
    mode: str = None
    encoder: str = None
    max_k: int = None

class ImageRetriever():
    def __init__(self, ret: RetrievalConfig):
        self.encoder: BaseEncoder = get_encoder(ret.encoder)
        self.mode = ret.mode
        self.max_k = ret.max_k

    def is_retrieved(self, sample: dict[str, dict]) -> bool:
        return (
            self.mode in sample and self.encoder.name in sample[self.mode]
        )

    def get_state(self) -> RetrievalState:
        return RetrievalState(
            encoder=self.encoder.name, 
            mode=self.mode, 
            max_k=self.max_k
        )

    def retrieve_images(self, dataset: Dataset) -> None:
        if self.mode == 'ViSAR':
            self.from_ViSAR(dataset)
        else:
            self.from_Baselines(dataset)
    
    def from_ViSAR(self, dataset: Dataset) -> None:
        data = dataset.load_data(from_results=True)
        ViSAR = ViSARSelector()

        for sample in tqdm(data, desc='retrieval '):
            if self.is_retrieved(sample):
                continue
            self.encoder.from_pretrained()
            
            embed_query = self.encoder.encode_query(sample['question'])
            embed_pages = self.encoder.load_embed_pages(sample['doc_id'], dataset.embedding_dir)
            
            ranking = ViSAR.score_pages_and_find_k_star(
                embed_query[0], embed_pages
            )

            ranking['scores'] = [round(s, 3) for s in ranking['scores']]

            sample.setdefault(self.mode, {}).update({
                self.encoder.name: {
                    "indexes": ranking['indexes'],
                    "scores": ranking['scores'],
                    "k_star": ranking['k_star'],
                    "latency": ranking['latency']
                }
            })

            dataset.new_data = True
            dataset.save_results(data, 30)
        
        dataset.save_results(data)
        self.encoder.empty_memory()

    def from_Baselines(self, dataset: Dataset) -> None:
        data = dataset.load_data(from_results=True)
        heurstics = HeuristicsSelector()

        for sample in tqdm(data, desc='retrieval '):
            if self.is_retrieved(sample):
                continue
            self.encoder.from_pretrained()
            
            embed_query = self.encoder.encode_query(sample['question'])
            embed_pages = self.encoder.load_embed_pages(sample['doc_id'], dataset.embedding_dir)

            ranking = self.late_interaction_scores(
                embed_query, embed_pages
            )

            sample.setdefault(self.mode, {}).update({
                self.encoder.name: {
                    "indexes": ranking["indexes"],
                    "scores": ranking["scores"],
                    "timings": ranking["timings"]
                }
            })

            if self.mode == "LargestGap":
                sample[self.mode][self.encoder.name]['k_star'] = heurstics.LargestGap(ranking["scores"])
            elif self.mode == 'ScoreCluster':
                sample[self.mode][self.encoder.name]['k_star'] = heurstics.ScoreCluster(ranking["scores"])

            dataset.new_data = True
            dataset.save_results(data, 30)

        dataset.save_results(data)
        self.encoder.empty_memory()

    def late_interaction_scores(self, embed_query: torch.Tensor, embed_pages: torch.Tensor):
        if not all(e.isfinite().all() for e in embed_pages):
            print('Warning: page embeddings contains NaN of Inf values. Skipping this query.')
            return {"indexes": [], "scores" : [], 'timings': {}}
        
        start  = time.perf_counter()
        scores = self.encoder.compute_scores(embed_pages, embed_query)
        timing = time.perf_counter() - start

        s_sorted = torch.tensor(scores).sort(descending=True)
        
        return {
            "indexes": s_sorted.indices.tolist(),
            "scores": [round(s, 3) for s in s_sorted.values.tolist()],
            "timings": {'total': timing}
        }

class HeuristicsSelector():
    def LargestGap(self, scores: list):
        """
        Adaptive retrieval heuristic 'Largest-Gap' reproduced from (see main paper):
            Taguchi, C.; Maekawa, S.; and Bhutani, N. 2025. Efficient Context Selection 
            for Long-Context QA: No Tuning, No Iteration, Just Adaptive-k. In Proceedings 
            of the 2025 Conference on Empirical Methods in Natural Language Processing, 20116–20141
        """        
        gaps = [scores[i] - scores[i+1] for i in range(len(scores)-1)]
        k_star = int(np.argmax(gaps) + 1)
        return k_star

    def ScoreCluster(self, scores: list):
        """
        Adaptive retrieval heuristic 'Score-Cluster' reproduced from (see main paper):
            Xu, Y.; Gupta, V.; Aggarwal, R.; Mahadevan, V.; and Krishnamachari, B. 2025. 
            Cluster-based Adaptive Retrieval: Dynamic Context Selection for RAG Applications. 
            arXiv preprint arXiv:2511.14769.
        """
        def _cluster(points: np.ndarray, method: str, param: float | int) -> np.ndarray:
            if method == "hdbscan":
                return HDBSCAN(min_cluster_size=param, copy=False).fit_predict(points)
            elif method == 'kmeans':
                return KMeans(n_clusters=param, random_state=42).fit_predict(points)
            elif method == 'agglomerative':
                return AgglomerativeClustering(n_clusters=param).fit_predict(points)
            elif method == "bisecting_kmeans":
                return BisectingKMeans(n_clusters=param, random_state=42).fit_predict(points)
            else:
                raise ValueError(f"Unknown method: {method}")
            
        def _silhouette(points: np.ndarray, labels: np.ndarray) -> float:
            unique = np.unique(labels)
            if len(unique) < 2:
                return -1.0
            return silhouette_score(points, labels)

        N = len(scores)

        if np.isnan(scores).any():
            return N
    
        dist = 1.0 - np.array(scores, dtype=np.float64)
        delta = (dist - dist.min()) / (dist.max() - dist.min())
        
        ranks = np.arange(N, dtype=np.float64)
        points = np.column_stack([delta, ranks])
    
        best_score  = -np.inf

        # PRELIMINARY EXPERIMENTS: 'hdbscan' yielded the best k_stars
        method = 'hdbscan' # 'hdbscan', 'kmeans', 'agglomerative', 'kmeans'
        param_grid = list(range(2, N))

        for param in param_grid:
            labels = _cluster(points, method, param)
            s = _silhouette(points, labels)
            if s > best_score:
                best_score = s
                best_label = labels
        
        S = [i for i in range(1, N) if best_label[i] != best_label[i - 1]]

        if len(S) == 0:
            C = N
        else:
            G = np.array([delta[i] - delta[i - 1] for i in S])

            # ORIGINAL : G / (G.max() + 1e-6) + np.array(S) / N
            # But it biased k_star toward overly high pages number.
            # we adjust to visual retrieval by removing np.array(S) / N
            # but k_star still remains too high on average.
            G_scores = G / (G.max() + 1e-6) # + np.array(S) / N
            
            i_star = S[int(np.argmax(G_scores))]
            C = i_star  # 0-indexed: i* instead of i* - 1)
    
        k_star = max(1, min(C, N))
        return k_star

# The actual implemnetation of ViSAR, detailed in the paper.
# While function's names are self-exlanatory, the code is commented
# to refer to the corresponding Equation or paragraph in the paper.
class ViSARSelector():
    def score_pages_and_find_k_star(
            self, embed_query: torch.Tensor, embed_pages: torch.Tensor,
            top_T: int = 50, block_size: int = 4, weight_thr: float = 0.0, top_j: int = None, top_p: int = None,
            gamma: float = 100000, epsilon: float = 0, top_s: int = None
        ) -> torch.Tensor:

        # Handles numerical instability observed using ColModernVBERT on few 
        # pages of LongDocURL, corresponding to 7.31% of queries, which are excluded 
        # from evaluation (see section 'Experimental Results' in the main paper)
        if not all(e.isfinite().all() for e in embed_pages):
            print('Warning: page embeddings contains NaN of Inf values. Skipping this query.')
            return {"indexes": [], "scores" : [], "k_star" : 0, "quality": 0, 'latency': {}}

        t0 = time.perf_counter()

        w_pj_h = self.compute_weights(
            embed_query, 
            embed_pages
        )
        t1 = time.perf_counter()

        Sim_pp = self.compute_sim_mat(
            embed_pages,
            w_pj_h,
            top_T=top_T,
            block_size=block_size,
            weight_thr=weight_thr,
            top_j=top_j,
            top_p=top_p,
        )
        t2 = time.perf_counter()

        p_rank = self.compute_ranking(
            Sim_pp,
            gamma=gamma,
            epsilon=epsilon,
            top_s=top_s,
        )
        t3 = time.perf_counter()

        p_rank['latency'] = {
            'weights': t1 - t0,
            'sim_mat': t2 - t1,
            'ranking': t3 - t2,
            'total'  : t3 - t0
        }

        return p_rank

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
        A_weighted = A_pi_tilde * w_i_hat[None, :] * w_p_hat[:, None] # Equation 8 subcomponant: Ã_p,i weighted.
        # get patche-level embeddings relevance score r_pj
        r_pj: list[torch.Tensor] = []
        for sim, A_w in zip(Sim, A_weighted):
            r_pj.append(
                (sim * A_w[:, None].pow(2)).amax(dim=0) # Equation 8
            )

        # Equaion 9
        # get patche-level weights
        r_pj_flat = torch.cat(r_pj)  # flatten to compute a global mean
        r_pj_mean = r_pj_flat.mean() # mean patch relevance across pages
        w_pj = (r_pj_flat - r_pj_mean).clamp(min=0) # Equation 9

        n_v_jp = [len(r_j) for r_j in r_pj] # reshape relevances r_pj back to [N, np] (num pages, num v_pj)
        mm, mx = w_pj.min(), w_pj.max()

        # min-max normalization of patche-level weights
        w_pj_hat = (w_pj - mm) / (mx - mm + 1e-6)
        w_pj_hat = list(torch.split(w_pj_hat, n_v_jp))
        
        return w_pj_hat # return patche-level weights

    def generate_blocks(self, embed_pages: list, patch_weights: list, block_size: int, start_from) -> Generator[Tuple[int, int, List[torch.Tensor], List[torch.Tensor]], None, None]:
        n_pages = len(embed_pages)
        for start in range(start_from, n_pages, block_size):
            end = min(start + block_size, n_pages)
            yield start, end, embed_pages[start:end], patch_weights[start:end]
    
    def same_sizes_blocks(self, pw_block_i: torch.Tensor, pw_block_j: torch.Tensor) -> bool:
        same_size_i = all(w.numel() == len(pw_block_i[0]) for w in pw_block_i)
        same_size_j = all(w.numel() == len(pw_block_j[0]) for w in pw_block_j)
        return same_size_i and same_size_j

    def compute_sim_mat(self, embed_pages: list[torch.Tensor], patch_weights: list[torch.Tensor], top_T: int, block_size: int, weight_thr: float, top_j: int | None, top_p: int | None):
        n_pages = len(embed_pages)
        device  = embed_pages[0].device

        # Identify and filter inactive pages (see main text section Implementation Details)
        # As they have all patch weight to zero, they do not contribute to final similarities and can be ignored to save compute.
        active_mask = torch.tensor([w.max().item() > weight_thr for w in patch_weights], dtype=torch.bool, device=device)
        active_idx  = active_mask.nonzero(as_tuple=True)[0].tolist()

        # If top_p is not None, apply ViSAR-Approx to further reduce the number of pages to the most promising ones.
        # ViSAR-Approx is an accelerated version of ViSAR for very long document, which approximates the similarity matrix
        # and is presented in supplemental material, subsection "Acceleration Strategies for Very Long Documents".
        if top_p is not None and len(active_idx) > top_p:
            active_idx = sorted(active_idx, key=lambda i: patch_weights[i].max().item(), reverse=True)[:top_p]
            active_idx = sorted(active_idx)  # restore original page order for sim mat indexing

        active_embeds  = [embed_pages[i]    for i in active_idx]
        active_weights = [patch_weights[i]  for i in active_idx]

        # If top_j is not None, apply ViSAR-Approx to further reduce the number of pages to the most promising ones.
        # ViSAR-Approx is an accelerated version of ViSAR for very long document, which approximates the similarity matrix
        # and is described in the supplemental material, subsection "Acceleration Strategies for Very Long Documents".
        if top_j is not None:
            active_embeds, active_weights = self.filter_patches(active_embeds, active_weights, top_j)

        # Only compute similarities for the selected pages: reduced [n_active x n_active] similarity matrix.
        n_active    = len(active_idx)
        reduced_sim = torch.zeros((n_active, n_active), device=device)

        # compute the (reduced) similarity matrix
        # generate_blocks allows to evaluate patch embedding similarity block-wise, to avoid peak memory usage.
        # This is strictly equivalent than using a dense tensor to compute all similarities at once.
        for i_start, i_end, block_i, pw_block_i in self.generate_blocks(active_embeds, active_weights, block_size, start_from=0):
            for j_start, j_end, block_j, pw_block_j in self.generate_blocks(active_embeds, active_weights, block_size, start_from=i_start):
                b1, b2 = i_end - i_start, j_end - j_start
                sim_i_block = torch.zeros((b1, b2), device=device)
                sim_j_block = torch.zeros_like(sim_i_block)

                # compute block similarities from homogenuous blocks
                # e.g Colpali encoder yield the same number of embedding per page
                if self.same_sizes_blocks(pw_block_i, pw_block_j):
                    bl_i = torch.stack(block_i)    # [b1, N_p, D]
                    bl_j = torch.stack(block_j)    # [b2, N_p, D]
                    pw_i = torch.stack(pw_block_i) # [b1, N_p]
                    pw_j = torch.stack(pw_block_j) # [b2, N_p]

                    # Dot-product in Equation 10
                    patch_sims = torch.einsum("pnd,qmd->pqnm", bl_i, bl_j)   # (b1, b2, n_patches, n_patches)

                    # Equation 10: 
                    # both direction p->p' and p'->p are computed simultaneously.
                    # The similarity is directional, since each source patch independently searches for its best match in the target page.
                    sim_i_to_j = (patch_sims * pw_j[None, :, None, :]).amax(dim=3) * pw_i[:, None, :]
                    sim_j_to_i = (patch_sims * pw_i[:, None, :, None]).amax(dim=2) * pw_j[None, :, :]

                    # Equation 11 (without the square root)
                    sim_i_block = sim_i_to_j.topk(min(top_T, sim_i_to_j.size(2)), dim=2).values.mean(dim=2)
                    sim_j_block = sim_j_to_i.topk(min(top_T, sim_j_to_i.size(2)), dim=2).values.mean(dim=2)

                # compute block similarities from heterogenuous blocks
                # e.g ColQwen encoder can yield a varying number of embeddings per page.
                else:
                    for bi, pi, wi in zip(range(b1), block_i, pw_block_i):
                        for bj, pj, wj in zip(range(b2), block_j, pw_block_j):
                            patch_sims = pi @ pj.T                            # [N_pi, N_pj]
                            # Equation 10: 
                            # both direction p->p' and p'->p are computed simultaneously.
                            # The similarity is directional, since each source patch independently searches for its best match in the target page.
                            sim_i_to_j = (patch_sims * wj[None, :]).amax(dim=1) * wi
                            sim_j_to_i = (patch_sims * wi[:, None]).amax(dim=0) * wj

                            # Equation 11 (without the square root)
                            sim_i_block[bi, bj] = sim_i_to_j.topk(min(top_T, sim_i_to_j.size(0))).values.mean()
                            sim_j_block[bi, bj] = sim_j_to_i.topk(min(top_T, sim_j_to_i.size(0))).values.mean()                            

                reduced_sim[i_start:i_end, j_start:j_end] = sim_i_block
                if i_start != j_start:
                    reduced_sim[j_start:j_end, i_start:i_end] = sim_j_block.T

        # Scatter back the reduced matrix into the full [n_pages x n_pages] matrix.
        similarity_matrix = torch.zeros((n_pages, n_pages), device=device)
        idx = torch.tensor(active_idx, device=device)
        similarity_matrix[idx[:, None], idx[None, :]] = reduced_sim        

        # Equation 11 square root
        similarity_matrix = similarity_matrix.sqrt()
        mn = similarity_matrix.min()
        mx = similarity_matrix.max()
        return (similarity_matrix - mn) / (mx - mn + 1e-6)

    def filter_patches(self, embed_pages: list[torch.Tensor], patch_weights: list[torch.Tensor], top_j: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        filtered_embeds  = []
        filtered_weights = []

        for emb, w in zip(embed_pages, patch_weights):
            n_p = w.shape[0]
            if n_p <= top_j:
                filtered_embeds.append(emb)
                filtered_weights.append(w)
            else:
                topn_idx = w.topk(top_j).indices          # [top_j]
                filtered_embeds.append(emb[topn_idx])     # [top_j, D]
                filtered_weights.append(w[topn_idx])      # [top_j]

        return filtered_embeds, filtered_weights    

    def compute_ranking(self, similarity_matrix: torch.Tensor, gamma: float, epsilon: float, top_s: int | None):
        n_pages = similarity_matrix.shape[0]
        pages = list(range(n_pages))

        # pages self-similarity Sim(p, p)
        s_p = torch.diag(similarity_matrix)
        w_sp = s_p / s_p.sum()

        # rank pages by self-similarity s_p
        ranked_p = s_p.argsort(descending=True).tolist()
        ranked_s = s_p[ranked_p].tolist()

        n_active = (s_p != 0).sum().item()

        # Check degenerated case when there is no leakage and only 1 non-zeros score
        # Which corresponds to singleton disconnected page scoring to 1.0.
        if n_active == 1:
            k_star = 1
        else:
            J_k = []

            # If top_s is not None, apply ViSAR-Approx to limit the search space,
            # ViSAR-Approx is an acceleration strategy of ViSAR for very long document presented 
            # in supplemental material, subsection "Acceleration Strategies for Very Long Documents".
            top_s = n_active if top_s is None else top_s
            max_pages = min(n_active, top_s + 3, n_pages - 1)

            for k in range(1, max_pages):
                # Build relevant and irrelevant candidate sets.
                R_k = ranked_p[:k]
                I_k = [p for p in pages if p not in set(R_k)]

                c_kp = similarity_matrix[R_k][:, R_k].mean(dim=1) # Equation 12
                l_kp = similarity_matrix[R_k][:, I_k].mean(dim=1) # Equation 13

                # Equation 14
                J_k.append(
                    (w_sp[R_k] * (c_kp - gamma * l_kp)).sum().item()
                )

            J_k = np.array(J_k)
            J_k = (J_k - J_k.min()) / (J_k.max() - J_k.min() + 1e-6)
            
            arg = np.argmin(J_k)
            offset = round(np.log(arg + 1))

            # As the minimum J_k* may imply residual leakage from pages in I_k, we evaluate whether k* 
            # corresponds to a sharp transition by accepting R_k*+1 only if J varies more sharply beyond k* than before it,
            # indicating a large drop in leakage caused by the transitioning page (see Algorithm~S1 in suplemental material)
            if arg != 0 and arg < len(J_k) - offset:
                v_m = np.mean([abs(J_k[arg + i] - J_k[arg - i - 1]) for i in range(0, offset)])
                v_p = np.mean([abs(J_k[arg + i] - J_k[arg + i + 1]) for i in range(0, offset)])
                arg = arg + int(v_p - v_m > epsilon)

            k_star = min(int(arg + 1), top_s)

        return {
            "indexes": ranked_p, 
            "scores" : ranked_s, 
            "k_star" : k_star,
        }
    