# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.engine import Engine

from core.utils.logger import get_experiment_name
from core.utils.reid_metric import R1_mAP, R1_mAP_reranking


def _resolve_device(cfg, device=None):
    if device is not None:
        return device
    if cfg.MODEL.DEVICE == 'cuda' and not torch.cuda.is_available():
        return 'cpu'
    return cfg.MODEL.DEVICE


def _detach_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _detach_to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_detach_to_cpu(item) for item in value]
    return value


def _unpack_eval_batch(batch):
    if len(batch) == 4:
        data, pids, camids, paths = batch
    else:
        data, pids, camids = batch
        paths = None
    return data, pids, camids, paths


def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data, return_visuals=False)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def extract_single_image_feature(cfg, model, image_tensor, device=None, return_visuals=None):
    """Extract one image embedding and optional MS-ReID visualization tensors."""
    device = _resolve_device(cfg, device)
    return_visuals = cfg.MODEL.RETURN_VISUALS if return_visuals is None else return_visuals

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    model.to(device)
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor, return_visuals=return_visuals)

    if torch.is_tensor(output):
        return {'embedding': output.detach().cpu(), 'visuals': None}

    visuals = output.get('visuals', {})
    return {
        'embedding': output['embedding'].detach().cpu(),
        'visuals': {
            'wavelet': _detach_to_cpu(visuals.get('wavelet')),
            'attention': _detach_to_cpu(visuals.get('fusion')),
        },
    }


def extract_gallery_features(cfg, model, gallery_loader, device=None):
    """Batch-extract gallery embeddings and keep metadata for later retrieval display."""
    device = _resolve_device(cfg, device)
    model.to(device)
    model.eval()

    embeddings = []
    all_pids = []
    all_camids = []
    all_paths = []

    with torch.no_grad():
        for batch in gallery_loader:
            data, pids, camids, paths = _unpack_eval_batch(batch)
            data = data.to(device)
            feat = model(data, return_visuals=False)
            embeddings.append(feat.detach().cpu())
            all_pids.extend(list(pids))
            all_camids.extend(list(camids))
            if paths is not None:
                all_paths.extend(list(paths))

    return {
        'embeddings': torch.cat(embeddings, dim=0) if embeddings else torch.empty(0),
        'pids': all_pids,
        'camids': all_camids,
        'paths': all_paths if all_paths else None,
    }


def search_single_query(query_embedding, gallery_features, topk=10, normalize=True):
    """Return a front-end friendly top-k retrieval result structure."""
    if isinstance(gallery_features, dict):
        gallery_embeddings = gallery_features['embeddings']
        pids = gallery_features.get('pids')
        camids = gallery_features.get('camids')
        paths = gallery_features.get('paths')
    else:
        gallery_embeddings = gallery_features
        pids = None
        camids = None
        paths = None

    if query_embedding.dim() == 1:
        query_embedding = query_embedding.unsqueeze(0)
    if gallery_embeddings.dim() == 1:
        gallery_embeddings = gallery_embeddings.unsqueeze(0)

    query_embedding = query_embedding.detach().cpu()
    gallery_embeddings = gallery_embeddings.detach().cpu()

    if gallery_embeddings.numel() == 0:
        return {
            'query_embedding': query_embedding[:1],
            'topk': 0,
            'metric': 'cosine' if normalize else 'inner_product',
            'matches': [],
        }

    if normalize:
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
        gallery_embeddings = F.normalize(gallery_embeddings, p=2, dim=1)

    scores = torch.mm(query_embedding[:1], gallery_embeddings.t()).squeeze(0)
    k = max(0, min(topk, scores.numel()))
    if k == 0:
        return {
            'query_embedding': query_embedding[:1],
            'topk': 0,
            'metric': 'cosine' if normalize else 'inner_product',
            'matches': [],
        }
    top_scores, top_indices = torch.topk(scores, k=k, largest=True, sorted=True)

    matches = []
    for rank, (score, index) in enumerate(zip(top_scores, top_indices), start=1):
        idx = int(index.item())
        matches.append({
            'rank': rank,
            'gallery_index': idx,
            'score': float(score.item()),
            'pid': pids[idx] if pids is not None else None,
            'camid': camids[idx] if camids is not None else None,
            'path': paths[idx] if paths is not None else None,
        })

    return {
        'query_embedding': query_embedding[:1],
        'topk': k,
        'metric': 'cosine' if normalize else 'inner_product',
        'matches': matches,
    }


def retrieve_single_query(cfg, model, query_tensor, gallery_loader, topk=10, device=None, return_visuals=None):
    """Convenience interface for one query image against a gallery loader."""
    query_output = extract_single_image_feature(
        cfg,
        model,
        query_tensor,
        device=device,
        return_visuals=return_visuals,
    )
    gallery_features = extract_gallery_features(cfg, model, gallery_loader, device=device)
    retrieval = search_single_query(query_output['embedding'], gallery_features, topk=topk)
    return {
        'query': query_output,
        'gallery': gallery_features,
        'retrieval': retrieval,
    }


def inference(
        cfg,
        model,
        val_loader,
        num_query
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("{}.inference".format(get_experiment_name(cfg.OUTPUT_DIR)))
    logger.info("Enter inferencing")
    if cfg.TEST.RE_RANKING == 'no':
        print("Create evaluator")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    elif cfg.TEST.RE_RANKING == 'yes':
        print("Create evaluator for reranking")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    else:
        print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(cfg.TEST.RE_RANKING))

    evaluator.run(val_loader)
    cmc, mAP = evaluator.state.metrics['r1_mAP']
    logger.info('Validation Results')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
