"""Group saved attempts into solving families and matched-setting repeats.

Use assignment while building viewer results, then summarize_clusters for a
descriptive replication overview. Families share recorded solving/model/server
settings, but may differ in seed, instrumentation, source commit or server reuse.
Matched groups retain those differences. Nothing is moved, selected by outcome,
or described as an independent replication solely because settings match.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from statistics import median


SOLVING_KEYS = (
    'parallelism', 'rollouts', 'schedule', 'first_pass_max_tokens', 'max_tokens',
    'max_attempts_per_question', 'max_rounds', 'no_continuation', 'target_correct',
    'temperature', 'top_p', 'disable_thinking', 'question_timeout',
    'token_budgets', 'seed_stride', 'max_concurrent_requests', 'budget_mode',
    'initial_rollouts', 'expansion_trigger', 'token_budget_scope', 'grader_cost',
    'skip_benchmark_prewarm', 'prewarm_max_tokens',
)
# Paths, ports, observed service health and timestamps do not change the policy.
TRANSIENT_KEYS = {
    'attempt_id', 'initialization_started_at_utc', 'official_started_at_utc',
    'inference_warmup', 'benchmark_prewarm', 'grader_health', 'server_models', 'models_dir',
    'vllm_python', 'vllm_binary', 'grader_python', 'vllm_url', 'grader_url',
    'vllm_port', 'grader_port', 'model_profile', 'vllm_command', 'questions',
    'dataset_provenance', 'launch_profile', 'system_prompt',
}


def fingerprint(value, prefix):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return prefix + hashlib.sha256(encoded.encode()).hexdigest()[:16]


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = item
    return result


def assignment(metadata, config):
    """Use semantic controls for families and all nontransient config for repeats."""
    runner = metadata['runner']['version']
    policy = 'coverage speedrun' if runner in ('speedrun_v1', 'speedrun_v2') else runner
    dataset = metadata['provenance'].get('dataset') or {}
    dataset = {key: dataset.get(key) for key in ('id', 'year', 'revision', 'prompt_sha256', 'grader_sha256')}
    dataset['name'] = metadata['controls']['dataset']
    profile = {key: value for key, value in (config.get('launch_profile') or {}).items()
               if key not in ('model', 'served-model-name', 'host', 'port')}
    model = {key: metadata['model'].get(key) for key in
             ('id', 'revision', 'quantization', 'activation_dtype', 'kv_cache_dtype', 'linear_backend')}
    family = {'policy': policy, 'model': model,
              'dataset': {key: dataset[key] for key in ('id', 'year', 'name')},
              'question_indices': metadata['controls']['question_indices'],
              'profile': profile, 'solving': {key: config.get(key) for key in SOLVING_KEYS},
              'prompt_sha256': metadata['controls']['hyperparameters'].get('system_prompt_sha256'),
              'simulation': metadata.get('simulation')}
    recorded = {key: value for key, value in config.items() if key not in TRANSIENT_KEYS}
    matched = {'family': family, 'config': recorded, 'dataset': dataset,
               'runner': metadata['runner'], 'profile_sha256': metadata['provenance'].get('profile_sha256')}
    # A dirty checkout cannot establish identical source across two attempts.
    if metadata['runner']['git_dirty'] or not metadata['runner']['git_commit']:
        matched['unverified_source_attempt'] = metadata['attempt_id']
    base = (model['id'] or 'Unknown model').split('/')[-1]
    quant = model['quantization']
    model_label = ('VibeThinker NVFP4' if 'VibeThinker' in base and quant == 'modelopt_fp4' else
                   'VibeThinker BF16' if 'VibeThinker' in base and quant == 'none' and model['activation_dtype'] == 'bfloat16' else base)
    budget = lambda n: f'{n/1024:g}K' if isinstance(n, (int, float)) else '?'
    schedule = config.get('schedule') or config.get('strategy', 'parallel streaming')
    backend = profile.get('attention-backend') or 'default attention'
    utilization = profile.get('gpu-memory-utilization')
    envelope = f'{utilization:.0%} VRAM' if isinstance(utilization, (int, float)) else 'VRAM unrecorded'
    label = (f"{model_label} · {policy} / {schedule} {config.get('parallelism', '?')}×{config.get('rollouts', '?')} · "
             f"{budget(config.get('first_pass_max_tokens', config.get('max_tokens')))}→{budget(config.get('max_tokens'))} · {envelope} · {backend}"
             + (' · SIMULATED' if metadata.get('simulation') else ''))
    return {'family_id': fingerprint(family, 'family-'), 'label': label,
            'replication_id': fingerprint(matched, 'repeat-'),
            'recorded_controls': flatten(matched)}


def measured_stats(rows, field):
    values = [row[field] for row in rows if isinstance(row.get(field), (int, float))
              and math.isfinite(row[field]) and row[field] >= 0]
    return {'n': len(values), 'median_s': median(values) if values else None,
            'min_s': min(values) if values else None, 'max_s': max(values) if values else None}


def summarize_clusters(rows):
    """Retain every outcome; successful-only time summaries also report total n."""
    families = {}
    for row in rows:
        if row.get('cluster'):
            families.setdefault(row['cluster']['family_id'], []).append(row)
    result = []
    for family_id, members in families.items():
        members = sorted(members, key=lambda r: (r.get('attempt_started_at_utc') or '', r['id']))
        repeats = {}
        for row in members:
            repeats.setdefault(row['cluster']['replication_id'], []).append(row)
        controls = [r['cluster']['recorded_controls'] for r in members]
        varying = []
        for key in sorted(set().union(*(c.keys() for c in controls))):
            values = {json.dumps(c.get(key), sort_keys=True): c.get(key) for c in controls}
            if len(values) > 1:
                varying.append({'variable': key, 'values': list(values.values())})
        result.append({'id': family_id, 'label': members[0]['cluster']['label'],
                       'benchmark_year': members[0].get('benchmark_year', 2025),
                       'attempt_ids': [r['id'] for r in members], 'count': len(members),
                       'statuses': dict(Counter(r['status'] for r in members)),
                       'time_to_18': measured_stats(members, 'time_to_18_s'),
                       'initial_latency': measured_stats(members, 'first_grader_request_s'),
                       'varying_controls': varying,
                       'replications': [{'id': key, 'attempt_ids': [r['id'] for r in group],
                                         'count': len(group), 'time_to_18': measured_stats(group, 'time_to_18_s')}
                                        for key, group in repeats.items()]})
    return sorted(result, key=lambda c: (-c['count'], c['label'], c['id']))
