"""Aggregate saved attempts into measured time-to-18 comparisons for the viewer.

Use build_results with AttemptStore after importing experiment evidence. It
counts distinct oracle-confirmed questions, keeps incomplete/failed attempts
unranked, compares annotated references, and never substitutes settlement time
for the eighteenth positive verdict. Metadata describes interventions; measured
differences across single runs do not establish their isolated causal effects.
"""
from __future__ import annotations

from datetime import datetime
import math

from qed.lib.metadata import build_metadata, dataset_label, validate_metadata
from qed.viewer.store import dataset_record
from qed.viewer.clusters import assignment, summarize_clusters

TARGET = 18
FLOOR_S = 54.0


def first_grader_request(store, attempt_id, overview):
    """Find the earliest recorded request start, including wrong/cancelled jobs.

    Use client dispatch timestamps; fall back to the grader's receipt timestamp
    when client timing is absent. Never infer a request from its verdict time.
    """
    start = overview['attempt']['started_at_utc']
    if not start:
        return None
    try:
        origin = datetime.fromisoformat(start.replace('Z', '+00:00'))
    except (ValueError, TypeError):
        return None
    candidates = []
    folder = store.folder(attempt_id)
    for question in overview['questions']:
        path = f"trace/{question['problem_idx']:02d}/verification.jsonl"
        for event in store.lines(folder, path):
            timestamp = event.get('verification_started_at_utc')
            source = 'client verification_started_at_utc'
            if not timestamp:
                timestamp = (event.get('result') or {}).get('submitted_at')
                source = 'grader submitted_at (receipt timestamp)'
            if not isinstance(timestamp, str):
                continue
            try:
                elapsed = (datetime.fromisoformat(timestamp.replace('Z', '+00:00')) - origin).total_seconds()
            except (ValueError, TypeError):
                continue
            if math.isfinite(elapsed) and elapsed >= 0:
                candidates.append({'elapsed_s': elapsed, 'at_utc': timestamp,
                                   'source': source, 'problem_idx': question['problem_idx']})
    return min(candidates, key=lambda r: r['elapsed_s']) if candidates else None


def solve_events(overview):
    start = overview['attempt']['started_at_utc']
    times = {}
    sources = {}
    events = list(overview['solved_events']) + [q['first_solved'] for q in overview['questions'] if q['first_solved']]
    for event in events:
        idx = event.get('problem_idx')
        elapsed = event.get('first_solved_elapsed_s')
        source = 'recorded first_solved_elapsed_s'
        if elapsed is None and start and event.get('first_solved_at_utc'):
            elapsed = (datetime.fromisoformat(event['first_solved_at_utc'].replace('Z', '+00:00')) -
                       datetime.fromisoformat(start.replace('Z', '+00:00'))).total_seconds()
            source = event.get('source', 'recorded first-solved timestamps')
        if idx is not None and isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed >= 0:
            if idx not in times or elapsed < times[idx]:
                times[idx], sources[idx] = elapsed, source
    return [{'problem_idx': idx, 'elapsed_s': time, 'source': sources[idx]}
            for idx, time in sorted(times.items(), key=lambda pair: pair[1])]


def flattened_controls(metadata):
    """Expose actual settings for reference comparisons, including unknowns."""
    result = {}
    for group in ('runner', 'model', 'gpu', 'controls'):
        for key, value in metadata[group].items():
            if key in ('quantization_source', 'configured_envelope_mib'):
                continue
            if isinstance(value, dict):
                result.update({f'{group}.{key}.{k}': v for k, v in value.items()})
            else:
                result[f'{group}.{key}'] = value
    return result


def build_results(store):
    inventory = store.list()
    warnings = list(inventory['warnings'])
    rows = []
    for attempt in reversed(inventory['attempts']):
        try:
            overview = store.overview(attempt['id'])
            metadata = overview['experiment_metadata']
            build, validate = build_metadata, validate_metadata
            missing = metadata is None
            if missing:
                metadata = build(store.folder(attempt['id']), overview['config'])
                warnings.append(f"{attempt['id']}: metadata.json missing; settings derived from config, intervention unannotated.")
            validate(metadata, attempt['id'])
            dataset = metadata['provenance'].get('dataset') or dataset_record(overview['config'])
            if metadata['controls']['dataset'] != dataset_label(dataset) or dataset['id'] != dataset_record(overview['config'])['id']:
                raise ValueError('Saved metadata dataset disagrees with recorded configuration')
            events = solve_events(overview)
            first_request = first_grader_request(store, attempt['id'], overview)
            time = events[TARGET - 1]['elapsed_s'] if len(events) >= TARGET else None
            summary = overview['summary'] or {}
            solved = summary.get('solved', sum(q['status'] == 'solved' for q in overview['questions']))
            status = ('reached' if time is not None else
                      'timing unavailable' if solved >= TARGET else
                      'interrupted' if attempt['status'] == 'interrupted' else
                      'failed' if attempt['status'] == 'failed' else
                      'in progress' if attempt['status'] not in ('completed', 'failed') else 'target unmet')
            cost = metadata['controls']['grader_cost_s']
            floor = TARGET * cost if cost is not None and metadata['controls']['grader_serial'] else None
            if time is not None and floor is not None and time < floor - 0.01:
                warnings.append(f"{attempt['id']}: recorded time is below its configured serial-grader floor; check timing evidence.")
            rows.append({'id': attempt['id'], 'benchmark_id': dataset['id'], 'benchmark_year': dataset['year'],
                         'benchmark_role': dataset['role'], 'metadata': metadata, 'metadata_missing': missing,
                         'status': status, 'attempt_status': attempt['status'], 'solved': solved,
                         'error': summary.get('error'),
                         'started_at_utc': attempt['started_at_utc'],
                         'attempt_started_at_utc': overview['config'].get('initialization_started_at_utc') or attempt['started_at_utc'],
                         'time_to_18_s': time, 'time_source': events[TARGET - 1]['source'] if time is not None else None,
                         'first_grader_request_s': first_request['elapsed_s'] if first_request else None,
                         'first_grader_request': first_request,
                         'cluster': assignment(metadata, overview['config']),
                         'settlement_s': attempt['official_latency_s'], 'events': events,
                         'grader_floor_s': floor, 'above_floor_s': time - floor if time is not None and floor is not None else None,
                         'comparison': None})
        except (ValueError, OSError, TypeError, KeyError) as exc:
            warnings.append(f"{attempt['id']}: {exc}")
            rows.append({'id': attempt['id'], 'benchmark_year': attempt.get('benchmark_year', 2025), 'metadata': None, 'status': 'invalid evidence',
                         'attempt_status': attempt['status'], 'solved': attempt['solved'],
                         'time_to_18_s': None, 'first_grader_request_s': None,
                         'first_grader_request': None, 'events': [], 'comparison': None})
    by_id = {row['id']: row for row in rows}
    for row in rows:
        if not row['metadata']:
            continue
        ref_id = row['metadata']['intervention']['reference_attempt_id']
        if not ref_id:
            continue
        ref = by_id.get(ref_id)
        if not ref or not ref['metadata']:
            warnings.append(f"{row['id']}: comparison reference {ref_id} unavailable.")
            continue
        current, control = flattened_controls(row['metadata']), flattened_controls(ref['metadata'])
        changed = [{'variable': key, 'before': control.get(key), 'after': current.get(key)}
                   for key in sorted(current.keys() | control.keys()) if current.get(key) != control.get(key)]
        matched = [key for key in sorted(current.keys() & control.keys())
                   if current[key] == control[key] and current[key] is not None]
        time, base = row['time_to_18_s'], ref['time_to_18_s']
        same_benchmark = row.get('benchmark_id') == ref.get('benchmark_id') and row['benchmark_year'] == ref['benchmark_year']
        if not same_benchmark:
            warnings.append(f"{row['id']}: reference uses a different AIME year; timing improvement not compared.")
        measurable = same_benchmark and time is not None and base is not None and base > 0
        row['comparison'] = {'reference_id': ref_id, 'reference_label': ref['metadata']['label'],
                             'changed_controls': changed, 'matched_controls': matched,
                             'saved_s': base - time if measurable else None,
                             'reduction_pct': (base - time) / base * 100 if measurable else None}
    return {'target_correct': TARGET, 'reference_floor_s': FLOOR_S,
            'clusters': summarize_clusters(rows),
            'floor_note': '54 seconds = 18 correct questions × 3 seconds per serial grader query. Assumes zero incorrect/duplicate queries and no startup delay. Generation can overlap grading; initialization and warmup are excluded.',
            'attempts': rows, 'warnings': warnings}
