"""Offline checks for event-derived targets, intervention controls, and metadata."""
import json
from pathlib import Path
import tempfile
import unittest

from qed.lib.metadata import build_metadata, validate_metadata
from qed.viewer.results import build_results
from qed.viewer.store import AttemptStore
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples' / 'attempts'


class AttemptResultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.attempts = self.root / 'attempts'
        self.store = AttemptStore(self.attempts)

    def make(self, id='control', times=None, status='completed', solved=None, reference=None, envelope=.8, metadata=True):
        folder = self.attempts / id
        folder.mkdir(parents=True)
        times = times if times is not None else [3 * i for i in range(1, 19)]
        config = {'attempt_id': id, 'model': 'test/model', 'git_commit': 'a'*40,
                  'strategy': 'coverage', 'parallelism': 30, 'rollouts': 1,
                  'question_indices': list(range(1, 21)), 'grader_cost': 3,
                  'target_correct': 18, 'max_tokens': 16384,
                  'launch_profile': {'dtype': 'bfloat16', 'gpu-memory-utilization': envelope,
                                     'tensor-parallel-size': 1, 'max-model-len': 65536}}
        summary = {'attempt_id': id, 'status': status, 'solved': len(times) if solved is None else solved,
                   'official_started_at_utc': '2026-10-03T20:00:00+00:00', 'official_latency_s': 1000,
                   'questions': [{'problem_idx': i, 'status': 'solved',
                                  'first_solved': {'problem_idx': i, 'first_solved_elapsed_s': t}}
                                 for i, t in enumerate(times, 1)]}
        (folder/'config.json').write_text(json.dumps(config))
        (folder/'summary.json').write_text(json.dumps(summary))
        (folder/'gpu.jsonl').write_text('{"vram_total_mib":81920}\n')
        m = build_metadata(folder)
        m['intervention']['reference_attempt_id'] = reference
        if metadata:
            (folder/'metadata.json').write_text(json.dumps(m))
        return folder, m

    def test_distinct_question_verdicts_determine_target_not_settlement(self):
        folder, _ = self.make(times=list(range(1, 20)))
        (folder/'solved.jsonl').write_text(''.join(json.dumps({'problem_idx': 1, 'first_solved_elapsed_s': 1})+'\n' for _ in range(20)))
        row = build_results(self.store)['attempts'][0]
        self.assertEqual(row['time_to_18_s'], 18)
        self.assertEqual(row['settlement_s'], 1000)
        self.assertEqual(len(row['events']), 19)

    def test_missing_timing_cannot_be_replaced_by_summary_latency(self):
        self.make(times=[], solved=18)
        row = build_results(self.store)['attempts'][0]
        self.assertIsNone(row['time_to_18_s'])
        self.assertEqual(row['status'], 'timing unavailable')

    def test_attempt_timestamp_is_separate_from_official_latency_clock(self):
        folder, _ = self.make()
        path = folder / 'config.json'
        config = json.loads(path.read_text())
        config['initialization_started_at_utc'] = '2026-10-03T19:58:00+00:00'
        path.write_text(json.dumps(config))
        row = build_results(self.store)['attempts'][0]
        self.assertEqual(row['attempt_started_at_utc'], '2026-10-03T19:58:00+00:00')
        self.assertEqual(row['started_at_utc'], '2026-10-03T20:00:00+00:00')
        self.assertEqual(row['time_to_18_s'], 54)
        config.pop('initialization_started_at_utc')
        path.write_text(json.dumps(config))
        row = build_results(self.store)['attempts'][0]
        self.assertEqual(row['attempt_started_at_utc'], row['started_at_utc'])

    def test_unmet_and_interrupted_runs_are_not_ranked(self):
        self.make('unmet', times=[5, 10])
        self.make('interrupted', times=[], status='interrupted')
        self.make('failed', times=[], status='failed')
        result = {r['id']: r for r in build_results(self.store)['attempts']}
        self.assertEqual(result['unmet']['status'], 'target unmet')
        self.assertEqual(result['interrupted']['status'], 'interrupted')
        self.assertEqual(result['failed']['status'], 'failed')
        self.assertTrue(all(r['time_to_18_s'] is None for r in result.values()))

    def test_first_request_includes_wrong_and_cancelled_jobs_before_verdicts(self):
        folder, _ = self.make(times=[], status='interrupted')
        events = [
            {'verification_started_at_utc': '2026-10-03T20:00:09Z', 'result': {'verdict': True}},
            {'verification_started_at_utc': '2026-10-03T20:00:04Z', 'result': {'verdict': False}},
            {'verification_started_at_utc': '2026-10-03T20:00:02.500Z', 'cancelled': True},
            {'verification_started_at_utc': 'bad timestamp'},
            {'verification_started_at_utc': '2026-10-03T19:59:59Z'},
            {'verification_finished_at_utc': '2026-10-03T20:00:01Z'}]
        trace = folder / 'trace/01'
        trace.mkdir(parents=True)
        (trace/'verification.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
        row = build_results(self.store)['attempts'][0]
        self.assertEqual(row['first_grader_request_s'], 2.5)
        self.assertEqual(row['first_grader_request']['problem_idx'], 1)
        self.assertEqual(row['first_grader_request']['source'], 'client verification_started_at_utc')
        self.assertIsNone(row['time_to_18_s'])

    def test_first_request_receipt_fallback_and_missing_timing(self):
        folder, _ = self.make()
        self.assertIsNone(build_results(self.store)['attempts'][0]['first_grader_request_s'])
        trace = folder / 'trace/02'
        trace.mkdir(parents=True)
        (trace/'verification.jsonl').write_text(json.dumps({'result': {
            'submitted_at': '2026-10-03T20:00:07.125Z', 'picked_at': '2026-10-03T20:00:09Z'}})+'\n')
        row = build_results(self.store)['attempts'][0]
        self.assertEqual(row['first_grader_request_s'], 7.125)
        self.assertIn('receipt timestamp', row['first_grader_request']['source'])

    def test_serial_grader_floor_and_reference_delta_include_vram_change(self):
        self.make(times=[10*i for i in range(1, 19)])
        self.make('treatment', times=[5*i for i in range(1, 19)], reference='control', envelope=.95)
        rows = {r['id']: r for r in build_results(self.store)['attempts']}
        row = rows['treatment']
        self.assertEqual(row['grader_floor_s'], 54)
        self.assertEqual(row['above_floor_s'], 36)
        self.assertEqual(row['comparison']['saved_s'], 90)
        self.assertEqual(row['comparison']['reduction_pct'], 50)
        self.assertIn('gpu.memory_utilization', [r['variable'] for r in row['comparison']['changed_controls']])
        self.assertIn('controls.hyperparameters.parallelism', row['comparison']['matched_controls'])

    def test_legacy_timestamp_fallback_and_invalid_events(self):
        folder, _ = self.make()
        summary = json.loads((folder/'summary.json').read_text())
        for q in summary['questions']:
            q['first_solved'].pop('first_solved_elapsed_s')
            q['first_solved']['first_solved_at_utc'] = f"2026-10-03T20:00:{q['problem_idx']*3:02d}Z"
        (folder/'summary.json').write_text(json.dumps(summary))
        (folder/'solved.jsonl').write_text('{"problem_idx":19,"first_solved_elapsed_s":-10}\n')
        row = build_results(self.store)['attempts'][0]
        self.assertEqual(row['time_to_18_s'], 54)
        self.assertIn('timestamps', row['time_source'])

    def test_missing_or_invalid_metadata_is_explicit(self):
        folder, _ = self.make(metadata=False)
        result = build_results(self.store)
        self.assertTrue(result['attempts'][0]['metadata_missing'])
        self.assertIn('unannotated', result['warnings'][0])
        (folder/'metadata.json').write_text('{"schema_version":5}')
        result = build_results(self.store)
        self.assertEqual(result['attempts'][0]['status'], 'invalid evidence')
        self.assertTrue(result['warnings'])

    def test_metadata_schema_bounds_and_identity(self):
        _, m = self.make()
        validate_metadata(m, 'control')
        with self.assertRaisesRegex(ValueError, 'attempt_id'):
            validate_metadata(m, 'different')
        m['gpu']['memory_utilization'] = 1.5
        with self.assertRaisesRegex(ValueError, 'Metadata schema'):
            validate_metadata(m, 'control')

    def test_bundled_example_attempt(self):
        store = AttemptStore(EXAMPLES)
        row = build_results(store)['attempts'][0]
        self.assertEqual(row['id'], '20261004T220514.018752Z')
        self.assertAlmostEqual(row['time_to_18_s'], 62.118150707974564)
        self.assertEqual(row['metadata']['gpu']['memory_utilization'], .95)
        self.assertFalse(row['metadata_missing'])
        self.assertEqual(len(store.overview(row['id'])['questions']), 30)
        self.assertTrue(store.overview(row['id'])['questions'][0]['problem'])


if __name__ == '__main__':
    unittest.main()
