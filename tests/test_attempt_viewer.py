"""Offline viewer checks for canonical traces, partial copies, and safe HTTP access."""
import contextlib
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from qed.viewer.store import AttemptStore, json_lines
from qed.viewer import server as viewer_server


class AttemptViewerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.attempts = self.root / 'attempts'
        self.id = '20261003T193711.679999Z'
        self.folder = self.attempts / self.id
        self.folder.mkdir(parents=True)
        self.store = AttemptStore(self.attempts)
        (self.folder / 'questions.json').write_text(json.dumps([{'problem_idx': n, 'problem': f'Problem {n}'} for n in (1, 2)]))
        self.save('config.json', {'attempt_id': self.id, 'model': 'Qwen/test', 'question_indices': [1, 2],
                                 'official_started_at_utc': '2026-10-03T19:40:00+00:00'})
        self.save('summary.json', {'attempt_id': self.id, 'status': 'completed', 'solved': 1,
                                  'official_started_at_utc': '2026-10-03T19:40:00+00:00',
                                  'official_latency_s': 30, 'target_correct': 1, 'target_reached': True,
                                  'questions': [{'problem_idx': 1, 'status': 'solved', 'verified_answer': '70'},
                                                {'problem_idx': 2, 'status': 'stopped'}]})
        self.rollouts = [{'rollout': 1, 'round': 1, 'status': 'completed', 'finish_reason': 'length',
                          'generation_censored': True, 'ttft_s': 0.4, 'usage': {'completion_tokens': 8192}},
                         {'rollout': 2, 'round': 2, 'status': 'cancelled', 'continuation_of_rollout': 1,
                          'cached_prompt_tokens': 8000, 'generation_censored': True}]
        self.save('trace/01/question.json', {'problem_idx': 1, 'status': 'solved', 'rollouts': self.rollouts,
                                          'first_solved': {'problem_idx': 1, 'first_solved_elapsed_s': 24,
                                                           'grader_query_id': 'oracle-1'}, 'rounds': [{'round': 1}, {'round': 2}]})
        for row in self.rollouts:
            base = f'trace/01/rollout-{row["rollout"]:02d}'
            self.save(f'{base}/telemetry.json', row)
            self.save(f'{base}/response.json', {'reasoning': 'saved reasoning', 'content': '\\boxed{70}'})
            self.save(f'{base}/request.json', {'messages': [{'role': 'user', 'content': 'Original problem'}]})
        self.lines('trace/01/verification.jsonl', [{'candidate': '69', 'rollout': 1, 'result': {'verdict': False}},
                                                 {'candidate': '70', 'rollout': 2, 'result': {'verdict': True}}])

    def save(self, relative, data):
        path = self.folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def lines(self, relative, rows):
        path = self.folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def test_summary_only_question_and_full_continuation_evidence(self):
        overview = self.store.overview(self.id)
        first, second = overview['questions']
        self.assertEqual(first['first_solved']['grader_query_id'], 'oracle-1')
        self.assertEqual(first['rollouts'][1]['continuation_of_rollout'], 1)
        self.assertEqual(first['rollouts'][1]['cached_prompt_tokens'], 8000)
        self.assertTrue(first['rollouts'][1]['generation_censored'])
        self.assertEqual(first['verification_count'], 2)
        self.assertEqual(second['status'], 'stopped')
        self.assertFalse(second['trace_available'])
        self.assertIsNone(second['verification_count'])
        self.assertNotIn('answer', first)
        trajectory = self.store.rollout(self.id, 1, 2)
        self.assertEqual(trajectory['response']['reasoning'], 'saved reasoning')
        self.assertIn('request.json', trajectory['files'])
        self.assertNotIn('stream.jsonl', trajectory['files'])

    def test_legacy_winner_retains_recorded_verdict_timestamp(self):
        self.save('trace/01/question.json', {'problem_idx': 1, 'status': 'solved', 'rollouts': [],
                                          'winner': {'candidate': '70', 'rollout': 2,
                                                     'verification_finished_at_utc': '2026-10-03T19:40:24+00:00'}})
        first = self.store.overview(self.id)['questions'][0]
        self.assertEqual(first['first_solved']['first_solved_at_utc'], '2026-10-03T19:40:24+00:00')
        self.assertEqual(first['first_solved']['source'], 'winner verification_finished_at_utc')

    def test_partial_attempt_can_show_rollout_without_question_or_summary(self):
        (self.folder / 'summary.json').unlink()
        (self.folder / 'trace/01/question.json').unlink()
        overview = self.store.overview(self.id)
        self.assertIsNone(overview['summary'])
        self.assertEqual(overview['attempt']['status'], 'incomplete')
        self.assertTrue(overview['questions'][0]['trace_available'])
        self.assertEqual(len(overview['questions'][0]['rollouts']), 2)

    def test_incomplete_jsonl_tail_is_ignored_but_corrupt_complete_line_is_not(self):
        path = self.folder / 'solved.jsonl'
        path.write_text('{"problem_idx":1}\n{"problem_idx":')
        self.assertEqual(json_lines(path), [{'problem_idx': 1}])
        path.write_text('{"problem_idx":1}\nbroken\n')
        with self.assertRaises(json.JSONDecodeError):
            json_lines(path)

    def test_gpu_downsampling_preserves_extrema_and_sample_count(self):
        rows = [{'timestamp_utc': str(i), 'vram_used_mib': 10, 'gpu_util_pct': 20} for i in range(1000)]
        rows[123]['vram_used_mib'] = 200
        rows[777]['gpu_util_pct'] = 100
        self.lines('gpu.jsonl', rows)
        data = self.store.gpu(self.id)
        self.assertEqual(data['sample_count'], 1000)
        self.assertLessEqual(len(data['samples']), 604)
        self.assertIn(rows[123], data['samples'])
        self.assertIn(rows[777], data['samples'])

    def test_invalid_directories_are_reported_without_hiding_good_attempts(self):
        (self.attempts / 'bad').mkdir()
        (self.attempts / 'bad/config.json').write_text('{broken')
        result = self.store.list()
        self.assertEqual([a['id'] for a in result['attempts']], [self.id])
        self.assertEqual(len(result['warnings']), 1)

    def test_new_empty_attempt_directory_is_ignored_until_files_exist(self):
        (self.attempts / 'initializing').mkdir()
        result = self.store.list()
        self.assertEqual([a['id'] for a in result['attempts']], [self.id])
        self.assertEqual(result['warnings'], [])

    def test_artifact_allowlist_and_symlink_escape(self):
        for relative in ['../../.env', 'grader_audit.jsonl', 'trace/01/rollout-01/../../config.json']:
            with self.assertRaises(ValueError):
                self.store.artifact(self.id, relative)
        secret = self.root / '.env'
        secret.write_text('secret')
        path = self.folder / 'trace/01/rollout-01/request.json'
        path.unlink()
        path.symlink_to(secret)
        with self.assertRaises(ValueError):
            self.store.rollout(self.id, 1, 1)
        with self.assertRaises(ValueError):
            self.store.artifact(self.id, 'trace/01/rollout-01/request.json')
        for name in ['..', '../outside', '/absolute']:
            with self.assertRaises((ValueError, FileNotFoundError)):
                self.store.folder(name)

    def test_both_viewers_and_canonical_http_contract(self):
        handler = viewer_server.make_handler(self.store)
        with patch.object(handler, 'log_message'):
            server = viewer_server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                conn = http.client.HTTPConnection('127.0.0.1', server.server_port)
                with contextlib.closing(conn):
                    cases = [('/', 200, 'qed · Attempts'.encode()),
                             ('/results', 200, 'qed · Overall results'.encode()),
                             ('/results/viewer.js', 200, b'/api/results'),
                             ('/api/results', 200, b'54.0'),
                             ('/attempts/viewer.js', 200, b'/api/attempts/'),
                             ('/api/attempts', 200, self.id.encode()),
                             (f'/api/attempts/{self.id}/overview', 200, b'oracle-1'),
                             (f'/api/attempts/{self.id}/questions/1/rollouts/2', 200, b'saved reasoning'),
                             (f'/api/attempts/{self.id}/files/config.json', 200, b'Qwen/test'),
                             (f'/api/attempts/{self.id}/files/grader_audit.jsonl', 404, b'Unsupported'),
                             (f'/api/attempts/{self.id}/files/%2E%2E/.env', 404, b'Unsupported')]
                    for path, status, body in cases:
                        conn.request('GET', path)
                        response = conn.getresponse()
                        data = response.read()
                        self.assertEqual(response.status, status, path)
                        self.assertIn(body, data, path)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == '__main__':
    unittest.main()
