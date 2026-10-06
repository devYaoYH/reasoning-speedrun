"""Offline checks for replication identity, related controls and honest aggregates."""
from copy import deepcopy
import unittest

from qed.viewer.clusters import assignment, summarize_clusters


class AttemptClusterTests(unittest.TestCase):
    def setUp(self):
        self.config = {'runner_id': 'speedrun_v2', 'strategy': 'speedrun_v2',
                       'model': 'test/model', 'parallelism': 30, 'rollouts': 1,
                       'schedule': 'barrier', 'first_pass_max_tokens': 8192,
                       'max_tokens': 16384, 'max_attempts_per_question': 4,
                       'seed': 42, 'benchmark': True, 'reuse_server': True,
                       'git_commit': 'a'*40, 'launch_profile': {'dtype': 'bfloat16',
                       'gpu-memory-utilization': .95, 'attention-backend': 'FLASHINFER'}}
        self.metadata = {'attempt_id': 'first',
                         'runner': {'module': 'src.attempt_runners.speedrun_v2',
                                    'version': 'speedrun_v2', 'git_commit': 'a'*40, 'git_dirty': False},
                         'model': {'id': 'test/model', 'quantization': 'none', 'activation_dtype': 'bfloat16'},
                         'controls': {'dataset': 'AIME 2025', 'question_indices': [1, 2],
                                      'hyperparameters': {'system_prompt_sha256': 'prompt'}},
                         'provenance': {'profile_sha256': 'profile',
                                        'dataset': {'id': 'aime_2025', 'year': 2025, 'revision': 'r1'}}}

    def changed(self, **changes):
        return assignment(self.metadata, {**self.config, **changes})

    def test_timestamps_paths_and_outcomes_do_not_split_matched_settings(self):
        first = self.changed(attempt_id='first', official_started_at_utc='2026-10-03T20:00:00Z',
                             inference_warmup={'duration': 20}, grader_python='/old/python')
        second = self.changed(attempt_id='second', official_started_at_utc='2026-10-03T21:00:00Z',
                              inference_warmup={'duration': 1}, grader_python='/new/python')
        self.assertEqual(first['replication_id'], second['replication_id'])

    def test_workload_warmup_controls_split_families_but_observed_cost_does_not(self):
        base = self.changed(skip_benchmark_prewarm=False, prewarm_max_tokens=8192,
                            benchmark_prewarm={'total_latency_s':50,'started_at_utc':'old'})
        repeat = self.changed(skip_benchmark_prewarm=False, prewarm_max_tokens=8192,
                              benchmark_prewarm={'total_latency_s':60,'started_at_utc':'new'})
        self.assertEqual(base['replication_id'], repeat['replication_id'])
        for changes in ({'skip_benchmark_prewarm':True,'prewarm_max_tokens':8192},
                        {'skip_benchmark_prewarm':False,'prewarm_max_tokens':4096}):
            self.assertNotEqual(base['family_id'], self.changed(**changes)['family_id'])

    def test_related_seed_profile_reuse_and_source_changes_are_separate_repeats(self):
        base = self.changed()
        for change in [{'seed': 43}, {'benchmark': False}, {'reuse_server': False}, {'git_commit': 'b'*40}]:
            with self.subTest(change=change):
                other = self.changed(**change)
                self.assertEqual(base['family_id'], other['family_id'])
                self.assertNotEqual(base['replication_id'], other['replication_id'])

    def test_semantic_policy_changes_create_separate_families(self):
        base = self.changed()
        changes = [{'parallelism': 8}, {'rollouts': 2}, {'first_pass_max_tokens': 16384},
                   {'schedule': 'eager'}, {'max_attempts_per_question': 8},
                   {'runner_id': 'speedrun_v4'},
                   {'launch_profile': {**self.config['launch_profile'], 'attention-backend': 'FLASH_ATTN'}}]
        for change in changes:
            with self.subTest(change=change):
                metadata = deepcopy(self.metadata)
                if 'runner_id' in change:
                    metadata['runner']['version'] = change['runner_id']
                self.assertNotEqual(base['family_id'], assignment(metadata, {**self.config, **change})['family_id'])
        for group, key, value in [('model', 'quantization', 'modelopt_fp4'),
                                  ('controls', 'question_indices', [1, 3])]:
            metadata = deepcopy(self.metadata)
            metadata[group][key] = value
            self.assertNotEqual(base['family_id'], assignment(metadata, self.config)['family_id'])

    def test_dataset_revision_is_related_but_never_matched(self):
        base = self.changed()
        metadata = deepcopy(self.metadata)
        metadata['provenance']['dataset']['revision'] = 'r2'
        other = assignment(metadata, self.config)
        self.assertEqual(base['family_id'], other['family_id'])
        self.assertNotEqual(base['replication_id'], other['replication_id'])
        metadata['provenance']['dataset']['year'] = 2026
        metadata['provenance']['dataset']['id'] = 'aime_2026'
        metadata['controls']['dataset'] = 'AIME 2026'
        self.assertNotEqual(base['family_id'], assignment(metadata, self.config)['family_id'])

    def test_unverified_source_is_not_claimed_as_identical(self):
        for source in [{'git_dirty': True}, {'git_commit': None}]:
            metadata = deepcopy(self.metadata)
            metadata['runner'].update(source)
            first = assignment(metadata, self.config)
            metadata['attempt_id'] = 'second'
            second = assignment(metadata, self.config)
            self.assertEqual(first['family_id'], second['family_id'])
            self.assertNotEqual(first['replication_id'], second['replication_id'])

    def test_cluster_stats_retain_unmet_runs_and_separate_repeated_settings(self):
        rows = [{'id': f'run-{i}', 'cluster': self.changed(seed=42 if i<3 else 43),
                 'benchmark_year': 2025, 'status': 'failed' if time is None else 'reached',
                 'time_to_18_s': time, 'first_grader_request_s': i+1}
                for i, time in enumerate([70, 90, None, 80])]
        clusters = summarize_clusters(rows)
        self.assertEqual(clusters, summarize_clusters(list(reversed(rows))))
        cluster = clusters[0]
        self.assertEqual(cluster['count'], 4)
        self.assertEqual(cluster['statuses'], {'reached': 3, 'failed': 1})
        self.assertEqual(cluster['time_to_18'], {'n': 3, 'median_s': 80, 'min_s': 70, 'max_s': 90})
        self.assertEqual(sorted(g['count'] for g in cluster['replications']), [1, 3])
        self.assertEqual(cluster['initial_latency']['n'], 4)
        self.assertIn('config.seed', [v['variable'] for v in cluster['varying_controls']])


if __name__ == '__main__':
    unittest.main()
