import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('collector', Path(__file__).resolve().parents[1] / 'tools/collect-comsol-case.py')
collector = importlib.util.module_from_spec(spec); spec.loader.exec_module(collector)


class VerifiedReadRecovery(unittest.TestCase):
    def test_transient_get_reuses_the_original_identity(self):
        failure = subprocess.CompletedProcess([], 1, '', 'Unverified HTTP 503 response: missing X-WorkAI-Nonce')
        success = subprocess.CompletedProcess([], 0, '{"server_signature_verified":true,"body":{"phase":"running"}}', '')
        with patch.object(collector.subprocess, 'run', side_effect=[failure] * 4 + [success]) as run, patch.object(collector.time, 'sleep'):
            result = collector.request_json(['client', '-JobId', 'immutable-existing-job'], 'status')
        self.assertEqual(result['body']['phase'], 'running')
        self.assertEqual(run.call_count, 5)
        self.assertTrue(all(call.args[0] == ['client', '-JobId', 'immutable-existing-job', '-Action', 'status'] for call in run.call_args_list))

    def test_submit_error_or_timeout_is_never_replayed(self):
        outcomes = [subprocess.CompletedProcess([], 1, '', 'HTTP 503'), subprocess.TimeoutExpired('client', 45)]
        for outcome in outcomes:
            with self.subTest(outcome=outcome), patch.object(collector.subprocess, 'run', side_effect=[outcome]) as run, patch.object(collector.time, 'sleep') as sleep:
                with self.assertRaises(RuntimeError):
                    collector.request_json(['client', '-JobId', 'new-fixed-job'], 'submit')
                self.assertEqual(run.call_count, 1); sleep.assert_not_called()

    def test_bad_signature_is_rejected_without_retry(self):
        response = subprocess.CompletedProcess([], 0, '{"server_signature_verified":false}', '')
        with patch.object(collector.subprocess, 'run', return_value=response) as run, patch.object(collector.time, 'sleep') as sleep:
            with self.assertRaises(ValueError): collector.request_json(['client'], 'status')
            self.assertEqual(run.call_count, 1); sleep.assert_not_called()

    def test_get_timeout_can_recover_without_a_post(self):
        success = subprocess.CompletedProcess([], 0, '{"server_signature_verified":true}', '')
        with patch.object(collector.subprocess, 'run', side_effect=[subprocess.TimeoutExpired('client', 45), success]) as run, patch.object(collector.time, 'sleep'):
            collector.request_json(['client'], 'log')
            self.assertEqual(run.call_count, 2)


if __name__ == '__main__': unittest.main()
