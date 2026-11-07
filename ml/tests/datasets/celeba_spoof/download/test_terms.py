"""The dataset terms gate must not be bypassed silently."""

import unittest

from liveness.datasets.celeba_spoof.download.terms import require_dataset_terms


class DatasetTermsTest(unittest.TestCase):
    def test_accepted_terms_do_not_raise(self):
        require_dataset_terms(True)

    def test_missing_acknowledgement_raises(self):
        with self.assertRaises(SystemExit):
            require_dataset_terms(False)
