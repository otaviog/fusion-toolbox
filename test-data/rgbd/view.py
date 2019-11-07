#!/usr/bin/env python

from pathlib import Path

import fire

from fiontb.data.ftb import load_ftb
from fiontb.viz.datasetviewer import DatasetViewer


class ViewSamples:
    def _view(self, path):
        dataset = load_ftb(Path(__file__).parent / path)
        DatasetViewer(dataset).run()

    def sample1(self):
        self._view("sample1")

    def sample2(self):
        self._view("sample2")


if __name__ == '__main__':
    fire.Fire(ViewSamples)
