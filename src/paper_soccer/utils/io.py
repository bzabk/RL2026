"""Bezpieczny zapis na klastrze: pliki atomowe + logi dopisywane na bieżąco."""
from __future__ import annotations

import csv
import json
import os
import pickle
from pathlib import Path


def save_pickle_atomic(obj, path) -> None:
    """Zapis do pliku tymczasowego + os.replace: ubity job nie zostawi uszkodzonego pliku."""
    path = str(path)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def write_json_atomic(obj, path) -> None:
    path = str(path)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


class CsvLogger:
    """Dopisuje wiersze do CSV i od razu flushuje (log przeżyje ubicie joba).

    Nagłówek zapisuje tylko, gdy plik jest nowy - dzięki temu ``--resume`` dopisuje do tego samego pliku.
    """

    def __init__(self, path, fieldnames):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        new = not self.path.exists() or self.path.stat().st_size == 0
        self._f = open(self.path, "a", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.fieldnames, extrasaction="ignore")
        if new:
            self._w.writeheader()
            self._f.flush()

    def log(self, row: dict) -> None:
        self._w.writerow(row)
        self._f.flush()

    def close(self) -> None:
        self._f.close()
