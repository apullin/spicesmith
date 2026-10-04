import pytest

from spicesmith import host


def test_only_explicit_affinity_is_checked(monkeypatch):
    monkeypatch.setattr(host.os, 'sched_getaffinity', lambda pid: {2, 3, 8, 9})
    host.check_cpus(None)
    host.check_cpus('2-3,8')
    for cpus in ('0-3', '3-2', '2,2'):
        with pytest.raises(ValueError, match='invalid or unavailable'):
            host.check_cpus(cpus)
