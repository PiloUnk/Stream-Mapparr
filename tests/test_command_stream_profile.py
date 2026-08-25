"""Throughput probing of channels that play through a command stream profile.

Some channels are not played by fetching Stream.url: Dispatcharr runs the
program held by the channel's stream profile, and the URL is only a marker that
program understands. Distalker's MAG portals are the case that surfaced this —
their URLs use a .invalid host and never resolve — and streamlink or yt-dlp
profiles behave the same way.

Opening such a URL always failed, so the probe returned nothing, and
_classify_stream_throughput tiers a stream with no measurement as 2 (unknown).
The sort is ascending on that tier, so every stream from such a provider ranked
below every provider that could be measured healthy, silently and permanently.
These tests pin the lookup and the piped measurement that fix it.
"""
import sys

import pytest


class _Objects:
    """Stands in for Model.objects with just .filter().first()."""

    def __init__(self, row=None, raises=None):
        self.row = row
        self.raises = raises

    def filter(self, **kwargs):
        if self.raises:
            raise self.raises
        return self

    def first(self):
        return self.row


class _Profile:
    def __init__(self, command=None, proxy=False, redirect=False):
        self._command = command if command is not None else ["/usr/bin/prog", "URL"]
        self._proxy = proxy
        self._redirect = redirect
        self.built = None

    def is_proxy(self):
        return self._proxy

    def is_redirect(self):
        return self._redirect

    def build_command(self, url, user_agent, channel_id=None):
        self.built = (url, user_agent, channel_id)
        return list(self._command)


class _Channel:
    def __init__(self, profile):
        self._profile = profile

    def get_stream_profile(self):
        return self._profile


STREAM = {"id": 7, "url": "http://distalker.invalid/portal/Y21k", "m3u_account_id": 3}


@pytest.fixture
def plugin(plugin_module):
    return plugin_module.Plugin.__new__(plugin_module.Plugin)


@pytest.fixture
def quiet_logger():
    import logging

    logger = logging.getLogger("stream_mapparr.tests")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def _wire(monkeypatch, plugin_module, channel=None, raises=None):
    monkeypatch.setattr(
        plugin_module, "Channel", type("C", (), {"objects": _Objects(channel, raises)}),
        raising=False,
    )


# --- finding the profile's command ---------------------------------------


def test_command_profile_yields_argv(plugin, plugin_module, monkeypatch, quiet_logger):
    profile = _Profile(command=["/py", "/resolver.py", "{streamUrl}", "{userAgent}"])
    _wire(monkeypatch, plugin_module, channel=_Channel(profile))

    command = plugin._stream_profile_command(STREAM, 42, "MAG200 stbapp", quiet_logger)

    assert command == ["/py", "/resolver.py", "{streamUrl}", "{userAgent}"]
    assert profile.built == (STREAM["url"], "MAG200 stbapp", 42)


@pytest.mark.parametrize("profile", [
    _Profile(proxy=True),
    _Profile(redirect=True),
    _Profile(command=[]),
])
def test_profiles_without_a_command_probe_the_url(
    plugin, plugin_module, monkeypatch, quiet_logger, profile
):
    _wire(monkeypatch, plugin_module, channel=_Channel(profile))
    assert plugin._stream_profile_command(STREAM, 42, "UA", quiet_logger) is None


def test_unassigned_stream_probes_the_url(plugin, plugin_module, monkeypatch, quiet_logger):
    _wire(monkeypatch, plugin_module, channel=_Channel(_Profile()))
    assert plugin._stream_profile_command(STREAM, None, "UA", quiet_logger) is None


def test_orm_failure_probes_the_url(plugin, plugin_module, monkeypatch, quiet_logger):
    """A surprise in the ORM costs the old behaviour, never the probe."""
    _wire(monkeypatch, plugin_module, raises=RuntimeError("database is gone"))
    assert plugin._stream_profile_command(STREAM, 42, "UA", quiet_logger) is None


# --- measuring through the profile ---------------------------------------


def _writer(payload_bytes, chunk=65536):
    """A program that writes payload_bytes to stdout, then exits."""
    return [
        sys.executable, "-c",
        "import sys\n"
        f"left = {payload_bytes}\n"
        f"block = b'x' * {chunk}\n"
        "while left > 0:\n"
        "    sys.stdout.buffer.write(block[:left])\n"
        "    left -= len(block[:left])\n"
        "sys.stdout.buffer.flush()\n",
    ]


def test_a_measurable_stream_reports_mbps(plugin, quiet_logger):
    mbps, edge = plugin._probe_command_throughput(_writer(2 * 1024 * 1024), 1, quiet_logger)

    assert mbps is not None and mbps > 0
    # No edge host: the connection is made inside the profile's program.
    assert edge is None


def test_a_program_that_produces_nothing_is_not_zero_mbps(plugin, quiet_logger):
    """Zero would rank the source as too slow; unknown is the honest answer."""
    mbps, edge = plugin._probe_command_throughput(
        [sys.executable, "-c", "pass"], 1, quiet_logger
    )

    assert mbps is None
    assert edge is None


def test_a_missing_program_is_not_a_crash(plugin, quiet_logger):
    mbps, _ = plugin._probe_command_throughput(
        ["/nonexistent/program/for/tests"], 1, quiet_logger
    )
    assert mbps is None


def test_a_silent_program_does_not_hang_the_probe(plugin, quiet_logger):
    """A program that connects and never writes must not hold the run open."""
    import time

    started = time.time()
    mbps, _ = plugin._probe_command_throughput(
        [sys.executable, "-c", "import time; time.sleep(120)"], 1, quiet_logger
    )
    elapsed = time.time() - started

    assert mbps is None
    # The watchdog is window + 15s; the point is that it is not 120.
    assert elapsed < 30


# --- the URL path is untouched -------------------------------------------


def test_url_probing_still_runs_without_a_command(plugin, plugin_module, monkeypatch, quiet_logger):
    calls = []

    def _urlopen(req, timeout=None):
        calls.append(req)
        raise RuntimeError("no network in tests")

    monkeypatch.setattr(plugin_module.urllib.request, "urlopen", _urlopen)
    mbps, edge = plugin._probe_stream_throughput(
        "http://provider.example/1.ts", 1, "UA", quiet_logger
    )

    assert (mbps, edge) == (None, None)
    assert len(calls) == 1


def test_a_command_stream_never_touches_urlopen(plugin, plugin_module, monkeypatch, quiet_logger):
    def _urlopen(req, timeout=None):
        raise AssertionError("the URL must not be opened for a command profile")

    monkeypatch.setattr(plugin_module.urllib.request, "urlopen", _urlopen)
    mbps, _ = plugin._probe_stream_throughput(
        STREAM["url"], 1, "UA", quiet_logger, source_command=_writer(2 * 1024 * 1024)
    )

    assert mbps is not None and mbps > 0
