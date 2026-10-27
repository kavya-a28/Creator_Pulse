"""
CreatorPulse — check that this computer can reach the YouTube Data API.

    python -m src.check_connection

Runs, in order: API key present, DNS lookup of both Google API addresses,
proxy settings, an HTTPS request to the API address the app will use, and one
real API call with your key (costs 1 quota unit). Prints what failed and how
to fix it. The key itself is never printed.
"""

import socket
import sys
import time
import urllib.error
import urllib.request

from src import collect

TEST_VIDEO = "dQw4w9WgXcQ"    # any public video id; only its id is requested


def _dns(host: str):
    t = time.perf_counter()
    try:
        addrs = sorted({a[4][0] for a in socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)})
        return True, f"{', '.join(addrs[:3])} ({(time.perf_counter() - t) * 1000:.0f} ms)"
    except OSError as exc:
        return False, f"lookup failed: {exc}"


def _https(host: str):
    url = f"https://{host}/youtube/v3/videos?part=id&id={TEST_VIDEO}"   # no key -> Google answers 403
    try:
        urllib.request.urlopen(url, timeout=15)
        return True, "reachable"
    except urllib.error.HTTPError as exc:      # any HTTP answer means the server was reached
        return True, f"reachable (HTTP {exc.code} without a key, as expected)"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:160]}"


def run_checks(call_api: bool = True) -> list:
    out = []

    def add(name, ok, detail, fix=""):
        out.append({"check": name, "ok": ok, "detail": detail, "fix": fix if not ok else ""})

    key_ok = collect.api_key_configured()
    add("YouTube API key in .env", key_ok, "configured" if key_ok else "missing",
        "Put YOUTUBE_API_KEY=<your key> in the .env file in the project folder and restart the app.")
    ok1, d1 = _dns(collect.API_HOST)
    add(f"Look up {collect.API_HOST}", ok1, d1,
        "Not fatal if the next check passes: the app then uses www.googleapis.com instead.")
    ok2, d2 = _dns(collect.CLASSIC_HOST)
    add(f"Look up {collect.CLASSIC_HOST}", ok2, d2,
        "Python can't use DNS on this network. Set your DNS to 8.8.8.8 and 1.1.1.1 (Windows Settings > Network "
        "> your connection > DNS server assignment > Edit), turn off VPN/ad-blocking DNS, or try a phone hotspot.")
    proxy = collect.system_proxy()
    out.append({"check": "Proxy", "ok": True, "detail": proxy or "none detected", "fix": ""})
    if proxy:   # names are looked up by the proxy, so local lookup failures don't matter
        for r in out:
            if r["check"].startswith("Look up") and not r["ok"]:
                r.update(ok=True, detail=r["detail"] + " (not needed: requests go through the proxy)", fix="")
    ep = collect.choose_endpoint()
    host = ep["host"]
    if ok1 or ok2 or proxy:
        ok3, d3 = _https(host)
        add(f"HTTPS connection to {host}", ok3, d3,
            "A firewall, antivirus or proxy is blocking Python. Allow python.exe in the firewall/antivirus, or "
            "set HTTPS_PROXY=http://proxy:port in .env if your network needs a proxy.")
    else:
        ok3 = False
    if call_api and key_ok and ok3:
        try:
            yt = collect.build_client()
            collect._call(lambda: yt.videos().list(part="id", id=TEST_VIDEO).execute())
            note = " (using the classic address)" if ep["fallback"] else ""
            add("YouTube API call with your key", True, f"OK via {host}{note}; 1 quota unit used")
        except collect.CollectionError as err:
            add("YouTube API call with your key", False, f"{err.message} {err.detail}".strip(), err.fix)
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    results = run_checks()
    for r in results:
        print(f"[{'OK ' if r['ok'] else 'FAIL'}] {r['check']}: {r['detail']}")
        if r["fix"]:
            print(f"       Fix: {r['fix']}")
    failed = [r for r in results if not r["ok"] and not r["check"].startswith(f"Look up {collect.API_HOST}")]
    print("\nAll good: the app can collect comments." if not failed else "\nSee the first FAIL above.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
