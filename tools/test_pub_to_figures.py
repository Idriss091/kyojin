#!/usr/bin/env python3
"""pub_to_figures.py against synthetic inputs (tests/fixtures/pub). CPU only, no model, no GPU.

Every run happens in a temporary copy of the repository's figure tooling, so the real doc/ and README.md are never written."""
import importlib.util
import json
import os
import shutil
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIX = os.path.join(ROOT, "tests", "fixtures", "pub")
SPEC = importlib.util.spec_from_file_location("pub_to_figures", os.path.join(HERE, "pub_to_figures.py"))
p2f = importlib.util.module_from_spec(SPEC)
sys.modules["pub_to_figures"] = p2f
SPEC.loader.exec_module(p2f)


def make_root(tmp_path):
    """A copy of the figure tooling and templates, with the fixture figures.json."""
    root = tmp_path / "repo"
    (root / "tools").mkdir(parents=True)
    (root / "doc").mkdir()
    for name in ("make_charts.py", "make_readme_table.py", "pub_to_figures.py"):
        shutil.copy(os.path.join(HERE, name), root / "tools" / name)
    shutil.copy(os.path.join(FIX, "figures.json"), root / "doc" / "figures.json")
    shutil.copy(os.path.join(ROOT, "README.tmpl.md"), root / "README.tmpl.md")
    shutil.copy(os.path.join(ROOT, "doc", "benchmarks.tmpl.md"), root / "doc" / "benchmarks.tmpl.md")
    return root


def make_scratch(tmp_path):
    scratch = tmp_path / "scratch"
    shutil.copytree(os.path.join(FIX, "scratch"), scratch)
    return scratch


def load(root):
    with open(root / "doc" / "figures.json", encoding="utf-8") as f:
        return json.load(f)


def model(F, mid):
    return next(m for m in F["models"] if m["id"] == mid)


def test_fixture_round_trips_byte_exact():
    """The writer reproduces the existing file format, so an update only changes the measured lines."""
    with open(os.path.join(FIX, "figures.json"), encoding="utf-8") as f:
        text = f.read()
    assert p2f.dump(json.loads(text)) == text


def test_dry_run_prints_diff_and_writes_nothing(tmp_path, capsys):
    root, scratch = make_root(tmp_path), make_scratch(tmp_path)
    before = (root / "doc" / "figures.json").read_text(encoding="utf-8")
    rc = p2f.main(["--root", str(root), "--scratch", str(scratch), "--commit", "abc1234", "--dry-run"])
    assert rc == 0
    assert (root / "doc" / "figures.json").read_text(encoding="utf-8") == before
    assert not (root / "README.md").exists() and not (root / "doc" / "img").exists()
    out = capsys.readouterr().out
    assert out.startswith("--- a/doc/figures.json")
    assert '+ "commit": "abc1234",' in out
    assert '- "commit": "1ef53ad",' in out
    assert '+    "8192": 1502,' in out


def test_writes_schema_and_renders(tmp_path, capsys):
    root, scratch = make_root(tmp_path), make_scratch(tmp_path)
    old = load(root)
    rc = p2f.main(["--root", str(root), "--scratch", str(scratch), "--commit", "abc1234"])
    assert rc == 0
    F = load(root)
    assert F["commit"] == "abc1234" and F["date"] == "2026-10-17"
    assert list(F) == list(old)  # same top-level keys, same order
    q, g, mimo = model(F, "qwen"), model(F, "glm"), model(F, "mimo")
    assert list(q) == list(model(old, "qwen"))
    assert q["prefill"] == {"8192": 1502, "32768": 1482, "65536": 1469, "131072": 1413, "262144": 1358}
    assert q["decode"] == {"8192": 52.4, "32768": 48.1, "65536": 49.3, "131072": 43.0, "262144": 44.3}
    assert q["spec"] == {"prose": 48.0, "chat": 53.1, "code": 74.6}
    assert q["table"] == {"prefill": "1502 at 8K", "decode": "48.0 to 74.6"}
    assert g["prefill"]["8192"] == 656 and g["table"]["prefill"] == "656 at 8K"
    # before = the previous current values, verbatim
    assert q["before"] == {"prefill": old["models"][0]["prefill"], "decode": old["models"][0]["decode"],
                           "spec": old["models"][0]["spec"]}
    assert g["before"]["spec"] == model(old, "glm")["spec"]
    assert mimo == model(old, "mimo")  # not measured in this pass: untouched
    assert F["rivals"] == old["rivals"] and F["quality"] == old["quality"]
    # the renderers ran
    assert (root / "doc" / "img" / "prefill.svg").exists()
    assert "1502 at 8K" in (root / "README.md").read_text(encoding="utf-8")
    assert "wrote" in capsys.readouterr().out


def test_500k_point_not_published_without_flag(tmp_path, capsys):
    root, scratch = make_root(tmp_path), make_scratch(tmp_path)
    assert p2f.main(["--root", str(root), "--scratch", str(scratch), "--commit", "abc1234", "--no-render"]) == 0
    assert "499712" not in model(load(root), "qwen")["prefill"]
    assert "499712 key" in capsys.readouterr().err


def test_500k_point_with_flag_writes_key(tmp_path):
    # make_charts.py has no 499712 tick yet: rendering a published 500K point fails (see NOTEBOOK.md)
    root, scratch = make_root(tmp_path), make_scratch(tmp_path)
    assert p2f.main(["--root", str(root), "--scratch", str(scratch), "--commit", "abc1234",
                     "--add-500k", "--no-render"]) == 0
    q = model(load(root), "qwen")
    assert q["prefill"]["499712"] == 1291 and q["decode"]["499712"] == 36.2
    assert list(q["prefill"])[-1] == "499712"


def test_missing_point_refuses_and_writes_nothing(tmp_path, capsys):
    root, scratch = make_root(tmp_path), make_scratch(tmp_path)
    before = (root / "doc" / "figures.json").read_text(encoding="utf-8")
    path = scratch / "curve_qwen_c272384.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    del d["points"]["262144"]
    path.write_text(json.dumps(d), encoding="utf-8")
    assert p2f.main(["--root", str(root), "--scratch", str(scratch), "--commit", "abc1234"]) == 1
    assert (root / "doc" / "figures.json").read_text(encoding="utf-8") == before
    err = capsys.readouterr().err
    assert "no point 262144" in err and "nothing written" in err


def test_bench_without_json_block_refuses(tmp_path, capsys):
    root, scratch = make_root(tmp_path), make_scratch(tmp_path)
    (scratch / "bench_glm_c272384.md").write_text("", encoding="utf-8")  # bench still running
    assert p2f.main(["--root", str(root), "--scratch", str(scratch), "--commit", "abc1234"]) == 1
    assert "no JSON block" in capsys.readouterr().err


def test_bad_prompt_token_count_refuses(tmp_path, capsys):
    root, scratch = make_root(tmp_path), make_scratch(tmp_path)
    path = scratch / "curve_glm_c272384.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    d["points"]["65536"]["usage"]["prompt_tokens"] = 40000  # a different prompt than the target
    path.write_text(json.dumps(d), encoding="utf-8")
    assert p2f.main(["--root", str(root), "--scratch", str(scratch), "--commit", "abc1234"]) == 1
    assert "prompt_tokens=40000" in capsys.readouterr().err
