"""CPU tests for the env -> argument mapping. Run: python -m pytest docker/test_kyojin_args.py"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import kyojin_args as ka


def pack(tmp_path, model_type, name="pack"):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": model_type, "architectures": ["X"]}))
    return d


def test_defaults_and_autodetect(tmp_path):
    p = pack(tmp_path, "qwen4_exp")
    argv, _ = ka.build({"MODEL_DIR": str(p)})
    assert argv == ["tools/qwen/serve.py", "--model", str(p), "--host", "0.0.0.0", "--port", "8000"]


@pytest.mark.parametrize("mt,server", [("glm5_next", "glm"), ("mimo_v2", "mimo"), ("qwen4_exp", "qwen")])
def test_detect(tmp_path, mt, server):
    argv, _ = ka.build({"MODEL_DIR": str(pack(tmp_path, mt))})
    assert argv[0] == f"tools/{server}/serve.py"


def test_explicit_model_overrides_detection(tmp_path):
    argv, _ = ka.build({"MODEL_DIR": str(pack(tmp_path, "qwen4_exp")), "KYOJIN_MODEL": "GLM"})
    assert argv[0] == "tools/glm/serve.py"


def test_parent_folder_with_one_pack(tmp_path):
    p = pack(tmp_path, "mimo_v2")
    argv, _ = ka.build({"MODEL_DIR": str(tmp_path)})
    assert argv[2] == str(p)


def test_parent_folder_with_two_packs(tmp_path):
    pack(tmp_path, "mimo_v2", "a"); pack(tmp_path, "qwen4_exp", "b")
    with pytest.raises(ka.ConfigError, match="2 packs"):
        ka.build({"MODEL_DIR": str(tmp_path)})


def test_port_context_options(tmp_path):
    p = pack(tmp_path, "qwen4_exp")
    argv, _ = ka.build({"MODEL_DIR": str(p), "PORT": "9000", "HOST": "127.0.0.1", "CONTEXT": "32768",
                        "KYOJIN_DEFAULT_TEMPERATURE": "0.6", "KYOJIN_NO_VISION": "1", "KYOJIN_DRAFT_TOKENS": "2",
                        "KYOJIN_REASONING_EFFORT": "low", "KYOJIN_NO_THINKING": "0"})
    assert argv[3:7] == ["127.0.0.1", "--port", "9000", "-c"] or "-c" in argv
    assert argv[argv.index("--port") + 1] == "9000" and argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("-c") + 1] == "32768"
    assert "--no-vision" in argv and "--no-thinking" not in argv
    assert argv[argv.index("--ndt") + 1] == "2" and argv[argv.index("--default-temperature") + 1] == "0.6"


def test_glm_draft_flag_name(tmp_path):
    argv, _ = ka.build({"MODEL_DIR": str(pack(tmp_path, "glm5_next")), "KYOJIN_DRAFT_TOKENS": "3"})
    assert argv[argv.index("--num-draft") + 1] == "3" and "--ndt" not in argv


def test_unsupported_option_is_an_error(tmp_path):
    with pytest.raises(ka.ConfigError, match="not supported by the glm"):
        ka.build({"MODEL_DIR": str(pack(tmp_path, "glm5_next")), "KYOJIN_NO_VISION": "1"})


def test_extra_args_passthrough_keeps_quotes(tmp_path):
    argv, _ = ka.build({"MODEL_DIR": str(pack(tmp_path, "qwen4_exp")),
                        "KYOJIN_EXTRA_ARGS": "--default-max-tokens 512 --model-id 'my model'"})
    assert argv[-4:] == ["--default-max-tokens", "512", "--model-id", "my model"]


@pytest.mark.parametrize("env,msg", [({"CONTEXT": "abc"}, "CONTEXT"), ({"PORT": "x"}, "PORT"), ({"KYOJIN_MODEL": "llama"}, "expected")])
def test_bad_values(tmp_path, env, msg):
    with pytest.raises(ka.ConfigError, match=msg):
        ka.build({"MODEL_DIR": str(pack(tmp_path, "qwen4_exp")), **env})


def test_missing_model_dir(tmp_path):
    with pytest.raises(ka.ConfigError, match="does not exist"):
        ka.build({"MODEL_DIR": str(tmp_path / "nope")})


def test_every_flag_exists_in_the_servers():
    """Each flag emitted for a family must be an add_argument of that server."""
    root = Path(__file__).parent.parent
    for family, script in ka.SERVERS.items():
        src = (root / script).read_text()
        for env_name, (flag, over, _) in ka.OPTIONS.items():
            f = over.get(family, flag)
            if f is not None:
                assert f'"{f}"' in src, f"{family} server has no {f}"
