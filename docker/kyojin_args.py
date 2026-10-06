#!/usr/bin/env python3
"""Map KYOJIN_* environment variables to the argument list of one Kyojin server.

Used by docker/entrypoint.sh. Run it directly to see the command a configuration produces:

    KYOJIN_MODEL=qwen MODEL_DIR=/models/pack python docker/kyojin_args.py
"""
import json
import os
import shlex
import sys
from pathlib import Path

SERVERS = {"qwen": "tools/qwen/serve.py", "glm": "tools/glm/serve.py", "mimo": "tools/mimo/serve.py"}
# model_type values found in the config.json of the published packs
TYPE_HINTS = (("qwen", "qwen"), ("glm", "glm"), ("mimo", "mimo"))

# env name -> (flag, {family: flag override or None when the family has no such option}, is_switch)
OPTIONS = {
    "MODEL_ID": ("--model-id", {}, False),
    "DEFAULT_TEMPERATURE": ("--default-temperature", {}, False),
    "DEFAULT_TOP_P": ("--default-top-p", {}, False),
    "DEFAULT_TOP_K": ("--default-top-k", {"glm": None, "mimo": None}, False),
    "DEFAULT_MIN_P": ("--default-min-p", {"glm": None, "mimo": None}, False),
    "REASONING_EFFORT": ("--default-reasoning-effort", {"mimo": None}, False),
    "DRAFT_TOKENS": ("--ndt", {"glm": "--num-draft"}, False),
    "DRAFT_POLICY": ("--draft-policy", {"glm": None, "mimo": None}, False),
    "SLOT_SAVE_PATH": ("--slot-save-path", {}, False),
    "NO_VISION": ("--no-vision", {"glm": None, "mimo": None}, True),
    "NO_THINKING": ("--no-thinking", {"glm": None, "mimo": None}, True),
    "NO_UNCENSOR": ("--no-uncensor", {"mimo": None}, True),
}
TRUE = {"1", "true", "yes", "on"}


class ConfigError(Exception):
    pass


def find_pack(model_dir: Path) -> Path:
    """MODEL_DIR is the pack itself, or a folder holding exactly one pack."""
    if (model_dir / "config.json").is_file():
        return model_dir
    packs = sorted(p.parent for p in model_dir.glob("*/config.json"))
    if len(packs) == 1:
        return packs[0]
    if not model_dir.is_dir():
        raise ConfigError(f"MODEL_DIR={model_dir} does not exist: mount the model folder (-v /path/to/pack:/models:ro)")
    if not packs:
        raise ConfigError(f"no config.json in MODEL_DIR={model_dir}: mount the model folder")
    raise ConfigError(f"MODEL_DIR={model_dir} holds {len(packs)} packs ({', '.join(p.name for p in packs)}): point MODEL_DIR at one")


def detect_family(pack: Path) -> str:
    try:
        cfg = json.loads((pack / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ConfigError(f"cannot read {pack}/config.json: {e}")
    text = " ".join([str(cfg.get("model_type", ""))] + [str(a) for a in cfg.get("architectures", [])]).lower()
    for hint, family in TYPE_HINTS:
        if hint in text:
            return family
    raise ConfigError(f"cannot detect the model family from {pack}/config.json: set KYOJIN_MODEL=qwen|glm|mimo")


def build(env) -> tuple:
    """Return (argv, notes). argv starts after the interpreter: [server.py, flags...]."""
    get = lambda k, d="": env.get("KYOJIN_" + k, d).strip()
    model_dir = Path(env.get("MODEL_DIR", "/models")).expanduser()
    pack = find_pack(model_dir)
    family = get("MODEL", "auto").lower()
    if family == "auto":
        family = detect_family(pack)
    if family not in SERVERS:
        raise ConfigError(f"KYOJIN_MODEL={family!r}: expected qwen, glm, mimo or auto")
    argv = [SERVERS[family], "--model", str(pack), "--host", env.get("HOST", "0.0.0.0"), "--port", env.get("PORT", "8000")]
    if not env.get("PORT", "8000").isdigit():
        raise ConfigError(f"PORT={env['PORT']!r} is not a number")
    ctx = env.get("CONTEXT", "").strip()
    if ctx:
        if not ctx.isdigit() or int(ctx) <= 0:
            raise ConfigError(f"CONTEXT={ctx!r} must be a positive number of tokens")
        argv += ["-c", ctx]
    for name, (flag, overrides, is_switch) in OPTIONS.items():
        value = get(name)
        if not value:
            continue
        flag = overrides.get(family, flag)
        if flag is None:
            raise ConfigError(f"KYOJIN_{name} is not supported by the {family} server")
        if is_switch:
            if value.lower() in TRUE:
                argv.append(flag)
        else:
            argv += [flag, value]
    argv += shlex.split(get("EXTRA_ARGS"))
    return argv, f"family={family} pack={pack}"


def main() -> int:
    try:
        argv, note = build(os.environ)
    except ConfigError as e:
        print(f"[kyojin] configuration error: {e}", file=sys.stderr)
        return 2
    if "--null" in sys.argv[1:]:
        sys.stderr.write(f"[kyojin] {note}\n")
        sys.stdout.write("\0".join(argv) + "\0")
    else:
        print(note)
        print(" ".join(shlex.quote(a) for a in ["python"] + argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
