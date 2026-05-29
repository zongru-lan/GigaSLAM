import rich

_log_styles = {
    "GigaSLAM": "bold green",
    "GUI": "bold magenta",
    "Eval": "bold red",
}

_DEFAULT_SUPPRESS_PREFIXES = [
    "Loop Closure is disabled!",
    "BACKEND: add_next_kf idx:",
    "GUI gaussian packet sent",
    "VIZ render:",
]

_quiet = False
_suppress_prefixes = list(_DEFAULT_SUPPRESS_PREFIXES)


def configure_logging(config=None):
    global _quiet, _suppress_prefixes

    results_cfg = {}
    if isinstance(config, dict):
        results_cfg = config.get("Results", {}) or {}
    logging_cfg = results_cfg.get("logging", {}) or {}

    _quiet = bool(logging_cfg.get("quiet", False))
    _suppress_prefixes = list(
        logging_cfg.get("suppress_prefixes", _DEFAULT_SUPPRESS_PREFIXES)
    )


def logging_is_quiet():
    return _quiet


def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"


def _should_suppress(args):
    if not _quiet:
        return False
    message = " ".join(str(arg) for arg in args)
    return any(message.startswith(prefix) for prefix in _suppress_prefixes)


def Log(*args, tag="GigaSLAM"):
    if _should_suppress(args):
        return
    style = get_style(tag)
    rich.print(f"[{style}]{tag}:[/{style}]", *args)
