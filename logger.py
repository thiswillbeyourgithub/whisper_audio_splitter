import gradio as gr
from typing import Callable, Any
from joblib import hash as jhash
import time
import re
from pathlib import Path
from tqdm import tqdm
import logging
from logging import handlers
import rtoml
import json
from functools import wraps
from platformdirs import user_cache_dir, user_log_dir

from .typechecker import optional_typecheck

cache_dir = Path(user_cache_dir(appname="WhisperAudioSplitter"))
assert cache_dir.parent.exists() or cache_dir.parent.parent.exists(
), f"Invalid cache dir location: '{cache_dir}'"
cache_dir.mkdir(parents=True, exist_ok=True)

log_dir = Path(user_log_dir(appname="WhisperAudioSplitter"))
assert log_dir.parent.exists() or log_dir.parent.parent.exists(
) or log_dir.parent.parent.parent.exists(), f"Invalid log_dir location: '{log_dir}'"
log_dir.mkdir(exist_ok=True, parents=True)
log_file = (log_dir / "logs.txt")
log_file.touch(exist_ok=True)

log_formatter = logging.Formatter(
        fmt='%(asctime)s ##%(levelname)s %(funcName)s(%(lineno)d)## %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S')
file_handler = handlers.RotatingFileHandler(
        log_file,
        mode='a',
        maxBytes=1000000,
        backupCount=100,
        encoding=None,
        delay=0,
        )
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(file_handler)
log_regex = re.compile(" ##.*?##")

mess = f"This is a test to check wether the logging works: {time.time()}"
log.info(mess)
assert mess in log_file.read_text().splitlines()[-1], "Logs does not appear to be working"

colors = {
    "red": "\033[91m",
    "yellow": "\033[93m",
    "reset": "\033[0m",
    "white": "\033[0m",
    "purple": "\033[95m",
    "italic": "\033[3m",
    "bold": "\033[1m",
    "underline": "\033[4m",
    "blink": "\033[5m",

    "bg_red": "\033[41m",
    "bg_green": "\033[42m",
    "bg_yellow": "\033[43m",
    "bg_blue": "\033[44m",
    "bg_magenta": "\033[45m",
    "bg_cyan": "\033[46m",
    "bg_white": "\033[47m",
}
# make highvisions more visible
colors["very_high_vis"] = colors["bold"] + colors["underline"] + colors["italic"] + colors["red"] + colors["blink"] + colors["bg_cyan"]
colors["high_vis"] = colors["bold"] + colors["underline"] + colors["italic"] + colors["red"] + colors["bg_cyan"]

@optional_typecheck
def get_coloured_logger(color_asked: str) -> Callable:
    """used to print color coded logs"""
    col = colors[color_asked]

    # all logs are considered "errors" otherwise the datascience libs just
    # overwhelm the logs
    def printer(string: Any, **args) -> str:
        inp = string
        if isinstance(string, dict):
            try:
                string = rtoml.dumps(string, pretty=True)
            except Exception:
                string = json.dumps(string, indent=2)
        if isinstance(string, list):
            try:
                string = ",".join(string)
            except:
                pass
        try:
            string = str(string)
        except:
            try:
                string = string.__str__()
            except:
                string = string.__repr__()
        log.info(string)
        tqdm.write(col + string + colors["reset"], **args)
        return inp
    return printer


whi = get_coloured_logger("white")
yel = get_coloured_logger("yellow")
red = get_coloured_logger("red")
purp = get_coloured_logger("purple")
ital = get_coloured_logger("italic")
bold = get_coloured_logger("bold")
underline = get_coloured_logger("underline")
high_vis = get_coloured_logger("high_vis")
very_high_vis = get_coloured_logger("very_high_vis")
