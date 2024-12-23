from typing import Callable
import os
from beartype import beartype, BeartypeConf


if "WHISPERAUDIOSPLITTER_TYPECHECKING" not in os.environ:
    os.environ["WHISPERAUDIOSPLITTER_TYPECHECKING"] = "warn"

if os.environ["WHISPERAUDIOSPLITTER_TYPECHECKING"] == "crash":
    optional_typecheck = beartype
elif os.environ["WHISPERAUDIOSPLITTER_TYPECHECKING"] == "warn":
    optional_typecheck = beartype(
        conf=BeartypeConf(violation_type=UserWarning))
elif os.environ["WHISPERAUDIOSPLITTER_TYPECHECKING"] == "disabled":
    @beartype
    def optional_typecheck(func: Callable) -> Callable:
        return func
else:
    raise ValueError("Unexpected WHISPERAUDIOSPLITTER_TYPECHECKING env value")
