from glob import glob
from importlib import import_module
from os.path import basename, dirname, isfile, join

from aichat.prompts.prompt import Prompt


class Prompts:
    def __init__(self) -> None:
        super().__init__()
        moduleName = Prompts.__module__
        modulePrefix = f"{moduleName[: moduleName.rfind('.') + 1]}"
        self.prompts: list[Prompt] = []

        # The following code loads all of the prompts from the prompts
        # directory.
        for f in glob(join(dirname(__file__), "*.py")):
            if not isfile(f) or f.endswith(("__init__.py", "prompt.py", "prompts.py")):
                continue

            module = import_module(
                f"{modulePrefix}{basename(f)[:-3]}"
            )  # nosemgrep: non-literal-import

            for attrName in dir(module):
                attr = getattr(module, attrName)

                if isinstance(attr, Prompt):
                    self.prompts.append(attr)

        self.prompts.sort(key=lambda prompt: prompt.priority)


prompts = Prompts()
