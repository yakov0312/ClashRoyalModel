import yaml


class Config:
    def __init__(self, path="Config.yml"):
        with open(path, "r") as file:
            self._data = yaml.safe_load(file) or {}

    def section(self, name):
        section = self._data.get(name)

        if section is None:
            raise KeyError(f"Missing config section: {name}")

        if not isinstance(section, dict):
            raise TypeError(f"Config section '{name}' must be an object")

        return section