"""Errors the CLI turns into exit code 2."""


class DatasetError(Exception):
    """The dataset file is missing, unreadable, or invalid."""


class ConfigError(Exception):
    """The retriever, thresholds, or a report file cannot be used."""
