"""Errors the CLI turns into exit code 2."""


class DatasetError(Exception):
    """The dataset file is missing, unreadable, or invalid."""


class ConfigError(Exception):
    """The retriever, thresholds, or a report file cannot be used."""


class RetrieverResponseError(Exception):
    """A retriever response that cannot be scored.

    The message is printed. It must not include header values, userinfo,
    or the query string.
    """
