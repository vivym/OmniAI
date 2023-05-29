import importlib
import pkg_resources
from packaging.version import Version
from typing import Callable


def compare_version(
    package: str,
    op: Callable[[Version, Version], bool],
    version: str,
    use_base_version: bool = False,
) -> bool:
    """Compare package version with some requirements.

    >>> compare_version("torch", operator.ge, "0.1")
    True
    >>> compare_version("does_not_exist", operator.ge, "0.0")
    False
    """

    try:
        pkg = importlib.import_module(package)
    except (ImportError, pkg_resources.DistributionNotFound):
        return False

    try:
        if hasattr(pkg, "__version__"):
            pkg_version = Version(pkg.__version__)
        else:
            # try pkg_resources to infer version
            pkg_version = Version(pkg_resources.get_distribution(package).version)
    except TypeError:
        # this is mocked by Sphinx, so it should return True to generate all summaries
        return True
    if use_base_version:
        pkg_version = Version(pkg_version.base_version)

    return op(pkg_version, Version(version))
