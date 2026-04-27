"""CogniCore Server — REST API for environments."""

try:
    from cognicore.server.app import create_app

    __all__ = ["create_app"]
except ImportError:
    __all__ = []
