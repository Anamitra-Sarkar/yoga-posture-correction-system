"""Minimal stand-in for the `modal` package so asanaai_train2.py's functions run UNCHANGED on Kaggle.
Decorated functions are returned as plain functions; Volume.reload/commit are no-ops."""
class _Chain:
    def __getattr__(self, name):
        return lambda *a, **k: self
class Image(_Chain):
    @staticmethod
    def debian_slim(*a, **k):
        return Image()
class _Vol:
    def reload(self): pass
    def commit(self): pass
class Volume:
    @staticmethod
    def from_name(*a, **k):
        return _Vol()
class Secret:
    @staticmethod
    def from_name(*a, **k):
        return None
class App:
    def __init__(self, *a, **k): pass
    def function(self, *a, **k):
        return lambda fn: fn
    def local_entrypoint(self, *a, **k):
        return lambda fn: fn
