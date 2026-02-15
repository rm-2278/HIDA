import enum
from functools import partial as bind

from . import worker


class Parallel:
    def __init__(self, ctor, strategy):
        self.worker = worker.Worker(bind(self._respond, ctor), strategy, state=True)
        self.callables = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self.callables:
            self.callables[name] = self.worker(Message.CALLABLE, name)()
        if self.callables[name] is Message.ERROR_ATTRIBUTE:
            raise AttributeError(name)
        if self.callables[name]:
            return bind(self.worker, Message.CALL, name)
        else:
            result = self.worker(Message.READ, name)()
            if result is Message.ERROR_ATTRIBUTE:
                raise AttributeError(name)
            return result

    def __len__(self):
        return self.worker(Message.CALL, "__len__")()

    def close(self):
        self.worker.close()

    @staticmethod
    def _respond(ctor, state, message, name, *args, **kwargs):
        state = state or ctor()
        try:
            if message == Message.CALLABLE:
                assert not args and not kwargs, (args, kwargs)
                result = callable(getattr(state, name))
            elif message == Message.CALL:
                result = getattr(state, name)(*args, **kwargs)
            elif message == Message.READ:
                assert not args and not kwargs, (args, kwargs)
                result = getattr(state, name)
            return state, result
        except AttributeError:
            return state, Message.ERROR_ATTRIBUTE


class Message(enum.Enum):
    CALLABLE = 2
    CALL = 3
    READ = 4
    ERROR_ATTRIBUTE = 5
