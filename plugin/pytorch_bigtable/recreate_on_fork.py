import os
from typing import Any,Callable

class RecreateOnFork:
    """Holds a value and recreates it lazilly on forks.

    C++ gRPC does not support forking. This means that we should not reuse the
    objects created by the google-cloud-cpp library in child processes. The
    parent process needs those objects e.g. to compute the sample row keys. We.
    therefore recreate the objects when we discover that a fork happened. We do
    require from the user to not memoize the values returned from this object
    and to not execute fork while references to those objects exist outside of
    this class.
    """
    def __init__(self, create_function: Callable[[], Any]):
        """Create a value which will be recreated in child process on fork.

        Args:
            create_function (Callable[[], ...]): a functor which creates the
                desired value
        """
        self._create_function = create_function
        self.pid = os.getpid()
        # Create the value eagerly to learn about potential problems early.
        self.value = self._create_function()

    def get(self):
        """Get the value created by `create_function` passed in the ctor.

        If the PID of the process changed (i.e. the process has forked), the
        object will be recreated.

        Returns:
            the object created by `create_funtion`
        """
        actual_pid = os.getpid()
        if actual_pid != self.pid:
            self.pid = actual_pid
            self.value = self._create_function()
        return self.value

