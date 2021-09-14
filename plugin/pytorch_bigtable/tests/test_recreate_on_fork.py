import os
from unittest import TestCase
from datetime import datetime

from pytorch_bigtable import row_set
from pytorch_bigtable.recreate_on_fork import RecreateOnFork

class RecreateOnForkTest(TestCase):
    def test_not_recreated(self):
        recreated = RecreateOnFork(lambda: "A newly created object: %s" %
                datetime.now())
        start_pid = os.getpid()
        instance = recreated.get()
        self.assertTrue(instance.startswith("A newly created object"))
        # Let's make sure it is not recreated every time.
        another_instance = recreated.get()
        self.assertTrue(start_pid != os.getpid() or
                id(instance) == id(another_instance))

    def test_on_pid_change(self):
        recreated = RecreateOnFork(lambda: "A newly created object: %s" %
                datetime.now())
        instance = recreated.get()
        child_pid = os.fork()
        if (child_pid == 0):
            # actually, I am the child
            if id(instance) == id(recreated.get()):
                # Doesn't work - still the same object, even though we forked.
                os._exit(1)
            os._exit(0)
        else:
            child_res = os.waitpid(child_pid, 0)
            self.assertEqual(child_pid, child_res[0])
            self.assertTrue(os.WIFEXITED(child_res[1]))
            self.assertEqual(0, os.WEXITSTATUS(child_res[1]))

