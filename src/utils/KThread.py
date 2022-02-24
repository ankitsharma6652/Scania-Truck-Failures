import sys
import trace
import threading

class KThread(threading.Thread):
    """
    This code module allows you to kill threads.  The
class KThread is a drop-in replacement for
threading.Thread.  It adds the kill() method, which
should stop most threads in their tracks.

#
---------------------------------------------------------------------
# KThread.py: A killable Thread implementation.
#
---------------------------------------------------------------------
-----------------------------------------------------------------------
How It Works:
-----------------------------------------------------------------------

The KThread class works by installing a trace in the
thread.  The trace
checks at every line of execution whether it should
terminate itself.
So it's possible to instantly kill any actively
executing Python code.
However, if your code hangs at a lower level than
Python, then the
thread will not actually be killed until the next
Python statement is
executed.


"""


    """A subclass of threading.Thread, with a kill() method."""
    def __init__(self, *args, **keywords):
        threading.Thread.__init__(self, *args, **keywords)
        self.killed = False

    def start(self):
        """Start the thread."""
        self.__run_backup = self.run
        self.run = self.__run      # Force the Thread to
#     install our trace.
        threading.Thread.start(self)

    def __run(self):
        """Hacked run function, which installs the
    trace."""
        sys.settrace(self.globaltrace)
        self.__run_backup()
        self.run = self.__run_backup

    def globaltrace(self, frame, why, arg):
        if why == 'call':
          return self.localtrace
        else:
          return None

    def localtrace(self, frame, why, arg):
        if self.killed:
          if why == 'line':
            raise SystemExit()
        return self.localtrace

    def kill(self):
        self.killed = True
