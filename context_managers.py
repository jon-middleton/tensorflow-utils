from contextlib import closing
from datetime import datetime

import os
import psutil
import subprocess
import socket



class Stopwatch:
    """Context manager for timing processes.
    """
    def __init__(self, time_format="%Y.%m.%d %H:%M:%S"):
        """Initializes a Stopwatch context.

        Args:
            time_format (str): format of the start and stop times.
        """
        self.time_format = time_format

    @property
    def elapsed_time(self):
        """Gives the elapsed time in 'days, hours:minutes:seconds' format.

        Returns:
            str
        """
        return str(self.stop_time - self.start_time)

    @property
    def start_timestamp(self):
        """Gives the start time formatted according to self.time_format.

        Returns:
            str
        """
        return self.start_time.strftime(self.time_format)

    @property
    def stop_timestamp(self):
        """Gives the stop time formatted according to self.time_format.

        Returns:
            str
        """
        return self.stop_time.strftime(self.time_format)

    def __enter__(self):
        """Enters a runtime context and records the start time.

        Returns:
            Stopwatch
        """
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Records stop time and computes elapsed time, performs exception handling if necessary,
        and exits the runtime context.

        References:
            https://book.pythontips.com/en/latest/context_managers.html#handling-exceptions
            https://docs.python.org/2/library/sys.html#sys.exc_info

        Args:
            exc_type (class): type of exception
            exc_val (object): value of an exception, always a class instance if the exception type
                is a class object
            exc_tb (traceback): traceback of an exception

        Returns:
            bool
        """
        self.stop_time = datetime.now()
        return False


class TensorboardProcess:
    """Context manager for a Tensorboard instance. Used to wrap a call to Model.fit.
    """
    def __init__(self, log_dir, n_retries=10):
        """Initializes a Tensorboard context manager.

        Args:
            log_dir (str): path to Tensorboard's log directory
            n_retries (int): number of times to check if TensorBoard has opened a socket and thus
            chosen a port number
        """
        self.log_dir = log_dir
        self.n_retries = n_retries
        self.process = None

    @property
    def tb_port(self):
        """Uses psutil to obtain the port on which the Tensorboard instance is served.

        Returns:
            str
        """
        process_object = psutil.Process(self.process.pid)
        process_connections = process_object.connections(kind="inet4")
        port = process_connections[0][3][1]
        return port

    @property
    def socket_exists(self):
        """Uses psutil to check if TensorBoard has opened a socket and thus chosen a port number.

        Returns:
            bool
        """
        process_object = psutil.Process(self.process.pid)
        process_connections = process_object.connections(kind="inet4")
        if len(process_connections) > 0:
            return True
        else:
            return False

    def __enter__(self):
        """Enters a runtime context and assigns a Tensorboard process with indicated log path to
        self.process.

        Returns:
            TensorboardProcess
        """
        self.process = subprocess.Popen(['tensorboard', '--logdir', f'{self.log_dir}'])
        while not self.socket_exists and self.n_retries > 0:
            print("Waiting for TensorBoard.")
            self.n_retries -= 1
            os.system("sleep 1")
        if self.n_retries == 0:
            raise Exception("TensorBoard instantiation failed. Consider increasing n_retries.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Issues kill -9 to the Tensorboard process, performs exception handling if necessary, and
        exits the runtime context.

        References:
            https://book.pythontips.com/en/latest/context_managers.html#handling-exceptions
            https://docs.python.org/2/library/sys.html#sys.exc_info

        Args:
            exc_type (class): type of exception
            exc_val (object): value of an exception, always a class instance if the exception type
                is a class object
            exc_tb (traceback): traceback of an exception

        Returns:
            bool
        """
        self.process.kill()
        return False

