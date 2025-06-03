from queue import Queue
from threading import Lock

class ThreadSafeQueue(Queue):
    def __init__(self, maxsize=0):
        super().__init__(maxsize=maxsize)
        self._lock = Lock()
    
    def clear(self):
        with self._lock:
            self.queue.clear()
            
    def safe_put(self, item):
        with self._lock:
            if not self.full():
                self.put(item)
                return True
            return False
            
    def safe_get(self):
        with self._lock:
            if not self.empty():
                return self.get()
            return None 