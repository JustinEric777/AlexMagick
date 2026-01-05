import psutil
import os
import time
from typing import Dict, Any, Tuple

class SystemMonitor:
    """Monitor system resources (CPU, Memory, IO)"""
    
    @staticmethod
    def get_current_usage() -> Dict[str, float]:
        """Get instantaneous resource usage"""
        process = psutil.Process(os.getpid())
        
        # CPU
        cpu_percent = process.cpu_percent(interval=None)
        
        # Memory
        mem_info = process.memory_info()
        mem_rss_mb = mem_info.rss / 1024 / 1024
        
        # IO (requires permission, might return 0 if not allowed/supported)
        try:
            io_counters = process.io_counters()
            read_bytes = io_counters.read_bytes
            write_bytes = io_counters.write_bytes
        except (AttributeError, psutil.AccessDenied):
            read_bytes = 0
            write_bytes = 0
            
        return {
            "cpu_percent": cpu_percent,
            "memory_mb": mem_rss_mb,
            "io_read_bytes": read_bytes,
            "io_write_bytes": write_bytes,
            "timestamp": time.time()
        }

    class Tracker:
        """Context manager to track usage over a period"""
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.peak_mem = 0
            self.peak_cpu = 0
            self.start_io = (0, 0)
            self._running = False
            
        def __enter__(self):
            self.start_time = time.time()
            
            # Initial readings
            self.process.cpu_percent(interval=None) # reset counter
            mem = self.process.memory_info().rss / 1024 / 1024
            self.peak_mem = mem
            
            try:
                io = self.process.io_counters()
                self.start_io = (io.read_bytes, io.write_bytes)
            except:
                self.start_io = (0, 0)
                
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_time = time.time()
            
        def sample(self):
            """Call this periodically to update peaks"""
            cpu = self.process.cpu_percent(interval=None)
            mem = self.process.memory_info().rss / 1024 / 1024
            
            if cpu > self.peak_cpu: self.peak_cpu = cpu
            if mem > self.peak_mem: self.peak_mem = mem
            
        def get_stats(self) -> Dict[str, Any]:
            end_mem = self.process.memory_info().rss / 1024 / 1024
            end_cpu = self.process.cpu_percent(interval=None)
            
            try:
                io = self.process.io_counters()
                end_io = (io.read_bytes, io.write_bytes)
            except:
                end_io = (0, 0)
                
            io_read_diff = end_io[0] - self.start_io[0]
            io_write_diff = end_io[1] - self.start_io[1]
            
            return {
                "avg_cpu": end_cpu, # Approximation
                "peak_cpu": max(self.peak_cpu, end_cpu),
                "avg_mem": (self.peak_mem + end_mem) / 2, # Rough approximation
                "peak_mem": max(self.peak_mem, end_mem),
                "io_read_mb": io_read_diff / 1024 / 1024,
                "io_write_mb": io_write_diff / 1024 / 1024
            }
