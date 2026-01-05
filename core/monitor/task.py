import functools
import traceback
import inspect
import uuid
from typing import Generator
from core.db.manager import DBManager
from .system import SystemMonitor

class TaskMonitor:
    def __init__(self, task_type: str, description: str = ""):
        self.task_type = task_type
        self.description = description
        self.db = DBManager()

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate Task ID
            task_id = str(uuid.uuid4())
            
            # Prepare inputs (convert to serializable format)
            # Filter out 'self' if it's a method
            clean_args = [str(a) for a in args[1:]] if args and hasattr(args[0], '__class__') else [str(a) for a in args]
            inputs = {
                "args": clean_args,
                "kwargs": {k: str(v) for k, v in kwargs.items()}
            }
            
            # Start Tracking
            tracker = SystemMonitor.Tracker()
            tracker.__enter__()
            
            # Insert Initial Record
            self.db.insert_task({
                "id": task_id,
                "task_type": self.task_type,
                "description": self.description,
                "inputs": inputs,
                "outputs": {},
                "status": "Running",
                "start_time": tracker.start_time,
                "cpu_usage": 0,
                "memory_usage": tracker.peak_mem
            })
            
            def record_success(output_val):
                tracker.__exit__(None, None, None)
                stats = tracker.get_stats()
                
                output_str = str(output_val)
                # Truncate very long output for DB storage
                output_data = {"result": output_str[:5000] + "..." if len(output_str) > 5000 else output_str}
                
                self.db.update_task(task_id, {
                    "outputs": output_data,
                    "status": "Success",
                    "end_time": tracker.end_time,
                    "duration": tracker.end_time - tracker.start_time,
                    "cpu_usage": stats["peak_cpu"],
                    "memory_usage": stats["peak_mem"]
                })

            def record_failure(error_val):
                tracker.__exit__(Exception, error_val, None)
                stats = tracker.get_stats()
                
                self.db.update_task(task_id, {
                    "status": "Failed",
                    "end_time": tracker.end_time,
                    "duration": tracker.end_time - tracker.start_time,
                    "cpu_usage": stats["peak_cpu"],
                    "memory_usage": stats["peak_mem"],
                    "error_msg": str(error_val) + "\n" + traceback.format_exc()
                })
            
            try:
                result = func(*args, **kwargs)
                
                if inspect.isgenerator(result):
                    def generator_wrapper():
                        last_val = None
                        try:
                            for item in result:
                                last_val = item
                                # Optional: Sample periodically during generation
                                tracker.sample() 
                                yield item
                            record_success(last_val)
                        except Exception as e:
                            record_failure(e)
                            raise e
                    return generator_wrapper()
                else:
                    record_success(result)
                    return result
                
            except Exception as e:
                record_failure(e)
                raise e
                
        return wrapper

def record_task(task_type: str, description: str = ""):
    return TaskMonitor(task_type, description)
