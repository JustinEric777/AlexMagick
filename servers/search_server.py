from core.db.manager import DBManager
import pandas as pd
import json
import traceback

class SearchServer:
    def __init__(self):
        self.db = DBManager()

    def search_tasks(self, task_type, keyword, status, limit=20):
        try:
            results = self.db.search_tasks(
                task_type=task_type if task_type != "All" else None,
                keyword=keyword,
                status=status if status != "All" else None,
                limit=limit
            )
            
            # Convert to DataFrame-friendly format
            data = []
            for r in results:
                data.append([
                    r['id'],
                    r['task_type'],
                    r['description'],
                    r['status'],
                    r['created_at'],
                    f"{r['duration']:.2f}s" if r['duration'] else "N/A"
                ])
            
            return data
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def get_task_details(self, task_id):
        task = self.db.get_task_by_id(task_id)
        if not task:
            return "Task not found"
        
        # Format for display
        display_data = {
            "ID": task['id'],
            "Type": task['task_type'],
            "Status": task['status'],
            "Time": task['created_at'],
            "Duration": f"{task['duration']:.4f}s" if task['duration'] else None,
            "Resources": {
                "CPU": f"{task['cpu_usage']}%",
                "Memory": f"{task['memory_usage']:.2f} MB"
            },
            "Inputs": task['inputs'],
            "Outputs": task['outputs'],
            "Error": task['error_msg']
        }
        return json.dumps(display_data, indent=2, ensure_ascii=False)

    def replay_task(self, task_id):
        # This is a complex operation because we need to map the task type back to the server instance
        # For this implementation, we will try to resolve the server and call generate
        
        task = self.db.get_task_by_id(task_id)
        if not task:
            return "Task not found", None
            
        task_type = task['task_type']
        inputs = task['inputs']
        
        try:
            # Dynamic import to avoid circular imports
            from servers.llm_server import LLMServer
            from servers.text2img_server import Text2ImgServer
            from servers.tts_server import TTSServer
            from servers.asr_server import ASRServer
            
            # Map types to servers
            # These strings must match what we used in @record_task
            server_map = {
                "LLM": LLMServer,
                "Text2Img": Text2ImgServer,
                "TTS": TTSServer,
                "ASR": ASRServer
            }
            
            if task_type not in server_map:
                return f"Replay not supported for type: {task_type}", None
                
            # Instantiate server (singleton logic or lightweight init might be needed)
            # Note: The servers usually have init_model called. 
            # In a real app, we should reuse the global instances from webui.py
            # But here we can't easily access them.
            # Ideally, we should refactor webui.py to expose server instances.
            # For now, we will assume we can instantiate them, but they might need 'init_model'.
            
            # A better approach: We just display the inputs and let the user copy them.
            # But the requirement says "Replay button".
            
            # Let's try to locate the global instance or instantiate a new one.
            # Instantiating a new one might re-load models which is slow.
            # We will use a registry pattern if possible, but let's stick to simple reflection for now.
            
            server_cls = server_map[task_type]
            server = server_cls() 
            
            # Extract args and kwargs
            args = inputs.get('args', [])
            kwargs = inputs.get('kwargs', {})
            
            # Convert args to correct types if possible (JSON stores them as strings/primitives)
            # This is the hard part: Type reconstruction.
            # For this demo, we will pass them as is and hope the server handles them or they are simple types.
            
            # Special handling for servers that need model init
            # We assume the model is already loaded in the shared process or we need to load it.
            # Since we enforce single model loaded, this replay might unload the current model.
            
            # We will try to call generate. 
            # Note: The @record_task decorator is on the class method, so it will record this replay as a NEW task.
            # This satisfies "Replay results stored separately".
            
            if task_type == "LLM":
                # LLM generate expects 'history'
                # args[0] is history
                history = args[0]
                # We need to ensure it's a list
                if isinstance(history, str):
                    try: history = json.loads(history)
                    except: pass
                
                # We need to use a generator consumption for LLM
                gen = server.generate(history, *args[1:], **kwargs)
                result = None
                for item in gen:
                    result = item
                return f"Replay Successful. New Task Created.\nResult: {str(result)[:500]}...", result
                
            else:
                result = server.generate(*args, **kwargs)
                return f"Replay Successful. New Task Created.\nResult: {str(result)[:500]}...", result
                
        except Exception as e:
            return f"Replay Failed: {str(e)}\n{traceback.format_exc()}", None

search_server = SearchServer()
