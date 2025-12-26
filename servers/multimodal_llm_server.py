from config.multimodal_llm_config import MODEL_LIST, TASK_TYPE
from servers.base_server import BaseServer
from core.models.engine import ModelEngine
from core.models.adapters import resolve_model_config
from core.task_monitor import record_task


def remove_metric(history):
    messages = [one_message.copy() for one_message in history]
    for line in messages:
        if line["role"] == "assistant" and type(line["content"]) is str and line["content"] is not None:
            arr = line["content"].split("\n")
            line["content"] = arr[0]

    return messages


class MultimodalLLMServer(BaseServer):
    def __init__(self):
        super().__init__()
        self.model_list = MODEL_LIST
        self.task_type = TASK_TYPE
        self.engine = None

    def init_model(self, params: dict):
        arch = params.get("infer_arch", "")
        device = params.get("device", "")
        model_name = params.get("model_name", "")
        version = params.get("model_version", "")

        if params.get("task_type") != self.task_type or arch not in self.model_list or model_name not in self.model_list[arch]:
            arch_list = self.get_infer_arch_list()
            arch = arch_list[0] if len(arch_list) > 0 else ""
            
            device_list = self.get_arch_device_list(infer_arch=arch)
            device = device_list[0] if len(device_list) > 0 else ""
            
            model_name_list = self.get_arch_model_list(infer_arch=arch)
            model_name = model_name_list[0] if len(model_name_list) > 0 else ""
            
            version_list = self.get_model_list(infer_arch=arch, model_name=model_name)
            version = version_list[0] if len(version_list) > 0 else ""

        model_path = self.model_list[arch][model_name]["model_path"]
        backend, impl = resolve_model_config("multimodal_llm", arch, model_name)
        self.engine = ModelEngine(
            model_type="multimodal_llm",
            model_name_or_path=f"{model_path}/{version}",
            device=device,
            backend=backend,
            impl=impl,
        )
        self.pipeline = self.engine.model
        self.infer_arch = arch
        self.device = device
        self.model_name = model_name
        self.model_version_name = version

    def reload_model(self, infer_arch: str = "", device: str = "", model_name: str = "", model_version: str = "", default: bool = False):
        if default:
            arch_list = self.get_infer_arch_list()
            infer_arch = arch_list[0] if len(arch_list) > 0 else ""
            device_list = self.get_arch_device_list(infer_arch)
            device = device_list[0] if len(device_list) > 0 else ""
            model_name_list = self.get_arch_model_list(infer_arch)
            model_name = model_name_list[0] if len(model_name_list) > 0 else ""
            model_version_list = self.get_model_list(infer_arch, model_name)
            model_version = model_version_list[0] if len(model_version_list) > 0 else ""
        
        if not infer_arch or infer_arch not in self.model_list:
             print("Warning: No valid model configuration found for Multimodal LLM. Skipping reload.")
             return infer_arch, device, model_name, model_version

        model_path = self.model_list[infer_arch][model_name]["model_path"]
        backend, impl = resolve_model_config("multimodal_llm", infer_arch, model_name)
        self.engine = ModelEngine(
            model_type="multimodal_llm",
            model_name_or_path=f"{model_path}/{model_version}",
            device=device,
            backend=backend,
            impl=impl,
        )
        self.pipeline = self.engine.model
        self.infer_arch = infer_arch
        self.device = device
        self.model_name = model_name
        self.model_version_name = model_version
        return infer_arch, device, model_name, model_version

    @record_task("MultimodalLLM", "Multimodal Generation")
    def generate(self, history, max_tokens, temperature, top_p, slider_context_times, return_audio):
        # history comes in as Server Format (content can be dict {"path": ...})
        
        # remove_metric makes a deep copy of the list and dicts, so 'messages' is safe to modify/use
        messages = remove_metric(history)
        
        # We need a yield_history that mirrors messages but tracks the new generation
        # Since remove_metric cleans up the history (removes old metrics), we can use it as base.
        yield_history = [m.copy() for m in messages]
        yield_history.append({"role": "assistant", "content": ""})
        
        for message, cost_time, words_count, single_word_cost_time, output_tokens_count, per_second_tokens in self.pipeline.chat(
                messages,
                max_tokens,
                temperature,
                top_p,
                slider_context_times,
                return_audio
        ):
            if "type" in message and message["type"] == "audio":
                # Server Format for audio output
                yield_history.append({"role": "assistant", "content": {"path": message["data"]}})
            else:
                yield_history[-1]["content"] = message
                if cost_time != 0 and words_count != 0 and single_word_cost_time != 0:
                    yield_history[-1]["content"] += self.get_metric(self.infer_arch, self.device, self.model_name,
                                                              self.model_version_name, cost_time,
                                                              words_count, single_word_cost_time, output_tokens_count, per_second_tokens)

            yield yield_history

    def get_metric(self, infer_arch: str, device: str, model_name: str, model_version: str, cost_time: float,
                   words_count: int,
                   single_word_cost_time: float,
                   output_tokens_count: int,
                   per_second_tokens: float, ) -> str:
        return f"""
                <span style="color: red; display:block; float:right; margin-right: 10px">
                infer_arch：{infer_arch}
                device：{device}
                model_name：{model_name}
                model_version：{model_version}
                cost_time：{cost_time} 
                words_count：{words_count} 
                per_second_tokens：{per_second_tokens} tokens / s
                output_tokens_count：{output_tokens_count} 
                single_word_cost_time：{single_word_cost_time}</span>
                """

