import time
from typing import Dict, Any, Union, List

class InferenceMonitor:
    """Monitor inference performance metrics"""
    
    @staticmethod
    def calc_llm_metrics(
        start_time: float, 
        end_time: float, 
        output_text: str, 
        input_text: str = ""
    ) -> Dict[str, Any]:
        duration = end_time - start_time
        if duration <= 0: duration = 0.001
        
        # Word count (approximation)
        words_count = len(output_text)
        
        # Metrics
        tokens_per_sec = words_count / duration # Char per second effectively if chinese, or use tokenizer if available
        sec_per_token = duration / max(1, words_count)
        
        return {
            "duration": round(duration, 3),
            "total_words": words_count,
            "tokens_per_sec": round(tokens_per_sec, 2),
            "sec_per_token": round(sec_per_token, 4),
        }

    @staticmethod
    def calc_audio_metrics(
        start_time: float, 
        end_time: float, 
        audio_duration: float
    ) -> Dict[str, Any]:
        cost = end_time - start_time
        if cost <= 0: cost = 0.001
        
        rtf = cost / audio_duration if audio_duration > 0 else 0
        
        return {
            "duration": round(cost, 3),
            "audio_duration": round(audio_duration, 3),
            "rtf": round(rtf, 3) # Real Time Factor
        }

    @staticmethod
    def calc_image_metrics(
        start_time: float,
        end_time: float,
        num_images: int = 1
    ) -> Dict[str, Any]:
        duration = end_time - start_time
        if duration <= 0: duration = 0.001
        
        return {
            "duration": round(duration, 3),
            "sec_per_image": round(duration / num_images, 3)
        }
