import sys
import os
sub_module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/Matcha-TTS")
if sub_module_path not in sys.path:
    sys.path.insert(0, sub_module_path)
