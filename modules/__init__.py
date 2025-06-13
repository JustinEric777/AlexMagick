from modules.servers.llm_server import LLMServer
from modules.servers.mt_server import MtServer
from modules.servers.asr_server import ASRServer
from modules.servers.tts_server import TTSServer
from modules.servers.text2img_server import Text2ImageServer
from modules.servers.img2img_server import Image2ImageServer
from modules.servers.inpainting_server import InpaintingServer
from modules.servers.text2embedding_server import TextEmbeddingServer
from modules.servers.video2embedding import VideoEmbeddingServer
from modules.servers.video2embedding import VideoEmbeddingServer
from modules.servers.multimodal_llm_server import MultimodalLLMServer
from modules.servers.audio2embedding_server import AudioEmbeddingServer

llm = LLMServer()
mt = MtServer()
asr = ASRServer()
tts = TTSServer()
text2img = Text2ImageServer()
img2img = Image2ImageServer()
inpainting = InpaintingServer()
text2embedding = TextEmbeddingServer()
video2embedding = VideoEmbeddingServer()
multimodal_llm = MultimodalLLMServer()
audio2embedding = AudioEmbeddingServer()
