from modules.servers.llm_server import LLMServer
from modules.servers.mt_server import MtServer
from modules.servers.asr_server import ASRServer
from modules.servers.text2img_server import Text2ImageServer
from modules.servers.img2img_server import Image2ImageServer
from modules.servers.inpainting_server import InpaintingServer

llm = LLMServer()
mt = MtServer()
asr = ASRServer()
text2img = Text2ImageServer()
img2img = Image2ImageServer()
inpainting = InpaintingServer()

