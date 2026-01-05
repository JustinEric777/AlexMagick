import gradio as gr
from pages.common import create_standard_page
from servers import llm_server

# Initialize server instance instead of importing module directly
llm = llm_server.LLMServer()


def create_ui(args: dict):
    components = {}

    def render_main():
        components['chatbot'] = gr.Chatbot(height=500)
        components['msg'] = gr.Textbox(label="Chatbot Input", lines=5, placeholder="Shift + Enter Send Message...", )
        with gr.Row():
            gr.Examples(
                label="Input Examples",
                examples=[
                    [
                        "左手一只鸭，右手一只鸡。交换两次后左右手里各是什么？",
                    ],
                    [
                        "鸡兔同笼，共35只头，94只脚，问鸡兔各多少？",
                    ],
                    [
                        "Strawberry里有几个r？",
                    ],
                    [
                        "一只死猫与核同位素、一瓶毒药和辐射探测器一起放入盒子中。如果辐射探测器检测到辐射，它将释放毒药。一天后，盒子打开。猫还活着吗？",
                    ],
                    [
                        "假设在一个电车轨道上被绑了5个人，而它的备用轨道上被绑了1个人，又有一辆失控的电车飞速驶来，而你身边正好有一个摇杆，你可以推动摇杆来让电车驶入备用轨道。你该怎么做起伤害最小？",
                    ],
                    [
                        "为什么会有这样的语言现象:生鱼片是死鱼片，等红灯是在等绿灯，咖啡因来自咖啡果，救火是在灭火，晒太阳是在晒人，肉夹馍是馍夹肉"
                    ]
                ],
                inputs=[components['msg']],
            )
        with gr.Row():
            components['clear'] = gr.Button("New Topic")
            components['re_generate'] = gr.Button("Regenerate")
            components['sent_bt'] = gr.Button("Send", variant="primary")
    
    def render_params():
        with gr.Accordion("generate params", open=True):
            components['max_tokens'] = gr.Slider(minimum=512, maximum=4096, label="max_tokens", value=2048)
            components['slider_temp'] = gr.Slider(minimum=0, maximum=1, label="temperature", value=0.6)
            components['slider_top_p'] = gr.Slider(minimum=0.5, maximum=1, label="top_p", value=0.95)
            components['slider_context_times'] = gr.Slider(minimum=0, maximum=5, label="context times", value=0, step=2.0)
        with gr.Accordion("History", open=True):
            with gr.Row():
                components['page'] = gr.Number(label="page", value=1, precision=0)
                components['page_size'] = gr.Number(label="page size", value=10, precision=0)
                components['refresh'] = gr.Button("Refresh")
            components['history_list'] = gr.Dataframe(headers=["timestamp", "model", "arch", "device", "preview"], datatype=["number", "str", "str", "str", "str"], interactive=False)

    create_standard_page(llm, args, render_main, render_params_ui=render_params, tab_label="LLM Model", tab_id="llm_tab")

    # Bind Events
    def fetch_history(page, page_size):
        limit = max(1, int(page_size))
        page = max(1, int(page))
        items_all = llm.history.list_recent(limit=limit * page)
        start = (page - 1) * limit
        items = items_all[start:start + limit]
        rows = []
        for it in items:
            md = it["metadata"]
            rows.append([it["timestamp"], md.get("model_name", ""), md.get("infer_arch", ""), md.get("device", ""), (it["content"] or "")[:120]])
        return rows
    
    components['refresh'].click(fetch_history, [components['page'], components['page_size']], [components['history_list']], queue=True)

    def user(user_message, history):
        if history is None:
            history = []
        return "", history + [[user_message, None]]

    def generate_wrapper(history, max_tokens, temperature, top_p, slider_context_times):
        # history is in [[user, bot], ...] format
        # Convert to list of dicts for LLM processing
        messages = []
        for u, b in history:
            if u: messages.append({"role": "user", "content": u})
            if b: messages.append({"role": "assistant", "content": b})
        
        # Call original generator
        for updated_messages in llm.generate(messages, max_tokens, temperature, top_p, slider_context_times):
            new_history = []
            current_pair = [None, None]
            
            for msg in updated_messages:
                if msg['role'] == 'user':
                    if current_pair[0] is not None:
                            new_history.append(current_pair)
                            current_pair = [None, None]
                    current_pair[0] = msg['content']
                elif msg['role'] == 'assistant':
                    current_pair[1] = msg['content']
                    new_history.append(current_pair)
                    current_pair = [None, None]
            
            if current_pair[0] is not None or current_pair[1] is not None:
                new_history.append(current_pair)
                
            yield new_history

    components['msg'].submit(user, [components['msg'], components['chatbot']], [components['msg'], components['chatbot']], queue=True).then(
        generate_wrapper,
        [components['chatbot'], components['max_tokens'], components['slider_temp'], components['slider_top_p'], components['slider_context_times']],
        components['chatbot'],
        queue=True
    )
    components['sent_bt'].click(user, [components['msg'], components['chatbot']], [components['msg'], components['chatbot']], queue=True).then(
        generate_wrapper,
        [components['chatbot'], components['max_tokens'], components['slider_temp'], components['slider_top_p'], components['slider_context_times']],
        components['chatbot'],
        queue=True
    )
    components['re_generate'].click(
        generate_wrapper,
        [components['chatbot'], components['max_tokens'], components['slider_temp'], components['slider_top_p'], components['slider_context_times']],
        components['chatbot'],
        queue=True
    )
    components['clear'].click(lambda: [], None, components['chatbot'], queue=True)
