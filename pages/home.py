import gradio as gr
from servers.search_server import search_server


def create_ui(params: dict):
    with gr.Tab(label="Home", id="home_tab"):
        with gr.Column():
            gr.Markdown("### Task History & Search")

            with gr.Row():
                task_type = gr.Dropdown(
                    choices=["All", "LLM", "Text2Img", "TTS", "ASR"],
                    value="All",
                    label="Task Type"
                )
                keyword = gr.Textbox(label="Keyword (ID/Description)")
                status = gr.Dropdown(
                    choices=["All", "Success", "Failed", "Running"],
                    value="All",
                    label="Status"
                )
                search_btn = gr.Button("Search", variant="primary")

            task_list = gr.Dataframe(
                headers=["ID", "Type", "Description", "Status", "Created At", "Duration"],
                interactive=False,
                label="Task List (Click to view details)"
            )

            with gr.Row():
                with gr.Column(scale=1):
                    task_details = gr.Code(language="json", label="Task Details")
                with gr.Column(scale=1):
                    replay_btn = gr.Button("Replay Selected Task")
                    replay_output = gr.Textbox(label="Replay Result", lines=10)

            # Hidden state to store selected task ID
            selected_task_id = gr.State()

            def on_search(t_type, key, stat):
                return search_server.search_tasks(t_type, key, stat)

            def on_select(evt: gr.SelectData, data):
                # evt.index is [row, col]
                row_index = evt.index[0]
                # ID is in the first column (index 0)
                if row_index < len(data):
                    t_id = data.iloc[row_index, 0]
                    details = search_server.get_task_details(t_id)
                    return t_id, details
                return None, ""

            def on_replay(t_id):
                if not t_id:
                    return "Please select a task first."
                msg, _ = search_server.replay_task(t_id)
                return msg

            search_btn.click(
                on_search,
                inputs=[task_type, keyword, status],
                outputs=[task_list]
            )

            task_list.select(
                on_select,
                inputs=[task_list],
                outputs=[selected_task_id, task_details]
            )

            replay_btn.click(
                on_replay,
                inputs=[selected_task_id],
                outputs=[replay_output]
            )

            # Initial search
            # task_list.load(on_search, inputs=[task_type, keyword, status], outputs=[task_list])
