
import gradio as gr
import os
import requests
import sys
from openai import OpenAI

# Check for API key
if not os.getenv("DASHSCOPE_API_KEY"):
    print("Error: DASHSCOPE_API_KEY environment variable not set.")
    print("Please set the environment variable and try again.")
    sys.exit(1)

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def file_reader(file):
    if file is not None:
        with open(file.name, 'r', encoding='utf-8') as f:
            content = f.read()
        return {"filename": file.name, "content": content}
    return None

def url_parser(url):
    if url:
        try:
            response = requests.get(f"https://r.jina.ai/{url}")
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.text
        except requests.exceptions.RequestException as e:
            return f"Error parsing URL: {e}"
    return None

def translate_text(text, target_language):
    if not text:
        return "", []

    messages = [
        {
            "role": "system",
            "content": "You are a translation expert, proficient in multiple languages."
        },
        {
            "role": "user",
            "content": f"Translate the following text to {target_language}: {text}"
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model="qwen3-30b-a3b-thinking-2507",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during translation: {e}"

def file_uploaded(file):
    return gr.Markdown("File uploaded successfully!", visible=True)

with gr.Blocks(css="""
    .progress-bar-container {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 30px;
    }
    .progress-bar {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        height: 10px;
        border-radius: 5px;
        width: 100%;
    }
    .progress-text {
        margin-left: 10px;
        color: #6b7280;
    }
    @keyframes gradient {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
    .copy-button {
        position: absolute;
        top: 5px;
        right: 5px;
        cursor: pointer;
        font-size: 20px;
        color: #6b7280;
    }
    .output-container {
        position: relative;
    }
""") as demo:
    gr.Markdown("# AI-Powered Translation Tool")
    
    with gr.Tabs():
        with gr.TabItem("Text"):
            text_input = gr.Textbox(lines=5, label="Input Text")
            progress_bar_text = gr.HTML("""
                <div class='progress-bar-container'>
                    <div class='progress-bar'></div>
                    <div class='progress-text'>正在翻译中...</div>
                </div>
            """, visible=False)
            text_button = gr.Button("Translate")
            with gr.Column(elem_classes=["output-container"]):
                text_output = gr.Textbox(label="Translation", interactive=False)
                copy_button_text = gr.Button("📋", elem_classes=["copy-button"], visible=False)
        with gr.TabItem("File"):
            file_input = gr.File(label="Upload File (.md, .txt, .html)")
            file_status = gr.Markdown(visible=False)
            progress_bar_file = gr.HTML("""
                <div class='progress-bar-container'>
                    <div class='progress-bar'></div>
                    <div class='progress-text'>正在翻译中...</div>
                </div>
            """, visible=False)
            file_button = gr.Button("Translate")
            with gr.Column(elem_classes=["output-container"]):
                file_output = gr.Textbox(label="Translation", interactive=False)
                copy_button_file = gr.Button("📋", elem_classes=["copy-button"], visible=False)
        with gr.TabItem("URL"):
            url_input = gr.Textbox(label="Enter URL")
            progress_bar_url = gr.HTML("""
                <div class='progress-bar-container'>
                    <div class='progress-bar'></div>
                    <div class='progress-text'>正在翻译中...</div>
                </div>
            """, visible=False)
            url_button = gr.Button("Translate")
            with gr.Column(elem_classes=["output-container"]):
                url_output = gr.Textbox(label="Translation", interactive=False)
                copy_button_url = gr.Button("📋", elem_classes=["copy-button"], visible=False)

    target_language = gr.Radio(["Chinese", "English"], label="Target Language", value="Chinese")

    def translate_interface_with_progress(text_input, file_input, url_input, target_language):
        yield gr.update(visible=True), "", gr.update(visible=False)
        
        try:
            if text_input:
                result = translate_text(text_input, target_language)
            elif file_input is not None:
                file_data = file_reader(file_input)
                if file_data:
                    result = translate_text(file_data["content"], target_language)
                else:
                    result = "Could not read file."
            elif url_input:
                parsed_content = url_parser(url_input)
                if parsed_content:
                    result = translate_text(parsed_content, target_language)
                else:
                    result = "Could not parse URL."
            else:
                result = "Please provide input for translation."
            
            yield gr.update(visible=False), result, gr.update(visible=True)
        except Exception as e:
            yield gr.update(visible=False), f"Translation failed: {e}", gr.update(visible=False)

    text_button.click(
        translate_interface_with_progress, 
        inputs=[text_input, gr.State(None), gr.State(None), target_language], 
        outputs=[progress_bar_text, text_output, copy_button_text]
    )
    
    file_input.upload(file_uploaded, inputs=file_input, outputs=file_status)
    file_button.click(
        translate_interface_with_progress, 
        inputs=[gr.State(None), file_input, gr.State(None), target_language], 
        outputs=[progress_bar_file, file_output, copy_button_file]
    )

    url_button.click(
        translate_interface_with_progress, 
        inputs=[gr.State(None), gr.State(None), url_input, target_language], 
        outputs=[progress_bar_url, url_output, copy_button_url]
    )

    def copy_to_clipboard(text):
        pass

    copy_button_text.click(copy_to_clipboard, inputs=text_output, outputs=None, js="(text) => navigator.clipboard.writeText(text)")
    copy_button_file.click(copy_to_clipboard, inputs=file_output, outputs=None, js="(text) => navigator.clipboard.writeText(text)")
    copy_button_url.click(copy_to_clipboard, inputs=url_output, outputs=None, js="(text) => navigator.clipboard.writeText(text)")

if __name__ == "__main__":
    demo.launch()
