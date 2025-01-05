# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

### This module contains the chatui gui for having a conversation. ###

import functools
from typing import Any, Dict, List, Tuple, Union

import gradio as gr
import shutil
import os
import subprocess
import time
import sys

from chatui import assets, chat_client
from chatui.prompts import prompts_llama3, prompts_mistral, defaults
from chatui.utils import compile, database, logger

from langgraph.graph import END, StateGraph

PATH = "/"
TITLE = "NVIDIA AI Workbench Virtual Assistant"
OUTPUT_TOKENS = 250
MAX_DOCS = 5

### Load in CSS here for components that need custom styling. ###

_LOCAL_CSS = """
#contextbox {
    overflow-y: scroll !important;
    max-height: 400px;
}

#params .tabs {
    display: flex;
    flex-direction: column;
    flex-grow: 1;
}
#params .tabitem[style="display: block;"] {
    flex-grow: 1;
    display: flex !important;
}
#params .gap {
    flex-grow: 1;
}
#params .form {
    flex-grow: 1 !important;
}
#params .form > :last-child{
    flex-grow: 1;
}
#accordion {
}
#rag-inputs .svelte-1gfkn6j {
    color: #76b900;
}
#rag-inputs .svelte-s1r2yt {
    color: #76b900;
}
#router_button {
    background-color: #F5CCCB !important;
    color: red !important;
}
#retrieval_button {
    background-color: #B3D6FB !important;
    color: blue !important;
}
#generator_button {
    background-color: #B3D6FB !important;
    color: blue !important;
}
#hallucination_button {
    background-color: #BFEFBF !important;
    color: green !important;
}
#answer_button {
    background-color: #BFEFBF !important;
    color: green !important;
}
"""

sys.stdout = logger.Logger("/project/code/output.log")

def build_page(client: chat_client.ChatClient) -> gr.Blocks:
    """
    Build the gradio page to be mounted in the frame.
    
    Parameters: 
        client (chat_client.ChatClient): The chat client running the application. 
    
    Returns:
        page (gr.Blocks): A Gradio page.
    """
    kui_theme, kui_styles = assets.load_theme("kaizen")
    
    """ Compile the agentic graph. """
    
    workflow = compile.compile_graph()
    app = workflow.compile()

    """ List of currently supported models. """
    
    model_list = ["meta/llama3-70b-instruct",
                  "meta/llama-3.1-405b-instruct",
                  "mistralai/mixtral-8x22b-instruct-v0.1"]

    with gr.Blocks(title=TITLE, theme=kui_theme, css=kui_styles + _LOCAL_CSS) as page:
        gr.Markdown(f"# {TITLE}")

        """ Keep state of which queries need to use NIMs vs API Endpoints. """
        
        router_use_nim = gr.State(False)
        retrieval_use_nim = gr.State(False)
        generator_use_nim = gr.State(False)
        hallucination_use_nim = gr.State(False)
        answer_use_nim = gr.State(False)
        initialized = gr.State(False)

        """ Build the Chat Application. """
        
        with gr.Row(equal_height=True):

            # Left Column will display the chatbot
            with gr.Column(scale=15, min_width=350):

                # Main chatbot panel. 
                with gr.Row(equal_height=True):
                    with gr.Column(min_width=350):
                        chatbot = gr.Chatbot(height=480, show_label=False)

                # Message box for user input
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3, min_width=450):
                        msg = gr.Textbox(
                            show_label=False,
                            placeholder="Enter text and press ENTER",
                            container=False,
                            interactive=False,
                        )

                    with gr.Column(scale=1, min_width=150):
                        _ = gr.ClearButton([msg, chatbot], value="Clear history")

                # Sample questions that users can click on to use
                with gr.Row(equal_height=True):
                    sample_query_1 = gr.Button("What OS versions are supported by AI Workbench?", variant="secondary", size="sm", interactive=False)
                    sample_query_2 = gr.Button("How do I get started with AI Workbench?", variant="secondary", size="sm", interactive=False)
                
                with gr.Row(equal_height=True):
                    sample_query_3 = gr.Button("How do I fix an inaccessible remote Location?", variant="secondary", size="sm", interactive=False)
                    sample_query_4 = gr.Button("How do I create a support bundle in AI Workbench CLI?", variant="secondary", size="sm", interactive=False)
            
            # Hidden column to be rendered when the user collapses all settings.
            with gr.Column(scale=1, min_width=100, visible=False) as hidden_settings_column:
                show_settings = gr.Button(value="< Expand", size="sm")
            
            # Right column to display all relevant settings
            with gr.Column(scale=10, min_width=350) as settings_column:
                
                with gr.Row():
                    agentic_flow = gr.Image("/project/code/chatui/static/agentic-flow.png", 
                                            show_label=False,
                                            show_download_button=False,
                                            interactive=False)
                
                with gr.Tabs(selected=0) as settings_tabs:

                    # Settings to initialize the app
                    with gr.TabItem("Welcome", id=0, interactive=True, visible=True) as welcome_settings:
                        gr.Markdown("<br />Welcome to your virtual assistant! I can help with all kinds of questions related to NVIDIA AI Workbench. Use me to troubleshoot, get started with, or to simply learn more about AI Workbench. ")
                        gr.Markdown("Press the button below to initialize the application. Happy chatting! ")
                        initialize = gr.Button("Initialize Virtual Assistant", variant="primary")
                        gr.Markdown("<br />")
                    
                    # Settings for each component model of the workflow
                    with gr.TabItem("Models", id=1, interactive=False, visible=False) as agent_settings:
    
                        ########################
                        ##### ROUTER MODEL #####
                        ########################
                        router_btn = gr.Button("Router", variant="sm", elem_id="router_button")
                        with gr.Group(visible=False) as group_router:
                            with gr.Tabs(selected=0) as router_tabs:
                                with gr.TabItem("API Endpoints", id=0) as router_api:
                                    model_router = gr.Dropdown(model_list, 
                                                               value=model_list[0],
                                                               label="Select a Model",
                                                               elem_id="rag-inputs", 
                                                               interactive=True)
                                    
                                with gr.TabItem("NIM Endpoints", id=1) as router_nim:
                                    with gr.Row():
                                        nim_router_ip = gr.Textbox(placeholder = "10.123.45.678", 
                                                   label = "Microservice Host", 
                                                   info = "IP Address running the microservice", 
                                                   elem_id="rag-inputs", scale=2)
                                        nim_router_port = gr.Textbox(placeholder = "8000", 
                                                   label = "Port", 
                                                   info = "Optional, (default: 8000)", 
                                                   elem_id="rag-inputs", scale=1)
                                    
                                    nim_router_id = gr.Textbox(placeholder = "meta/llama3-8b-instruct", 
                                               label = "Model running in microservice.", 
                                               info = "If none specified, defaults to: meta/llama3-8b-instruct", 
                                               elem_id="rag-inputs")

                                with gr.TabItem("Hide", id=2) as router_hide:
                                    gr.Markdown("")

                            with gr.Accordion("Configure the Router Prompt", 
                                              elem_id="rag-inputs", open=False) as accordion_router:
                                prompt_router = gr.Textbox(value=prompts_llama3.router_prompt,
                                                           lines=12,
                                                           show_label=False,
                                                           interactive=True)
    
                        ##################################
                        ##### RETRIEVAL GRADER MODEL #####
                        ##################################
                        retrieval_btn = gr.Button("Retrieval Grader", variant="sm", elem_id="retrieval_button")
                        with gr.Group(visible=False) as group_retrieval:
                            with gr.Tabs(selected=0) as retrieval_tabs:
                                with gr.TabItem("API Endpoints", id=0) as retrieval_api:
                                    model_retrieval = gr.Dropdown(model_list, 
                                                                         value=model_list[0],
                                                                         label="Select a Model",
                                                                         elem_id="rag-inputs", 
                                                                         interactive=True)
                                with gr.TabItem("NIM Endpoints", id=1) as retrieval_nim:
                                    with gr.Row():
                                        nim_retrieval_ip = gr.Textbox(placeholder = "10.123.45.678", 
                                                   label = "Microservice Host", 
                                                   info = "IP Address running the microservice", 
                                                   elem_id="rag-inputs", scale=2)
                                        nim_retrieval_port = gr.Textbox(placeholder = "8000", 
                                                   label = "Port", 
                                                   info = "Optional, (default: 8000)", 
                                                   elem_id="rag-inputs", scale=1)
                                    
                                    nim_retrieval_id = gr.Textbox(placeholder = "meta/llama3-8b-instruct", 
                                               label = "Model running in microservice.", 
                                               info = "If none specified, defaults to: meta/llama3-8b-instruct", 
                                               elem_id="rag-inputs")

                                with gr.TabItem("Hide", id=2) as retrieval_hide:
                                    gr.Markdown("")
                            
                            with gr.Accordion("Configure the Retrieval Grader Prompt", 
                                              elem_id="rag-inputs", open=False) as accordion_retrieval:
                                prompt_retrieval = gr.Textbox(value=prompts_llama3.retrieval_prompt,
                                                                     lines=21,
                                                                     show_label=False,
                                                                     interactive=True)
    
                        ###########################
                        ##### GENERATOR MODEL #####
                        ###########################
                        generator_btn = gr.Button("Generator", variant="sm", elem_id="generator_button")
                        with gr.Group(visible=False) as group_generator:
                            with gr.Tabs(selected=0) as generator_tabs:
                                with gr.TabItem("API Endpoints", id=0) as generator_api:
                                    model_generator = gr.Dropdown(model_list, 
                                                                  value=model_list[0],
                                                                  label="Select a Model",
                                                                  elem_id="rag-inputs", 
                                                                  interactive=True)
                                with gr.TabItem("NIM Endpoints", id=1) as generator_nim:
                                    with gr.Row():
                                        nim_generator_ip = gr.Textbox(placeholder = "10.123.45.678", 
                                                   label = "Microservice Host", 
                                                   info = "IP Address running the microservice", 
                                                   elem_id="rag-inputs", scale=2)
                                        nim_generator_port = gr.Textbox(placeholder = "8000", 
                                                   label = "Port", 
                                                   info = "Optional, (default: 8000)", 
                                                   elem_id="rag-inputs", scale=1)
                                    
                                    nim_generator_id = gr.Textbox(placeholder = "meta/llama3-8b-instruct", 
                                               label = "Model running in microservice.", 
                                               info = "If none specified, defaults to: meta/llama3-8b-instruct", 
                                               elem_id="rag-inputs")

                                with gr.TabItem("Hide", id=2) as generator_hide:
                                    gr.Markdown("")
                            
                            with gr.Accordion("Configure the Generator Prompt", 
                                              elem_id="rag-inputs", open=False) as accordion_generator:
                                prompt_generator = gr.Textbox(value=prompts_llama3.generator_prompt,
                                                          lines=15,
                                                          show_label=False,
                                                          interactive=True)
    
                        ######################################
                        ##### HALLUCINATION GRADER MODEL #####
                        ######################################
                        hallucination_btn = gr.Button("Hallucination Grader", variant="sm", elem_id="hallucination_button")
                        with gr.Group(visible=False) as group_hallucination:
                            with gr.Tabs(selected=0) as hallucination_tabs:
                                with gr.TabItem("API Endpoints", id=0) as hallucination_api:
                                    model_hallucination = gr.Dropdown(model_list, 
                                                                             value=model_list[0],
                                                                             label="Select a Model",
                                                                             elem_id="rag-inputs", 
                                                                             interactive=True)
                                with gr.TabItem("NIM Endpoints", id=1) as hallucination_nim:
                                    with gr.Row():
                                        nim_hallucination_ip = gr.Textbox(placeholder = "10.123.45.678", 
                                                   label = "Microservice Host", 
                                                   info = "IP Address running the microservice", 
                                                   elem_id="rag-inputs", scale=2)
                                        nim_hallucination_port = gr.Textbox(placeholder = "8000", 
                                                   label = "Port", 
                                                   info = "Optional, (default: 8000)", 
                                                   elem_id="rag-inputs", scale=1)
                                    
                                    nim_hallucination_id = gr.Textbox(placeholder = "meta/llama3-8b-instruct", 
                                               label = "Model running in microservice.", 
                                               info = "If none specified, defaults to: meta/llama3-8b-instruct", 
                                               elem_id="rag-inputs")

                                with gr.TabItem("Hide", id=2) as hallucination_hide:
                                    gr.Markdown("")
                            
                            with gr.Accordion("Configure the Hallucination Prompt", 
                                              elem_id="rag-inputs", open=False) as accordion_hallucination:
                                prompt_hallucination = gr.Textbox(value=prompts_llama3.hallucination_prompt,
                                                                         lines=17,
                                                                         show_label=False,
                                                                         interactive=True)
    
                        ###############################
                        ##### ANSWER GRADER MODEL #####
                        ###############################
                        answer_btn = gr.Button("Answer Grader", variant="sm", elem_id="answer_button")
                        with gr.Group(visible=False) as group_answer:
                            with gr.Tabs(selected=0) as answer_tabs:
                                with gr.TabItem("API Endpoints", id=0) as answer_api:
                                    model_answer = gr.Dropdown(model_list, 
                                                                      value=model_list[0],
                                                                      elem_id="rag-inputs",
                                                                      label="Select a Model",
                                                                      interactive=True)
                                with gr.TabItem("NIM Endpoints", id=1) as answer_nim:
                                    with gr.Row():
                                        nim_answer_ip = gr.Textbox(placeholder = "10.123.45.678", 
                                                   label = "Microservice Host", 
                                                   info = "IP Address running the microservice", 
                                                   elem_id="rag-inputs", scale=2)
                                        nim_answer_port = gr.Textbox(placeholder = "8000", 
                                                   label = "Port", 
                                                   info = "Optional, (default: 8000)", 
                                                   elem_id="rag-inputs", scale=1)
                                    
                                    nim_answer_id = gr.Textbox(placeholder = "meta/llama3-8b-instruct", 
                                               label = "Model running in microservice.", 
                                               info = "If none specified, defaults to: meta/llama3-8b-instruct", 
                                               elem_id="rag-inputs")

                                with gr.TabItem("Hide", id=2) as answer_hide:
                                    gr.Markdown("")
                                    
                            with gr.Accordion("Configure the Answer Prompt", 
                                              elem_id="rag-inputs", open=False) as accordion_answer:
                                prompt_answer = gr.Textbox(value=prompts_llama3.answer_prompt,
                                                                  lines=17,
                                                                  show_label=False,
                                                                  interactive=True)
                        
                    # Second tab item is for uploading to and clearing the vector database
                    with gr.TabItem("Database", id=2, interactive=False, visible=False) as document_settings:
                        gr.Markdown("")
                        gr.Markdown("Upload webpages, pdfs, images, and videos to the vector store.<br />**Note:** Clearing docs will empty the database!\n")
                        with gr.Tabs(selected=0) as document_tabs:
                            with gr.TabItem("Webpages", id=0) as url_tab:
                                web_url_btn = gr.Button("Webpage URLs", variant="sm")
                                with gr.Group(visible=False) as group_web_url:
                                    url_docs = gr.Textbox(value=defaults.webpage_url_defaults,
                                                      lines=5, 
                                                      info="Enter a list of URLs, one per line", 
                                                      label="Webpage URLs", 
                                                      elem_id="rag-inputs",
                                                      interactive=True)
                                
                                    with gr.Row():
                                        url_docs_upload = gr.Button(value="Upload Docs")
                                        url_docs_clear = gr.Button(value="Clear Docs")

                                video_url_btn = gr.Button("YouTube URLs", variant="sm")
                                with gr.Group(visible=False) as group_video_url:
                                    url_videos = gr.Textbox(value=defaults.video_url_defaults,
                                                          lines=5, 
                                                          info="Enter a list of URLs, one per line", 
                                                          label="Youtube URLs", 
                                                          elem_id="rag-inputs",
                                                          interactive=True)
                                    with gr.Row():
                                        url_videos_upload = gr.Button(value="Upload Videos")
                                        url_videos_clear = gr.Button(value="Clear Docs")
                            
                            with gr.TabItem("File Upload", id=1) as pdf_tab:
                                pdf_upload_btn = gr.Button("PDF Upload", variant="sm")
                                with gr.Group(visible=False) as group_pdf_upload:
                                    pdf_docs_upload = gr.File(interactive=True, 
                                                              label="PDFs", 
                                                              file_types=[".pdf"], 
                                                              file_count="multiple")
                                image_upload_btn = gr.Button("Image Upload", variant="sm")
                                with gr.Group(visible=False) as group_image_upload:
                                    image_upload = gr.File(interactive=True, 
                                                          label="Images", 
                                                          file_types=["image"], 
                                                          file_count="multiple")
                                video_upload_btn = gr.Button("Video Upload", variant="sm")
                                with gr.Group(visible=False) as group_video_upload:
                                    video_upload = gr.File(interactive=True, 
                                                          label="Videos", 
                                                          file_types=["video"], 
                                                          file_count="multiple")
                                file_upload_clear = gr.Button(value="Clear Docs")
    
                    # Third tab item is for the actions output console. 
                    with gr.TabItem("Monitor", id=3, interactive=False, visible=False) as console_settings:
                        gr.Markdown("")
                        gr.Markdown("Monitor pipeline actions and view the trace of the latest response.\n")
                        with gr.Tabs(selected=0) as console_tabs:
                            with gr.TabItem("Actions Console", id=0) as actions_tab:
                                logs = gr.Textbox(show_label=False, lines=15, max_lines=15, interactive=False)
                            with gr.TabItem("Response Trace", id=1) as trace_tab:
                                actions = gr.JSON(
                                    scale=1,
                                    show_label=False,
                                    visible=True,
                                    elem_id="contextbox",
                                )
                    
                    # Third tab item is for collapsing the entire settings pane for readability. 
                    with gr.TabItem("Hide All Settings", id=4, interactive=True, visible=True) as hide_all_settings:
                        gr.Markdown("")

        page.load(logger.read_logs, None, logs, every=1)

        """ This helper function runs the initialization tasks of the chat application, if needed. """

        def _toggle_welcome(progress=gr.Progress()):
            progress(0.17, desc="Initializing Task")
            time.sleep(0.75)
            progress(0.33, desc="Initializing Retriever(s)")
            database.initialize_web_retriever()
            progress(0.5, desc="Initializing PDF Retriever")
            database.initialize_pdf_retriever()
            progress(0.67, desc="Initializing Image Retriever")
            database.initialize_img_retriever()
            progress(0.83, desc="Cleaning up")
            time.sleep(0.75)
            return {
                initialized: True,
                msg: gr.update(interactive=True), 
                sample_query_1: gr.update(interactive=True), 
                sample_query_2: gr.update(interactive=True), 
                sample_query_3: gr.update(interactive=True), 
                sample_query_4: gr.update(interactive=True), 
                settings_tabs: gr.update(selected=1), 
                welcome_settings: gr.update(visible=False, interactive=False), 
                agent_settings: gr.update(visible=True, interactive=True), 
                document_settings: gr.update(visible=True, interactive=True), 
                console_settings: gr.update(visible=True, interactive=True), 
                agentic_flow: gr.update(visible=True),
            }

        initialize.click(_toggle_welcome, [], [initialized, 
                                               msg, 
                                               sample_query_1, 
                                               sample_query_2, 
                                               sample_query_3, 
                                               sample_query_4, 
                                               settings_tabs, 
                                               welcome_settings, 
                                               agent_settings, 
                                               document_settings, 
                                               console_settings, 
                                               agentic_flow])

        """ These helper functions hide the expanded component model settings when the Hide tab is clicked. """
        
        def _toggle_hide_router():
            return {
                group_router: gr.update(visible=False),
                router_tabs: gr.update(selected=0),
                router_btn: gr.update(visible=True),
            }

        def _toggle_hide_retrieval():
            return {
                group_retrieval: gr.update(visible=False),
                retrieval_tabs: gr.update(selected=0),
                retrieval_btn: gr.update(visible=True),
            }

        def _toggle_hide_generator():
            return {
                group_generator: gr.update(visible=False),
                generator_tabs: gr.update(selected=0),
                generator_btn: gr.update(visible=True),
            }

        def _toggle_hide_hallucination():
            return {
                group_hallucination: gr.update(visible=False),
                hallucination_tabs: gr.update(selected=0),
                hallucination_btn: gr.update(visible=True),
            }

        def _toggle_hide_answer():
            return {
                group_answer: gr.update(visible=False),
                answer_tabs: gr.update(selected=0),
                answer_btn: gr.update(visible=True),
            }

        router_hide.select(_toggle_hide_router, None, [group_router, router_tabs, router_btn])
        retrieval_hide.select(_toggle_hide_retrieval, None, [group_retrieval, retrieval_tabs, retrieval_btn])
        generator_hide.select(_toggle_hide_generator, None, [group_generator, generator_tabs, generator_btn])
        hallucination_hide.select(_toggle_hide_hallucination, None, [group_hallucination, hallucination_tabs, hallucination_btn])
        answer_hide.select(_toggle_hide_answer, None, [group_answer, answer_tabs, answer_btn])

        """ These helper functions set state and prompts when either the NIM or API Endpoint tabs are selected. """
        
        def _toggle_router_endpoints(api_model: str, nim_model: str, evt: gr.EventData):
            if (evt._data['value'] == "NIM Endpoints") and ("llama3" in nim_model or len(nim_model) == 0):
                value = prompts_llama3.router_prompt
            elif (evt._data['value'] == "NIM Endpoints") and ("mistral" in nim_model or "mixtral" in nim_model):
                value = prompts_mistral.router_prompt
            elif (evt._data['value'] == "API Endpoints") and ("llama3" in api_model):
                value = prompts_llama3.router_prompt
            elif (evt._data['value'] == "API Endpoints") and ("mistral" in api_model or "mixtral" in api_model):
                value = prompts_mistral.router_prompt
            return True if evt._data['value'] == "NIM Endpoints" else False, gr.update(value=value) if value is not None else gr.update(visible=True)

        def _toggle_retrieval_endpoints(api_model: str, nim_model: str, evt: gr.EventData):
            if (evt._data['value'] == "NIM Endpoints") and ("llama3" in nim_model or len(nim_model) == 0):
                value = prompts_llama3.retrieval_prompt
            elif (evt._data['value'] == "NIM Endpoints") and ("mistral" in nim_model or "mixtral" in nim_model):
                value = prompts_mistral.retrieval_prompt
            elif (evt._data['value'] == "API Endpoints") and ("llama3" in api_model):
                value = prompts_llama3.retrieval_prompt
            elif (evt._data['value'] == "API Endpoints") and ("mistral" in api_model or "mixtral" in api_model):
                value = prompts_mistral.retrieval_prompt
            return True if evt._data['value'] == "NIM Endpoints" else False, gr.update(value=value) if value is not None else gr.update(visible=True)

        def _toggle_generator_endpoints(api_model: str, nim_model: str, evt: gr.EventData):
            if (evt._data['value'] == "NIM Endpoints") and ("llama3" in nim_model or len(nim_model) == 0):
                value = prompts_llama3.generator_prompt
            elif (evt._data['value'] == "NIM Endpoints") and ("mistral" in nim_model or "mixtral" in nim_model):
                value = prompts_mistral.generator_prompt
            elif (evt._data['value'] == "API Endpoints") and ("llama3" in api_model):
                value = prompts_llama3.generator_prompt
            elif (evt._data['value'] == "API Endpoints") and ("mistral" in api_model or "mixtral" in api_model):
                value = prompts_mistral.generator_prompt
            return True if evt._data['value'] == "NIM Endpoints" else False, gr.update(value=value) if value is not None else gr.update(visible=True)

        def _toggle_hallucination_endpoints(api_model: str, nim_model: str, evt: gr.EventData):
            if (evt._data['value'] == "NIM Endpoints") and ("llama3" in nim_model or len(nim_model) == 0):
                value = prompts_llama3.hallucination_prompt
            elif (evt._data['value'] == "NIM Endpoints") and ("mistral" in nim_model or "mixtral" in nim_model):
                value = prompts_mistral.hallucination_prompt
            elif (evt._data['value'] == "API Endpoints") and ("llama3" in api_model):
                value = prompts_llama3.hallucination_prompt
            elif (evt._data['value'] == "API Endpoints") and ("mistral" in api_model or "mixtral" in api_model):
                value = prompts_mistral.hallucination_prompt
            return True if evt._data['value'] == "NIM Endpoints" else False, gr.update(value=value) if value is not None else gr.update(visible=True)

        def _toggle_answer_endpoints(api_model: str, nim_model: str, evt: gr.EventData):
            if (evt._data['value'] == "NIM Endpoints") and ("llama3" in nim_model or len(nim_model) == 0):
                value = prompts_llama3.answer_prompt
            elif (evt._data['value'] == "NIM Endpoints") and ("mistral" in nim_model or "mixtral" in nim_model):
                value = prompts_mistral.answer_prompt
            elif (evt._data['value'] == "API Endpoints") and ("llama3" in api_model):
                value = prompts_llama3.answer_prompt
            elif (evt._data['value'] == "API Endpoints") and ("mistral" in api_model or "mixtral" in api_model):
                value = prompts_mistral.answer_prompt
            return True if evt._data['value'] == "NIM Endpoints" else False, gr.update(value=value) if value is not None else gr.update(visible=True)

        router_api.select(_toggle_router_endpoints, [model_router, nim_router_id], [router_use_nim, prompt_router])
        router_nim.select(_toggle_router_endpoints, [model_router, nim_router_id], [router_use_nim, prompt_router])
        retrieval_api.select(_toggle_retrieval_endpoints, [model_retrieval, nim_retrieval_id], [retrieval_use_nim, prompt_retrieval])
        retrieval_nim.select(_toggle_retrieval_endpoints, [model_retrieval, nim_retrieval_id], [retrieval_use_nim, prompt_retrieval])
        generator_api.select(_toggle_generator_endpoints, [model_generator, nim_generator_id], [generator_use_nim, prompt_generator])
        generator_nim.select(_toggle_generator_endpoints, [model_generator, nim_generator_id], [generator_use_nim, prompt_generator])
        hallucination_api.select(_toggle_hallucination_endpoints, [model_hallucination, nim_hallucination_id], [hallucination_use_nim, prompt_hallucination])
        hallucination_nim.select(_toggle_hallucination_endpoints, [model_hallucination, nim_hallucination_id], [hallucination_use_nim, prompt_hallucination])
        answer_api.select(_toggle_answer_endpoints, [model_answer, nim_answer_id], [answer_use_nim, prompt_answer])
        answer_nim.select(_toggle_answer_endpoints, [model_answer, nim_answer_id], [answer_use_nim, prompt_answer])
        
        """ These helper functions hide and show the right-hand settings panel when toggled. """
        
        def _toggle_hide_all_settings():
            return {
                settings_column: gr.update(visible=False),
                hidden_settings_column: gr.update(visible=True),
            }

        def _toggle_show_all_settings(initialized):
            return {
                settings_column: gr.update(visible=True),
                settings_tabs: gr.update(selected=1) if initialized else gr.update(selected=0),
                hidden_settings_column: gr.update(visible=False),
            }

        hide_all_settings.select(_toggle_hide_all_settings, None, [settings_column, hidden_settings_column])
        show_settings.click(_toggle_show_all_settings, [initialized], [settings_column, settings_tabs, hidden_settings_column])
        
        """ This helper function ensures the model settings are reset when a user re-navigates to the tab. """
        
        def _toggle_model_tab():
            return {
                group_router: gr.update(visible=False),
                group_retrieval: gr.update(visible=False),
                group_generator: gr.update(visible=False),
                group_hallucination: gr.update(visible=False),
                group_answer: gr.update(visible=False),
                router_btn: gr.update(visible=True),
                retrieval_btn: gr.update(visible=True),
                generator_btn: gr.update(visible=True),
                hallucination_btn: gr.update(visible=True),
                answer_btn: gr.update(visible=True),
            }
        
        agent_settings.select(_toggle_model_tab, [], [group_router,
                                                      group_retrieval,
                                                      group_generator,
                                                      group_hallucination,
                                                      group_answer,
                                                      router_btn,
                                                      retrieval_btn,
                                                      generator_btn,
                                                      hallucination_btn,
                                                      answer_btn])

        """ This helper function ensures only one component model settings are expanded at a time when selected. """

        def _toggle_model(btn: str):
            if btn == "Router":
                group_visible = [True, False, False, False, False]
                button_visible = [False, True, True, True, True]
            elif btn == "Retrieval Grader":
                group_visible = [False, True, False, False, False]
                button_visible = [True, False, True, True, True]
            elif btn == "Generator":
                group_visible = [False, False, True, False, False]
                button_visible = [True, True, False, True, True]
            elif btn == "Hallucination Grader":
                group_visible = [False, False, False, True, False]
                button_visible = [True, True, True, False, True]
            elif btn == "Answer Grader":
                group_visible = [False, False, False, False, True]
                button_visible = [True, True, True, True, False]
            return {
                group_router: gr.update(visible=group_visible[0]),
                group_retrieval: gr.update(visible=group_visible[1]),
                group_generator: gr.update(visible=group_visible[2]),
                group_hallucination: gr.update(visible=group_visible[3]),
                group_answer: gr.update(visible=group_visible[4]),
                router_btn: gr.update(visible=button_visible[0]),
                retrieval_btn: gr.update(visible=button_visible[1]),
                generator_btn: gr.update(visible=button_visible[2]),
                hallucination_btn: gr.update(visible=button_visible[3]),
                answer_btn: gr.update(visible=button_visible[4]),
            }

        router_btn.click(_toggle_model, [router_btn], [group_router,
                                                       group_retrieval,
                                                       group_generator,
                                                       group_hallucination,
                                                       group_answer,
                                                       router_btn,
                                                       retrieval_btn,
                                                       generator_btn,
                                                       hallucination_btn,
                                                       answer_btn])
        
        retrieval_btn.click(_toggle_model, [retrieval_btn], [group_router,
                                                                           group_retrieval,
                                                                           group_generator,
                                                                           group_hallucination,
                                                                           group_answer,
                                                                           router_btn,
                                                                           retrieval_btn,
                                                                           generator_btn,
                                                                           hallucination_btn,
                                                                           answer_btn])
        
        generator_btn.click(_toggle_model, [generator_btn], [group_router,
                                                             group_retrieval,
                                                             group_generator,
                                                             group_hallucination,
                                                             group_answer,
                                                             router_btn,
                                                             retrieval_btn,
                                                             generator_btn,
                                                             hallucination_btn,
                                                             answer_btn])
        
        hallucination_btn.click(_toggle_model, [hallucination_btn], [group_router,
                                                                                   group_retrieval,
                                                                                   group_generator,
                                                                                   group_hallucination,
                                                                                   group_answer,
                                                                                   router_btn,
                                                                                   retrieval_btn,
                                                                                   generator_btn,
                                                                                   hallucination_btn,
                                                                                   answer_btn])
        
        answer_btn.click(_toggle_model, [answer_btn], [group_router,
                                                                     group_retrieval,
                                                                     group_generator,
                                                                     group_hallucination,
                                                                     group_answer,
                                                                     router_btn,
                                                                     retrieval_btn,
                                                                     generator_btn,
                                                                     hallucination_btn,
                                                                     answer_btn])

        """ This helper function tracks which webpage is currently open for user upload. """
        
        def _toggle_webpages(btn: str):
            if btn == "Webpage URLs":
                group_visible = [True, False]
                button_visible = [False, True]
            elif btn == "YouTube URLs":
                group_visible = [False, True]
                button_visible = [True, False]
            return {
                group_web_url: gr.update(visible=group_visible[0]),
                group_video_url: gr.update(visible=group_visible[1]),
                web_url_btn: gr.update(visible=button_visible[0]),
                video_url_btn: gr.update(visible=button_visible[1]),            
            }
        
        web_url_btn.click(_toggle_webpages, [web_url_btn], [group_web_url,
                                                         group_video_url,
                                                         web_url_btn,
                                                         video_url_btn])

        video_url_btn.click(_toggle_webpages, [video_url_btn], [group_web_url,
                                                         group_video_url,
                                                         web_url_btn,
                                                         video_url_btn])

        """ This helper function tracks which local files are currently open for user upload. """

        def _toggle_upload(btn: str):
            if btn == "PDF Upload":
                group_visible = [True, False, False]
                button_visible = [False, True, True]
            elif btn == "Image Upload":
                group_visible = [False, True, False]
                button_visible = [True, False, True]
            elif btn == "Video Upload":
                group_visible = [False, False, True]
                button_visible = [True, True, False]
            return {
                group_pdf_upload: gr.update(visible=group_visible[0]),
                group_image_upload: gr.update(visible=group_visible[1]),
                group_video_upload: gr.update(visible=group_visible[2]),
                pdf_upload_btn: gr.update(visible=button_visible[0]),
                image_upload_btn: gr.update(visible=button_visible[1]),            
                video_upload_btn: gr.update(visible=button_visible[2]),            
            }
        
        pdf_upload_btn.click(_toggle_upload, [pdf_upload_btn], [group_pdf_upload,
                                                                group_image_upload,
                                                                group_video_upload,
                                                                pdf_upload_btn,
                                                                image_upload_btn,
                                                                video_upload_btn])

        image_upload_btn.click(_toggle_upload, [image_upload_btn], [group_pdf_upload,
                                                                group_image_upload,
                                                                group_video_upload,
                                                                pdf_upload_btn,
                                                                image_upload_btn,
                                                                video_upload_btn])

        video_upload_btn.click(_toggle_upload, [video_upload_btn], [group_pdf_upload,
                                                                group_image_upload,
                                                                group_video_upload,
                                                                pdf_upload_btn,
                                                                image_upload_btn,
                                                                video_upload_btn])

        """ These helper functions track the API Endpoint selected and regenerates the prompt accordingly. """
        
        def _toggle_model_router(selected_model: str):
            match selected_model:
                case "meta/llama-3.1-405b-instruct":
                    return gr.update(value=prompts_llama3.router_prompt)
                case "meta/llama3-70b-instruct":
                    return gr.update(value=prompts_llama3.router_prompt)
                case "mistralai/mixtral-8x22b-instruct-v0.1":
                    return gr.update(value=prompts_mistral.router_prompt)
                case _:
                    return gr.update(value=prompts_llama3.router_prompt)
        
        def _toggle_model_retrieval(selected_model: str):
            match selected_model:
                case "meta/llama-3.1-405b-instruct":
                    return gr.update(value=prompts_llama3.retrieval_prompt)
                case "meta/llama3-70b-instruct":
                    return gr.update(value=prompts_llama3.retrieval_prompt)
                case "mistralai/mixtral-8x22b-instruct-v0.1":
                    return gr.update(value=prompts_mistral.retrieval_prompt)
                case _:
                    return gr.update(value=prompts_llama3.retrieval_prompt)

        def _toggle_model_generator(selected_model: str):
            match selected_model:
                case "meta/llama-3.1-405b-instruct":
                    return gr.update(value=prompts_llama3.generator_prompt)
                case "meta/llama3-70b-instruct":
                    return gr.update(value=prompts_llama3.generator_prompt)
                case "mistralai/mixtral-8x22b-instruct-v0.1":
                    return gr.update(value=prompts_mistral.generator_prompt)
                case _:
                    return gr.update(value=prompts_llama3.generator_prompt)
            
        def _toggle_model_hallucination(selected_model: str):
            match selected_model:
                case "meta/llama-3.1-405b-instruct":
                    return gr.update(value=prompts_llama3.hallucination_prompt)
                case "meta/llama3-70b-instruct":
                    return gr.update(value=prompts_llama3.hallucination_prompt)
                case "mistralai/mixtral-8x22b-instruct-v0.1":
                    return gr.update(value=prompts_mistral.hallucination_prompt)
                case _:
                    return gr.update(value=prompts_llama3.hallucination_prompt)
            
        def _toggle_model_answer(selected_model: str):
            match selected_model:
                case "meta/llama-3.1-405b-instruct":
                    return gr.update(value=prompts_llama3.answer_prompt)
                case "meta/llama3-70b-instruct":
                    return gr.update(value=prompts_llama3.answer_prompt)
                case "mistralai/mixtral-8x22b-instruct-v0.1":
                    return gr.update(value=prompts_mistral.answer_prompt)
                case _:
                    return gr.update(value=prompts_llama3.answer_prompt)
            
        model_router.change(_toggle_model_router, [model_router], [prompt_router])
        model_retrieval.change(_toggle_model_retrieval, [model_retrieval], [prompt_retrieval])
        model_generator.change(_toggle_model_generator, [model_generator], [prompt_generator])
        model_hallucination.change(_toggle_model_hallucination, [model_hallucination], [prompt_hallucination])
        model_answer.change(_toggle_model_answer, [model_answer], [prompt_answer])
        
        """ These helper functions upload and clear the documents to/from the LanceDB. """

        def _clear_documents(progress=gr.Progress()):
            progress(0.25, desc="Initializing Task")
            time.sleep(0.75)
            progress(0.5, desc="Clearing Database")
            database.clear()
            progress(0.75, desc="Cleaning Up")
            time.sleep(0.75)
            return {
                url_docs_upload: gr.update(value="Upload Docs", variant="secondary", interactive=True),
                url_docs_clear: gr.update(value="Docs Cleared", variant="primary", interactive=False),
                pdf_docs_upload: gr.update(value=None),
                file_upload_clear: gr.update(value="Docs Cleared", variant="primary", interactive=False),
                url_videos_upload: gr.update(value="Upload Videos", variant="secondary", interactive=True),
                url_videos_clear: gr.update(value="Docs Cleared", variant="primary", interactive=False),
                agentic_flow: gr.update(visible=True),
            }

        def _upload_webpage_url(docs: str, progress=gr.Progress()):
            progress(0.2, desc="Initializing Task")
            time.sleep(0.75)
            progress(0.4, desc="Processing URL List")
            docs_list = docs.splitlines()
            progress(0.6, desc="Uploading Docs (Be patient, this may take a while)")
            try:
                database.upload_webpage_url(docs_list)
            except Exception as e:
                gr.Warning("Webpage URL upload failed, try again later. Exception: " + str(e))
                return {
                    url_docs_upload: gr.update(value="Upload Docs", variant="secondary", interactive=True),
                    url_docs_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                    file_upload_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                    url_videos_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                    agentic_flow: gr.update(visible=True),
                }
            progress(0.8, desc="Cleaning Up")
            time.sleep(0.75)
            return {
                url_docs_upload: gr.update(value="Docs Uploaded", variant="primary", interactive=False),
                url_docs_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                file_upload_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                url_videos_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                agentic_flow: gr.update(visible=True),
            }

        url_docs_upload.click(_upload_webpage_url, [url_docs], [url_docs_upload, 
                                                              url_docs_clear, 
                                                              file_upload_clear, 
                                                              url_videos_clear, 
                                                              agentic_flow])
        url_docs_clear.click(_clear_documents, [], [url_docs_upload, 
                                                    url_docs_clear, 
                                                    pdf_docs_upload, 
                                                    file_upload_clear, 
                                                    url_videos_upload, 
                                                    url_videos_clear, 
                                                    agentic_flow])
        
        def _upload_video_url(videos, progress=gr.Progress()):
            progress(0.2, desc="Initializing Task")
            time.sleep(0.75)
            progress(0.4, desc="Processing URL List")
            videos_list = videos.splitlines()
            progress(0.6, desc="Uploading Videos (Be patient, this may take a while)")
            try:
                database.upload_video_url(videos_list)
            except Exception as e:
                gr.Warning("Video URL upload failed, try again later. Exception: " + str(e))
                return {
                    url_docs_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                    file_upload_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                    url_videos_upload: gr.update(value="Upload Videos", variant="secondary", interactive=True),
                    url_videos_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                    agentic_flow: gr.update(visible=True),
                }
            progress(0.8, desc="Cleaning Up")
            time.sleep(0.75)
            return {
                url_docs_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                file_upload_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                url_videos_upload: gr.update(value="Videos Uploaded", variant="primary", interactive=False),
                url_videos_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                agentic_flow: gr.update(visible=True),
            }

        url_videos_upload.click(_upload_video_url, [url_videos], [url_docs_clear, 
                                                                                file_upload_clear, 
                                                                                url_videos_upload, 
                                                                                url_videos_clear, 
                                                                                agentic_flow])
        url_videos_clear.click(_clear_documents, [], [url_docs_upload, 
                                                      url_docs_clear, 
                                                      pdf_docs_upload, 
                                                      file_upload_clear, 
                                                      url_videos_upload, 
                                                      url_videos_clear, 
                                                      agentic_flow])

        def _upload_pdf_documents(files, progress=gr.Progress()):
            progress(0.25, desc="Initializing Task")
            time.sleep(0.75)
            progress(0.5, desc="Uploading Docs (Be patient, this may take a while)")
            try:
                database.upload_pdf(files)
            except Exception as e:
                gr.Warning("PDF file upload failed, try again later. Exception: " + str(e))
                return {
                    url_docs_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                    file_upload_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                    url_videos_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                    agentic_flow: gr.update(visible=True),
                }
            progress(0.75, desc="Cleaning Up")
            time.sleep(0.75)
            return {
                url_docs_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                file_upload_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                url_videos_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                agentic_flow: gr.update(visible=True),
            }

        pdf_docs_upload.upload(_upload_pdf_documents, [pdf_docs_upload], [url_docs_clear, 
                                                                          file_upload_clear, 
                                                                          url_videos_clear, 
                                                                          agentic_flow])
        
        def _upload_image_documents(files, progress=gr.Progress()):
            progress(0.25, desc="Initializing Task")
            time.sleep(0.75)
            progress(0.5, desc="Uploading Images (Be patient, this may take a while)")
            try:
                database.upload_image(files)
            except Exception as e:
                gr.Warning("Image file upload failed, try again later. Exception: " + str(e))
                return {
                    url_docs_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                    file_upload_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                    url_videos_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                    agentic_flow: gr.update(visible=True),
                }
            progress(0.75, desc="Cleaning Up")
            time.sleep(0.75)
            return {
                url_docs_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                file_upload_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                url_videos_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                agentic_flow: gr.update(visible=True),
            }

        image_upload.upload(_upload_image_documents, [image_upload], [url_docs_clear, 
                                                                          file_upload_clear, 
                                                                          url_videos_clear, 
                                                                          agentic_flow])

        def _upload_video_documents(files, progress=gr.Progress()):
            progress(0.25, desc="Initializing Task")
            time.sleep(0.75)
            progress(0.5, desc="Uploading Videos (Be patient, this may take a while)")
            try:
                database.upload_video(files)
            except Exception as e:
                gr.Warning("Video file upload failed, try again later. Exception: " + str(e))
                return {
                    url_docs_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                    file_upload_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                    url_videos_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                    agentic_flow: gr.update(visible=True),
                }
            progress(0.75, desc="Cleaning Up")
            time.sleep(0.75)
            return {
                url_docs_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                file_upload_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                url_videos_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                agentic_flow: gr.update(visible=True),
            }

        video_upload.upload(_upload_video_documents, [video_upload], [url_docs_clear, 
                                                                          file_upload_clear, 
                                                                          url_videos_clear, 
                                                                          agentic_flow])

        file_upload_clear.click(_clear_documents, [], [url_docs_upload, 
                                                    url_docs_clear, 
                                                    pdf_docs_upload, 
                                                    file_upload_clear, 
                                                    url_videos_upload, 
                                                    url_videos_clear, 
                                                    agentic_flow])

        """ This helper function builds out the submission function call when a user submits a query. """
        
        _my_build_stream = functools.partial(_stream_predict, client, app)

        # Submit a sample query
        sample_query_1.click(
            _my_build_stream, [sample_query_1, 
                               model_generator,
                               model_router,
                               model_retrieval,
                               model_hallucination,
                               model_answer,
                               prompt_generator,
                               prompt_router,
                               prompt_retrieval,
                               prompt_hallucination,
                               prompt_answer,
                               router_use_nim,
                               retrieval_use_nim,
                               generator_use_nim,
                               hallucination_use_nim,
                               answer_use_nim,
                               nim_generator_ip,
                               nim_router_ip,
                               nim_retrieval_ip,
                               nim_hallucination_ip,
                               nim_answer_ip,
                               nim_generator_port,
                               nim_router_port,
                               nim_retrieval_port,
                               nim_hallucination_port,
                               nim_answer_port,
                               nim_generator_id,
                               nim_router_id,
                               nim_retrieval_id,
                               nim_hallucination_id,
                               nim_answer_id,
                               chatbot], [msg, chatbot, actions]
        )

        sample_query_2.click(
            _my_build_stream, [sample_query_2, 
                               model_generator,
                               model_router,
                               model_retrieval,
                               model_hallucination,
                               model_answer,
                               prompt_generator,
                               prompt_router,
                               prompt_retrieval,
                               prompt_hallucination,
                               prompt_answer,
                               router_use_nim,
                               retrieval_use_nim,
                               generator_use_nim,
                               hallucination_use_nim,
                               answer_use_nim,
                               nim_generator_ip,
                               nim_router_ip,
                               nim_retrieval_ip,
                               nim_hallucination_ip,
                               nim_answer_ip,
                               nim_generator_port,
                               nim_router_port,
                               nim_retrieval_port,
                               nim_hallucination_port,
                               nim_answer_port,
                               nim_generator_id,
                               nim_router_id,
                               nim_retrieval_id,
                               nim_hallucination_id,
                               nim_answer_id,
                               chatbot], [msg, chatbot, actions]
        )

        sample_query_3.click(
            _my_build_stream, [sample_query_3, 
                               model_generator,
                               model_router,
                               model_retrieval,
                               model_hallucination,
                               model_answer,
                               prompt_generator,
                               prompt_router,
                               prompt_retrieval,
                               prompt_hallucination,
                               prompt_answer,
                               router_use_nim,
                               retrieval_use_nim,
                               generator_use_nim,
                               hallucination_use_nim,
                               answer_use_nim,
                               nim_generator_ip,
                               nim_router_ip,
                               nim_retrieval_ip,
                               nim_hallucination_ip,
                               nim_answer_ip,
                               nim_generator_port,
                               nim_router_port,
                               nim_retrieval_port,
                               nim_hallucination_port,
                               nim_answer_port,
                               nim_generator_id,
                               nim_router_id,
                               nim_retrieval_id,
                               nim_hallucination_id,
                               nim_answer_id,
                               chatbot], [msg, chatbot, actions]
        )

        sample_query_4.click(
            _my_build_stream, [sample_query_4, 
                               model_generator,
                               model_router,
                               model_retrieval,
                               model_hallucination,
                               model_answer,
                               prompt_generator,
                               prompt_router,
                               prompt_retrieval,
                               prompt_hallucination,
                               prompt_answer,
                               router_use_nim,
                               retrieval_use_nim,
                               generator_use_nim,
                               hallucination_use_nim,
                               answer_use_nim,
                               nim_generator_ip,
                               nim_router_ip,
                               nim_retrieval_ip,
                               nim_hallucination_ip,
                               nim_answer_ip,
                               nim_generator_port,
                               nim_router_port,
                               nim_retrieval_port,
                               nim_hallucination_port,
                               nim_answer_port,
                               nim_generator_id,
                               nim_router_id,
                               nim_retrieval_id,
                               nim_hallucination_id,
                               nim_answer_id,
                               chatbot], [msg, chatbot, actions]
        )

        # Submit a custom query
        msg.submit(
            _my_build_stream, [msg, 
                               model_generator,
                               model_router,
                               model_retrieval,
                               model_hallucination,
                               model_answer,
                               prompt_generator,
                               prompt_router,
                               prompt_retrieval,
                               prompt_hallucination,
                               prompt_answer,
                               router_use_nim,
                               retrieval_use_nim,
                               generator_use_nim,
                               hallucination_use_nim,
                               answer_use_nim,
                               nim_generator_ip,
                               nim_router_ip,
                               nim_retrieval_ip,
                               nim_hallucination_ip,
                               nim_answer_ip,
                               nim_generator_port,
                               nim_router_port,
                               nim_retrieval_port,
                               nim_hallucination_port,
                               nim_answer_port,
                               nim_generator_id,
                               nim_router_id,
                               nim_retrieval_id,
                               nim_hallucination_id,
                               nim_answer_id,
                               chatbot], [msg, chatbot, actions]
        )

    page.queue()
    return page

""" This helper function verifies that a user query is nonempty. """

def valid_input(query: str):
    return False if query.isspace() or query is None or query == "" or query == '' else True

""" This helper function executes and generates a response to the user query. """

def _stream_predict(
    client: chat_client.ChatClient,
    app, 
    question: str,
    model_generator: str,
    model_router: str,
    model_retrieval: str,
    model_hallucination: str,
    model_answer: str,
    prompt_generator: str,
    prompt_router: str,
    prompt_retrieval: str,
    prompt_hallucination: str,
    prompt_answer: str,
    router_use_nim: bool,
    retrieval_use_nim: bool,
    generator_use_nim: bool,
    hallucination_use_nim: bool,
    answer_use_nim: bool,
    nim_generator_ip: str,
    nim_router_ip: str,
    nim_retrieval_ip: str,
    nim_hallucination_ip: str,
    nim_answer_ip: str,
    nim_generator_port: str,
    nim_router_port: str,
    nim_retrieval_port: str,
    nim_hallucination_port: str,
    nim_answer_port: str,
    nim_generator_id: str,
    nim_router_id: str,
    nim_retrieval_id: str,
    nim_hallucination_id: str,
    nim_answer_id: str,
    chat_history: List[Tuple[str, str]],
) -> Any:

    inputs = {"question": question, 
              "generator_model_id": model_generator, 
              "router_model_id": model_router, 
              "retrieval_model_id": model_retrieval, 
              "hallucination_model_id": model_hallucination, 
              "answer_model_id": model_answer, 
              "prompt_generator": prompt_generator, 
              "prompt_router": prompt_router, 
              "prompt_retrieval": prompt_retrieval, 
              "prompt_hallucination": prompt_hallucination, 
              "prompt_answer": prompt_answer, 
              "router_use_nim": router_use_nim, 
              "retrieval_use_nim": retrieval_use_nim, 
              "generator_use_nim": generator_use_nim, 
              "hallucination_use_nim": hallucination_use_nim, 
              "nim_generator_ip": nim_generator_ip,
              "nim_router_ip": nim_router_ip,
              "nim_retrieval_ip": nim_retrieval_ip,
              "nim_hallucination_ip": nim_hallucination_ip,
              "nim_answer_ip": nim_answer_ip,
              "nim_generator_port": nim_generator_port,
              "nim_router_port": nim_router_port,
              "nim_retrieval_port": nim_retrieval_port,
              "nim_hallucination_port": nim_hallucination_port,
              "nim_answer_port": nim_answer_port,
              "nim_generator_id": nim_generator_id,
              "nim_router_id": nim_router_id,
              "nim_retrieval_id": nim_retrieval_id,
              "nim_hallucination_id": nim_hallucination_id,
              "nim_answer_id": nim_answer_id,
              "answer_use_nim": answer_use_nim}
    
    if not valid_input(question):
        yield "", chat_history + [[str(question), "*** ERR: Unable to process query. Query cannot be empty. ***"]], gr.update(show_label=False)
    else: 
        try:
            actions = {}
            for output in app.stream(inputs):
                actions.update(output)
                yield "", chat_history + [[question, "Working on getting you the best answer..."]], gr.update(value=actions)
                for key, value in output.items():
                    final_value = value
            yield "", chat_history + [[question, final_value["generation"]]], gr.update(show_label=False)
        except Exception as e: 
            yield "", chat_history + [[question, "*** ERR: Unable to process query. See Monitor tab for details. ***\n\nException: " + str(e)]], gr.update(show_label=False)
