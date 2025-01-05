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

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, UnstructuredPDFLoader
from langchain_community.vectorstores import LanceDB
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from typing import Any, Dict, List, Tuple, Union

from moviepy.editor import VideoFileClip
from pathlib import Path
import speech_recognition as sr
from pytubefix import YouTube
from pprint import pprint

import os 
import shutil
import nltk
import requests, base64

from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.node_parser import SentenceSplitter

from chatui_public.prompts import defaults

""" Global variables. Cuts down on retrieval time for now, can refactor. """

text_store = None
image_store = None
storage_context = None
documents = None
img_vectorstore = None
web_vectorstore = None
pdf_vectorstore = None

def download_video(url, output_path):
    """
    Download a video from a given url and save it to the output path.

    Parameters:
    url (str): The url of the video to download.
    output_path (str): The path to save the video to.

    Returns:
    dict: A dictionary containing the metadata of the video.
    """
    yt = YouTube(url)
    # metadata = {"Author": yt.author, "Title": yt.title, "Views": yt.views}
    yt.streams.get_highest_resolution().download(
        output_path=output_path, filename="input_vid.mp4"
    )
    return {}


def video_to_images(video_path, output_folder, num_vids):
    """
    Convert a video to a sequence of images and save them to the output folder.

    Parameters:
    video_path (str): The path to the video file.
    output_folder (str): The path to the folder to save the images to.

    """
    clip = VideoFileClip(video_path)
    clip.write_images_sequence(
        os.path.join(output_folder, str(num_vids) + "_frame%04d.png"), fps=0.2
    )


def video_to_audio(video_path, output_audio_path):
    """
    Convert a video to audio and save it to the output path.

    Parameters:
    video_path (str): The path to the video file.
    output_audio_path (str): The path to save the audio to.

    """
    clip = VideoFileClip(video_path)
    audio = clip.audio
    audio.write_audiofile(output_audio_path)


def audio_to_text(audio_path):
    """
    Convert audio to text using the SpeechRecognition library.

    Parameters:
    audio_path (str): The path to the audio file.

    Returns:
    test (str): The text recognized from the audio.

    """
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(audio_path)

    with audio as source:
        # Record the audio data
        audio_data = recognizer.record(source)

        try:
            # Recognize the speech
            text = recognizer.recognize_whisper(audio_data)
        except sr.UnknownValueError:
            print("Speech recognition could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results from service; {e}")

    return text

def upload_webpage_url(urls: List[str]):
    """ This is a helper function for parsing the user inputted URLs and uploading them into the vector store. """
    global web_vectorstore
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    
    # Add to vectorDB
    web_vectorstore = LanceDB.from_documents(
        uri="/project/data/lancedb",
        table_name="web_collection",
        documents=doc_splits,
        embedding=NVIDIAEmbeddings(model='NV-Embed-QA'),
    )
    return web_vectorstore

def upload_pdf(pdfs: List[str]):
    """ This is a helper function for parsing the user inputted URLs and uploading them into the vector store. """
    global pdf_vectorstore
    nltk.download('punkt_tab')
    docs = [UnstructuredPDFLoader(document).load() for document in pdfs]
    docs_list = [item for sublist in docs for item in sublist]
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    
    # Add to vectorDB
    pdf_vectorstore = LanceDB.from_documents(
        uri="/project/data/lancedb",
        table_name="pdf_collection",
        documents=doc_splits,
        embedding=NVIDIAEmbeddings(model='NV-Embed-QA'),
    )
    return pdf_vectorstore

def get_num_vids():
    """ This is a helper function for getting the number of videos already uploaded to the database. """
    count = 0
    for filename in os.listdir("/project/data/video_data/"):
        count += 1
    return count

def upload_video_url(videos: List[str]):
    """ This is a helper function for parsing the user inputted URLs and uploading them into the vector store. """
    global img_vectorstore
    output_video_path = "/project/data/video_data/"
    output_folder = "/project/data/mixed_data/"
    output_audio_path = "/project/data/mixed_data/output_audio.wav"
    
    Path(output_video_path).mkdir(parents=True, exist_ok=True)
    
    for video_url in videos: 
        num_vids = get_num_vids()
        filepath = output_video_path + "input_vid_" + str(num_vids) + ".mp4"
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        download_video(video_url, output_video_path)
        video_to_images(filepath, output_folder, num_vids)
        video_to_audio(filepath, output_audio_path)
        text_data = audio_to_text(output_audio_path)
    
        with open(output_folder + "yt_output_text_" + str(num_vids) + ".txt", "w") as file:
            file.write(text_data)
        print("Text data saved to file")
        file.close()
        os.remove(output_audio_path)
        print("Audio file removed")
    
    global text_store, image_store, storage_context, documents, vectorstore

    text_store = LanceDBVectorStore(uri="/project/data/lancedb", 
                                    table_name="text_img_collection", 
                                    embedding=NVIDIAEmbeddings(model='NV-Embed-QA'))
    image_store = LanceDBVectorStore(uri="/project/data/lancedb", 
                                     table_name="image_collection", 
                                     embedding=NVIDIAEmbeddings(model='nvidia/nvclip'))
    storage_context = StorageContext.from_defaults(
        vector_store=text_store, image_store=image_store
    )
    
    documents = SimpleDirectoryReader(output_folder).load_data()
    
    img_vectorstore = MultiModalVectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    
    return img_vectorstore

def get_num_images():
    """ This is a helper function for getting the number of images already uploaded to the database. """
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg']
    file_count = 0
    for root, dirs, files in os.walk("/project/data/mixed_data/"):
        file_count += sum(1 for file in files if os.path.splitext(file)[1].lower() in image_extensions)
    return file_count

def get_base_64(image):
    """ This is a helper function for getting base64 encoding of an image. """
    with open(image, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
    return image_b64

def generate_image_description(image_b64):
    """ This is a helper function for getting a Llama 3.2 vision description of an image. """
    invoke_url = "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct/chat/completions"
    headers = {
        "Authorization": "Bearer " + str(os.environ["NVIDIA_API_KEY"]),
        "Accept": "application/json"
    }
    payload = {
        "model": 'meta/llama-3.2-90b-vision-instruct',
        "messages": [
            {
                "role": "user",
                "content": f'{defaults.vllm_prompt} <img src="data:image/png;base64,{image_b64}" />'
            }
        ],
        "max_tokens": 512,
        "temperature": 1.00,
        "top_p": 1.00,
        "stream": False
    }
    response = requests.post(invoke_url, headers=headers, json=payload)
    return response.json()

def upload_image(images: List[str]):
    """ This is a helper function for parsing the user specific image file and uploading them into the vector store. """
    global img_vectorstore
    output_folder = "/project/data/mixed_data/"
    
    for image_path in images: 
        num_images = get_num_images()
        image_b64 = get_base_64(image_path)
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        shutil.move(image_path.name, output_folder + "output_image_" + str(num_images) + "." + image_path.name.split(".")[1])
        text_data = generate_image_description(image_b64)["choices"][0]["message"]["content"]
    
        with open(output_folder + "img_output_text_" + str(num_images) + ".txt", "w") as file:
            file.write(text_data)
        print("Text data saved to file")
        file.close()
    
    global text_store, image_store, storage_context, documents, vectorstore
    
    text_store = LanceDBVectorStore(uri="/project/data/lancedb", 
                                    table_name="text_img_collection", 
                                    embedding=NVIDIAEmbeddings(model='NV-Embed-QA'))
    image_store = LanceDBVectorStore(uri="/project/data/lancedb", 
                                     table_name="image_collection", 
                                     embedding=NVIDIAEmbeddings(model='nvidia/nvclip'))
    storage_context = StorageContext.from_defaults(
        vector_store=text_store, image_store=image_store
    )
    
    documents = SimpleDirectoryReader(output_folder).load_data()
    
    img_vectorstore = MultiModalVectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    
    return img_vectorstore

def upload_video(videos: List[str]):
    """ This is a helper function for parsing the user specific video file and uploading them into the vector store. """
    global img_vectorstore
    output_video_path = "/project/data/video_data/"
    output_folder = "/project/data/mixed_data/"
    output_audio_path = "/project/data/mixed_data/output_audio.wav"
    
    Path(output_video_path).mkdir(parents=True, exist_ok=True)
    
    for video_path in videos: 
        num_vids = get_num_vids()
        filepath = output_video_path + "input_vid_" + str(num_vids) + ".mp4"
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        shutil.move(video_path.name, filepath)
        video_to_images(filepath, output_folder, num_vids)
        video_to_audio(filepath, output_audio_path)
        text_data = audio_to_text(output_audio_path)
    
        with open(output_folder + "vid_output_text_" + str(num_vids) + ".txt", "w") as file:
            file.write(text_data)
        print("Text data saved to file")
        file.close()
        os.remove(output_audio_path)
        print("Audio file removed")

    global text_store, image_store, storage_context, documents, vectorstore
    
    text_store = LanceDBVectorStore(uri="/project/data/lancedb", 
                                    table_name="text_img_collection", 
                                    embedding=NVIDIAEmbeddings(model='NV-Embed-QA'))
    image_store = LanceDBVectorStore(uri="/project/data/lancedb", 
                                     table_name="image_collection", 
                                     embedding=NVIDIAEmbeddings(model='nvidia/nvclip'))
    storage_context = StorageContext.from_defaults(
        vector_store=text_store, image_store=image_store
    )
    
    documents = SimpleDirectoryReader(output_folder).load_data()
    
    img_vectorstore = MultiModalVectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    
    return img_vectorstore

def clear():
    """ This is a helper function for emptying the collection the vector store. """
    if os.path.exists('/project/data/lancedb/web_collection.lance'):
        vectorstore = LanceDB(
            uri="/project/data/lancedb",
            table_name="web_collection",
            embedding=NVIDIAEmbeddings(model='NV-Embed-QA'),
        )
        
        vectorstore.delete(delete_all=True)

    if os.path.exists('/project/data/lancedb/pdf_collection.lance'):
        vectorstore = LanceDB(
            uri="/project/data/lancedb",
            table_name="pdf_collection",
            embedding=NVIDIAEmbeddings(model='NV-Embed-QA'),
        )
        
        vectorstore.delete(delete_all=True)

    if os.path.exists('/project/data/lancedb/text_img_collection.lance'):
        vectorstore = LanceDB(
            uri="/project/data/lancedb",
            table_name="text_img_collection",
            embedding=NVIDIAEmbeddings(model='NV-Embed-QA'),
        )
        
        vectorstore.delete(delete_all=True)

    if os.path.exists('/project/data/lancedb/image_collection.lance'):
        vectorstore = LanceDB(
            uri="/project/data/lancedb",
            table_name="image_collection",
            embedding=NVIDIAEmbeddings(model='NV-Embed-QA'),
        )
    
        vectorstore.delete(delete_all=True)
    
    if os.path.exists("/project/data/mixed_data/"):
        shutil.rmtree("/project/data/mixed_data/")
    if os.path.exists("/project/data/video_data/"):
        shutil.rmtree("/project/data/video_data/")

def initialize_img_retriever():
    """ This is a helper function for initializing the image retriever on start up. """
    global text_store, image_store, storage_context, documents, img_vectorstore
    if os.path.exists('/project/data/mixed_data/') and bool(os.listdir('/project/data/mixed_data/')):
        text_store = LanceDBVectorStore(uri="/project/data/lancedb", 
                                        table_name="text_img_collection", 
                                        embedding=NVIDIAEmbeddings(model='NV-Embed-QA'))
        image_store = LanceDBVectorStore(uri="/project/data/lancedb", 
                                         table_name="image_collection", 
                                         embedding=NVIDIAEmbeddings(model='nvidia/nvclip'))
        storage_context = StorageContext.from_defaults(
            vector_store=text_store, image_store=image_store
        )
        
        documents = SimpleDirectoryReader("/project/data/mixed_data/").load_data()
        
        img_vectorstore = MultiModalVectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )

def get_img_retriever():
    """ This is a helper function for returning the retriever object of the image vector store. """

    global img_vectorstore

    retriever_engine = img_vectorstore.as_retriever(
        similarity_top_k=3, image_similarity_top_k=3
    )

    return retriever_engine

def initialize_web_retriever():
    """ This is a helper function for initializing the web retriever on start up. """
    global web_vectorstore
    if os.path.exists('/project/data/lancedb/web_collection.lance'):
        web_vectorstore = LanceDB(
            uri="/project/data/lancedb",
            table_name="web_collection",
            embedding=NVIDIAEmbeddings(model='NV-Embed-QA'),
        )

def get_webpage_retriever(): 
    """ This is a helper function for returning the retriever object of the web vector store. """
    global web_vectorstore
    return web_vectorstore

def initialize_pdf_retriever():
    """ This is a helper function for initializing the pdf retriever on start up. """
    global pdf_vectorstore
    if os.path.exists('/project/data/lancedb/pdf_collection.lance'):
        pdf_vectorstore = LanceDB(
            uri="/project/data/lancedb",
            table_name="pdf_collection",
            embedding=NVIDIAEmbeddings(model='NV-Embed-QA'),
        )

def get_pdf_retriever(): 
    """ This is a helper function for returning the retriever object of the pdf vector store. """
    global pdf_vectorstore
    return pdf_vectorstore
