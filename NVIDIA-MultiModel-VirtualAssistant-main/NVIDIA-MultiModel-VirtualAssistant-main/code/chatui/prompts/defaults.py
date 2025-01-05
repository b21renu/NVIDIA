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

""" These are the default webpages you would like to expose to the user/developer. Adjust as needed! """

webpage_url_defaults = "https://docs.nvidia.com/ai-workbench/user-guide/latest/overview/introduction.html\nhttps://docs.nvidia.com/ai-workbench/user-guide/latest/installation/overview.html\nhttps://docs.nvidia.com/ai-workbench/user-guide/latest/installation/windows.html\nhttps://docs.nvidia.com/ai-workbench/user-guide/latest/installation/macos.html\nhttps://docs.nvidia.com/ai-workbench/user-guide/latest/installation/ubuntu-local.html\nhttps://docs.nvidia.com/ai-workbench/user-guide/latest/installation/ubuntu-remote.html\nhttps://docs.nvidia.com/ai-workbench/user-guide/latest/quickstart/quickstart-basic.html\nhttps://docs.nvidia.com/ai-workbench/user-guide/latest/quickstart/quickstart-cli.html\nhttps://docs.nvidia.com/ai-workbench/user-guide/latest/quickstart/quickstart-environment.html\nhttps://docs.nvidia.com/ai-workbench/user-guide/latest/quickstart/quickstart-environment-cli.html\nhttps://docs.nvidia.com/ai-workbench/user-guide/latest/quickstart/quickstart-custom-app.html\nhttps://docs.nvidia.com/ai-workbench/user-guide/latest/quickstart/quickstart-hybrid-rag.html\nhttps://docs.nvidia.com/ai-workbench/user-guide/latest/quickstart/example-projects.html\nhttps://docs.nvidia.com/ai-workbench/user-guide/latest/locations/remote.html\nhttps://docs.nvidia.com/ai-workbench/user-guide/latest/reference/components.html\nhttps://docs.nvidia.com/ai-workbench/user-guide/latest/reference/cli.html\nhttps://docs.nvidia.com/ai-workbench/user-guide/latest/troubleshooting/troubleshooting.html\nhttps://docs.nvidia.com/ai-workbench/user-guide/latest/troubleshooting/logging.html\nhttps://hackaichallenge.devpost.com/\nhttps://raw.githubusercontent.com/NVIDIA/workbench-example-hybrid-rag/refs/heads/main/README.md\nhttps://raw.githubusercontent.com/NVIDIA/workbench-example-agentic-rag/refs/heads/main/README.md\nhttps://raw.githubusercontent.com/NVIDIA/nim-anywhere/refs/heads/main/README.md"

""" These are the default Youtube URLs you would like to expose to the user/developer. Adjust as needed! """

video_url_defaults = "https://www.youtube.com/watch?v=UXNyf88xI8w\nhttps://www.youtube.com/watch?v=DcrFL_zNRKM"

""" This is the default prompt for the vision language model for understanding of uploaded images. Adjust as needed! """

vllm_prompt = """
You are an expert data analyst and visualization specialist. You are about to see an image that may be, but not necessarily, related to a developer tool for container management and Git-based projects called NVIDIA AI Workbench. Your task is to carefully examine the provided image, which may be of a chart, table, or diagram, and create an highly extensive and comprehensive description that captures all the key information in the image. 

For charts, graphs, tables, and diagrams, transcribe the data being displayed into a table. Make sure you include every single data entry and their corresponding value(s). Then, describe any trends, patterns, outliers, or key insights visible in the data. Finally, summarize the main message or conclusion that can be drawn from the visualization and infer how it may relate to NVIDIA AI Workbench.
"""