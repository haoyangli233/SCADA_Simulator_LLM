-----

# AI/ML for Networks (ECE-GY-9313) - Final Project

**Institution:** NYU Tandon School of Engineering
**Semester:** Fall 2025

This repository contains the final project code for **ECE-GY-9313: AI/ML for Networks**. The system utilizes Docker containers to integrate **Dify** and **VLLM** for SCADA system reasoning and simulation.

-----

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Model Setup](#model-setup)
3. [Hosting with VLLM](#hosting-with-vllm)
4. [Dify Installation & Configuration](#dify-installation--configuration)
5. [Running the Simulation](#running-the-simulation)

-----

## Prerequisites

Ensure **Docker** and **Docker Compose** are installed on your machine.

  * **Linux:** [Official Docker Engine Installation Guide](https://docs.docker.com/engine/install/)
  * **Windows:** [Docker Desktop for Windows Guide](https://docs.docker.com/desktop/install/windows-install/)

-----

## Model Setup

This project is optimized for **Qwen-4B-Instruct**, though other models will work theoretically.

1.  **Download the Model:**
    You can obtain the model from Hugging Face: [Qwen-4B-Instruct](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

-----

## Hosting with VLLM

We use a Docker container to host the model via VLLM.

**1. Start the Container:**
Run the following command in your terminal (make sure you are in the project folder):

```bash
docker compose -f qwen3-4b.yaml up -d
```

**2. Check Logs:**
To ensure the model is loaded and running correctly, check the logs:

```bash
docker logs qwen3-4b -f
```

You can also see how many request it could process in parallel.
-----

## Dify Installation & Configuration

### 1\. Install Dify

Clone the official Dify repository and follow their deployment guide:

  * **Repository:** [https://github.com/langgenius/dify](https://github.com/langgenius/dify)
  * Follow the standard Docker Compose deployment instructions in their README.

### 2\. Add Model to Dify

Once Dify is running, you must configure the LLM backend.

1.  Go to **Settings \> Model Providers**.
2.  Select **OpenAI-API-compatible** (since VLLM serves an OpenAI-compatible endpoint).
3.  Enter the model details (Endpoint URL and Model Name).

### 3\. Import DSL Scripts

There are two Workflow DSL (`.yml`) files provided in this repository:

  * `LLM for SCADA.yml`
  * `Reasoning for SCADA.yml`

Navigate to the Dify dashboard and import these files to create the apps.

### 4\. Configure Knowledge Base

The system relies on a device information dataset.

1.  Navigate to the **Knowledge** tab and click **Create Knowledge**.
2.  Upload the provided `device_info.csv` file.
3.  Preview the data, then click **Save and Process**.

### 5\. Configure Workflow Context

You must link the Model and Knowledge Base to the imported workflows.

**Workflow Overview:**
Both workflows follow a similar structure, differing primarily in their prompts.

**Steps:**

1.  **Add Knowledge:** inside the workflow context, add the `device_info` knowledge base.
2.  **Configure Model:** Select the OpenAI-Compatible model you added earlier.

### 6\. Publish and Generate API Keys

1.  Click the **Publish** button to make the workflows active.
2.  On the left sidebar, click **API Access**.
3.  Create a **New Secret Key** for *each* of the two workflows.

> **Note:** Copy these two API keys. You will need to paste them into `simulator.py`.

-----

## Running the Simulation

1.  Open `simulator.py` in your code editor.
2.  Paste the two API keys generated in the previous step into the designated configuration variables.
3.  Run the simulator:

<!-- end list -->

```bash
python simulator.py
```
