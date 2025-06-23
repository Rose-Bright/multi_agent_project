# Vertex AI Setup Guide

This guide explains how to configure your application to use Google Cloud Vertex AI instead of OpenAI.

## Prerequisites

1. **Google Cloud Project**: You need an active Google Cloud project with billing enabled.
2. **Vertex AI API**: Enable the Vertex AI API in your project.
3. **Authentication**: Set up authentication using one of the methods below.

## Setup Steps

### 1. Enable Vertex AI API

```bash
# Install Google Cloud CLI if not already installed
# Then enable the Vertex AI API
gcloud services enable aiplatform.googleapis.com
```

### 2. Authentication Options

Choose one of the following authentication methods:

#### Option A: Service Account Key (Recommended for Development)

1. Create a service account:
   ```bash
   gcloud iam service-accounts create vertex-ai-agent \
     --display-name="Vertex AI Agent Service Account"
   ```

2. Grant necessary permissions:
   ```bash
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member="serviceAccount:vertex-ai-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/aiplatform.user"
   ```

3. Create and download the service account key:
   ```bash
   gcloud iam service-accounts keys create service-account-key.json \
     --iam-account=vertex-ai-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com
   ```

4. Set the environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
   ```

#### Option B: Application Default Credentials (Recommended for Production)

```bash
gcloud auth application-default login
```

### 3. Environment Variables

Create a `.env` file in your project root with the following variables:

```env
# Google Cloud Project Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1

# If using service account key file
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account-key.json
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Available Models

The code is configured to use `gemini-2.5-flash`, but you can change to other available models:

- `gemini-2.5-flash` (fast, cost-effective)
- `gemini-2.5-pro` (more capable, higher cost)
- `gemini-1.0-pro` (legacy model)

To change the model, update the `model_name` parameter in `agents/simple_agent.py`:

```python
self.llm = ChatVertexAI(
    model_name="gemini-2.5-pro",  # Change this line
    project=project_id,
    location=location,
    temperature=0.0,
    max_output_tokens=1024
)
```

## Regional Availability

Vertex AI is available in multiple regions. Common options include:
- `us-central1` (default)
- `us-east1`
- `europe-west1`
- `asia-southeast1`

Update the `GOOGLE_CLOUD_LOCATION` environment variable to use a different region.

## Cost Considerations

- Gemini models are priced per token (input and output)
- `gemini-2.5-flash` is more cost-effective for simple tasks
- `gemini-2.5-pro` offers better performance for complex reasoning
- Monitor usage in the Google Cloud Console

## Troubleshooting

### Common Issues

1. **Authentication Error**: Ensure `GOOGLE_APPLICATION_CREDENTIALS` points to a valid service account key file.

2. **Permission Denied**: Verify the service account has the `roles/aiplatform.user` role.

3. **API Not Enabled**: Ensure Vertex AI API is enabled in your project:
   ```bash
   gcloud services list --enabled | grep aiplatform
   ```

4. **Region Issues**: Ensure the specified region supports Vertex AI and the chosen model.

### Verification

Test your setup with this simple script:

```python
from langchain_google_vertexai import ChatVertexAI
import os

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

llm = ChatVertexAI(
    model_name="gemini-2.5-flash",
    project=project_id,
    location=location
)

response = llm.invoke("Hello, how are you?")
print(response.content)
```

## Migration from OpenAI

The key changes made to your code:

1. **Import Change**: `langchain_openai.ChatOpenAI` → `langchain_google_vertexai.ChatVertexAI`
2. **Authentication**: OpenAI API key → Google Cloud credentials
3. **Model Configuration**: OpenAI model names → Vertex AI model names
4. **Environment Variables**: `OPENAI_API_KEY` → `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`

All functionality remains the same - the agent can still:
- Evaluate mathematical expressions safely
- Summarize text by returning the first sentence
- Respond with "Tool not found" for unsupported requests