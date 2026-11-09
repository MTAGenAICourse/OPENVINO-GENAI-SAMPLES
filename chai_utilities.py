# ChAI Utilities 
import os
import time
import requests
from io import BytesIO
import json

# Use truststore for system certificate store (handles enterprise certificates)
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    print("⚠️ truststore not available. SSL may fail with enterprise certificates.")

# Optional imports - not required for core ask_llm functionality
try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

try:
    import pygame
except ImportError:
    pygame = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    dotenv = None

# Check if the environment variables exist before deleting them

# Use CHAI_SDK_ASSISTANT from environment or default to Claude Sonnet 4.5
ChAI_assistant = os.environ.get("CHAI_SDK_ASSISTANT", "Claude Sonnet 4.5")

# Check for ChAI credentials
if "CHAI_SDK_CLIENT_SECRET" not in os.environ:
    print("Please add your ChAI SDK Client SECRET to continue.")
# Set ChAI API URL

#url_base = "http://ChAIAPI-dev.apps1-fm-htz.icloud.intel.com/"
#url_base = "http://ChAIAPI.apps1-fm-htz.icloud.intel.com/"
url_base = "https://chai-api.intel.com/"

url = url_base + "ask-ChAI"
chai_client_secret = os.environ.get("CHAI_SDK_CLIENT_SECRET", None)
if not chai_client_secret:
    print("Please set the ChAI SDK Client SECRET in your environment variables.")
    chai_client_secret = "your_chai_sdk_client_secret"  # Replace with your actual secret if needed



def manage_proxies(action="remove", saved_proxies=None):
    """
    Manage HTTP/HTTPS proxy settings.
    
    Args:
        action (str): Either "remove" or "restore" proxy settings
        saved_proxies (dict): Dictionary of saved proxy settings for restoration
        
    Returns:
        dict: Dictionary of saved proxy settings if removing
    """
    if action == "remove":
        # Initialize new dictionary to store proxy settings
        saved_proxies = {}
        
        # Save and remove proxy settings
        if "http_proxy" in os.environ:
            saved_proxies["http_proxy"] = os.environ["http_proxy"]
            print(f"Removing http_proxy: {saved_proxies['http_proxy']}")
            del os.environ["http_proxy"]
            
        if "https_proxy" in os.environ:
            saved_proxies["https_proxy"] = os.environ["https_proxy"]
            print(f"Removing https_proxy: {saved_proxies['https_proxy']}")
            del os.environ["https_proxy"]
            
        return saved_proxies
        
    elif action == "restore" and saved_proxies:
        # Restore proxy settings
        for key, value in saved_proxies.items():
            os.environ[key] = value
            print(f"Restored {key}: {value}")
    
    return {}






# Initialize the placeholder at the beginning of your app or in the relevant section

#print(f"Endpoint: {endpoint}, Deployment: {deployment}, Subscription Key: {subscription_key}, API Version: {api_version}")

def restore_proxies():
    global saved_http_proxy, saved_https_proxy

    #print(f"Saved HTTP Proxy: {saved_http_proxy}") #if 'saved_http_proxy' in locals() else ''}")
    os.environ["http_proxy"] = saved_http_proxy # if 'saved_http_proxy' in locals() else ''
    os.environ["https_proxy"] = saved_https_proxy #if 'saved_https_proxy' in locals() else ''
    os.environ["HTTP_PROXY"] = saved_http_proxy #if 'saved_http_proxy' in locals() else ''
    os.environ["HTTPS_PROXY"] = saved_https_proxy 


def ask_llm(prompt: str, ChAI_assistant, client_secret: str) -> str:
    """
    Send a query to the ChAI API and get a response.
    """
    # Make sure to wait for 1 second before sending the next request
    time.sleep(1)
    print(f'prompt: {prompt}')
    print(f'assistant: {ChAI_assistant}')
    print(f'client_secret: {client_secret}')
    # Print the http_proxy environment variable
    http_proxy = os.environ.get('http_proxy', 'Not set')
    print(http_proxy)
    # Print the https_proxy environment variable
    https_proxy = os.environ.get('https_proxy', 'Not set')
    print(https_proxy)
    input = {
        "queries": [prompt],
        "assistant_name": ChAI_assistant,
        "client_secret": client_secret
    }
    json_input = json.dumps(input)
    response = requests.post(url, json={"input": json_input})
    #manage_proxies('restore')  # Restore proxies after request
    if response.status_code == 200:
        response_json = response.json()
        if response_json.get('answers') and len(response_json['answers']) > 0:
            return response_json['answers'][0]
        else:
            print(f"LLM service returned empty answers array: {response_json}")
            return None
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
        return None