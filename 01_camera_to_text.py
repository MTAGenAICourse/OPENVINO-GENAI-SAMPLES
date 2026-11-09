import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from PIL import Image
import streamlit as st
import io
import time

def initialize_session():
    """Initialize the session state for model and history."""
    if "model_loaded" not in st.session_state:
        st.session_state["model_loaded"] = False
    
    if "pipe" not in st.session_state:
        st.session_state["pipe"] = None
        
    if "history" not in st.session_state:
        st.session_state["history"] = []

def load_model(model_path, device="GPU"):
    """Load the OpenVINO model."""
    with st.spinner("Loading AI model... This may take a moment."):
        try:
            pipeline = ov_genai.VLMPipeline(str(model_path), device)
            # Phi-3.5 ends its turn with <|end|> (token 32007), but the exported
            # generation_config.json only lists </s>: stop there, or it keeps writing.
            _cfg = pipeline.get_generation_config()
            _cfg.stop_token_ids = {32007}
            pipeline.set_generation_config(_cfg)
            st.session_state["model_loaded"] = True
            st.session_state["pipe"] = pipeline
            return pipeline
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return None

def process_image(image):
    """Process image for model input."""
    # Resize large images to prevent memory issues
    max_size = 1024
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
        
    # Convert to OpenVINO tensor
    image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.uint8)
    return ov.Tensor(image_data)


# Revise answer
prompt_template = """ You are a helpful image analysis assistant. Please provide an answer to user's question {prompt}. Focus only on what's asked and avoid unrelated content or follow-up questions. Be precise in your response and do not add any content or questions that are not related to user query and do not cut the response in the middle of the statement. Please do provide answers related to the image only.
Please do not add any content or questions that are not related to user query and do not cut the response in the middle of the statement. Please do provide answers related to the image only. 
When asked to describe the image, ONLY answer in the context of the image and  do not end with any broken statements or formats that use special characters. 
A sample response may be in the style of -  the image shows a beautiful sunset over a mountain range, with vibrant colors in the sky and a serene landscape - in case that the image is a sunset over a mountain range.
prompt : {prompt}
Revised prompt : """


def generate_description(pipe, image_tensor, prompt_text):
    """Generate description from image using model."""
    """Generate description from image using model."""
    # Add system prompt to guide the model
    system_prompt = "You are a helpful  image analysis assistant. Give direct answers to questions about images. Focus only on what's asked and avoid unrelated content or follow-up questions.Be precise in your response and do not add any content or questions that are not related to user queryand do not cut the response in the middle of the statement. Please do provide answers related to the image only"
    
    # Combine system prompt and user prompt
    #template = "<|im_start|>system\n{}\n<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n"
    template = "<|im_start|>system\n{}\n<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\nAnalyzing the image based on your question: "
    formatted_prompt = template.format(system_prompt, prompt_text)
    prompt = prompt_template.format(prompt=prompt_text)
    #prompt = template.format(system_prompt, prompt_text)
    #prompt = system_prompt +  prompt_text
   
    
    with st.spinner("Generating response..."):
        try:
            start_time = time.time()
            
            # Make prompt more focused and direct

            focused_prompt = prompt #f"Analyze this image and {prompt.strip()}. Provide only relevant details without any how-to sections or additional questions."
            #result = str(pipe.generate(focused_prompt, image=image_tensor, max_new_tokens=150))
            result = str(pipe.generate(formatted_prompt, image=image_tensor, max_new_tokens=150))

            # Clean up response by removing any "How to" sections
            result = result.split("How to")[0].strip()
             # Clean up response by removing any "How to" sections
            result = result.split("Assistant:")[0].strip()
            result = result.split("User:")[0].strip()
            result = result.split("#")[0].strip()
            generation_time = time.time() - start_time
            return result, generation_time
        except Exception as e:
            st.error(f"Error generating response : {str(e)}")
            return None, 0

def main():
    st.title("Camera to Text Generation")
    st.subheader("Take a photo and get an AI response to user query")
    
    # Initialize session
    initialize_session()
    
    # Sidebar for model settings
    with st.sidebar:
        st.header("Model Settings")
        model_path = st.text_input("Model Path", value="./Phi-3.5-vision-instruct-int4-ov/")
        device = st.radio("Device", ["GPU", "CPU"], index=0)
        
        if not st.session_state["model_loaded"]:
            if st.button("Load Model"):
                load_model(model_path, device)
        else:
            st.success("Model loaded successfully!")
            
            # Add option to reload model with different settings
            if st.button("Reload Model"):
                st.session_state["model_loaded"] = False
                st.experimental_rerun()
    
    # Main content area
    if not st.session_state["model_loaded"]:
        st.warning("Please load the model first using the sidebar options.")
        return
    
    # Camera input
    camera_col, preview_col = st.columns(2)
    
    with camera_col:
        # Camera input widget
        camera_image = st.camera_input("Take a photo")
    
    # Process image if available
    if camera_image is not None:
        with preview_col:
            st.image(camera_image, caption="Captured Image", use_container_width=True)
        
        # Process the image
        image = Image.open(io.BytesIO(camera_image.getvalue()))
        image_tensor = process_image(image)
        
        # Get prompt from user
        prompt_text = st.text_input("Enter your prompt:", value="Can you describe this image?")
        
        # Generate description when button is clicked
        if st.button("Generate"):
            result, generation_time = generate_description(
                st.session_state["pipe"], 
                image_tensor, 
                prompt_text
            )
            
            if result:
                # Display result
                st.success(f"Generated in {generation_time:.2f} seconds")
                st.markdown("### AI Response")
                st.markdown(result)
                
                # Add to history
                entry = {
                    "image": camera_image,
                    "prompt": prompt_text,
                    "description": result,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state["history"].insert(0, entry)
    
    # Display history
    if st.session_state["history"]:
        st.markdown("---")
        st.markdown("## History")
        
        for i, entry in enumerate(st.session_state["history"]):
            with st.expander(f"Entry {i+1} - {entry['timestamp']}"):
                cols = st.columns(2)
                with cols[0]:
                    st.image(entry["image"], caption="Captured Image", use_container_width=True)
                with cols[1]:
                    st.markdown(f"**Prompt:** {entry['prompt']}")
                    st.markdown(f"**Description:** {entry['description']}")

if __name__ == "__main__":
    main()