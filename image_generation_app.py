"""
Image Generation Studio - Streamlit Application
================================================
Interactive web application for generating AI images with guided prompt building.

Features:
- Guided prompt builder with pre-filled options
- Real-time prompt preview
- Advanced generation parameters
- Image gallery with download options
- Multilingual support
- Batch generation

Author: Gila Kamhi
Date: November 2025
"""

import streamlit as st
import time
from pathlib import Path
from PIL import Image
import os
from typing import List, Dict, Optional
import zipfile
import io

# Import the image generation service (with auto-detection)
from image_generation_service import (
    ImageGenerationService, 
    create_image_from_prompt,
    list_available_models,
    get_available_model_path
)

# ============================================================================
# Configuration & Constants
# ============================================================================

# Prompt template components
SUBJECTS = [
    "A friendly animal", "A magical creature", "A child learning", 
    "A wise teacher", "A playful character", "A brave explorer",
    "A curious scientist", "A kind robot", "A happy family",
    "Custom (enter below)"
]

ACTIONS = [
    "reading a book", "exploring nature", "learning math",
    "painting a picture", "playing with friends", "discovering something new",
    "solving a puzzle", "building something", "dancing joyfully",
    "teaching others", "Custom (enter below)"
]

SETTINGS = [
    "a cozy library", "a magical forest", "a colorful classroom",
    "a peaceful garden", "a bright playground", "a futuristic lab",
    "a warm home", "an enchanted castle", "a starry night sky",
    "a bustling city", "Custom (enter below)"
]

ART_STYLES = [
    "children's book illustration", "anime style", "watercolor painting",
    "digital art", "cartoon style", "photorealistic", "oil painting",
    "pencil sketch", "3D render", "pixel art", "Custom (enter below)"
]

CAMERA_ANGLES = [
    "eye level", "slight overhead", "low angle", "close-up",
    "wide shot", "medium shot", "bird's eye view", "worm's eye view",
    "Custom (enter below)"
]

LIGHTING = [
    "soft natural", "warm golden hour", "bright studio",
    "dramatic side", "gentle ambient", "magical glowing",
    "sunset", "moonlight", "candlelight", "Custom (enter below)"
]

MOODS = [
    "happy and cheerful", "calm and peaceful", "exciting and energetic",
    "mysterious and magical", "warm and cozy", "inspiring and uplifting",
    "playful and fun", "serene and tranquil", "Custom (enter below)"
]

COLOR_PALETTES = [
    "bright and vibrant", "soft pastels", "warm earth tones",
    "cool blues and purples", "rainbow colors", "monochromatic",
    "golden and amber", "natural greens", "Custom (enter below)"
]

TIME_WEATHER = [
    "sunny day", "golden sunset", "starry night", "cloudy afternoon",
    "gentle rain", "spring morning", "autumn evening", "winter wonderland",
    "clear day", "Custom (enter below)"
]

QUALITY_STYLES = [
    "highly detailed", "professionally composed", "award-winning",
    "masterpiece quality", "8k resolution", "ultra detailed",
    "sharp focus", "cinematic", "Custom (enter below)"
]

# Device options
DEVICES = ["GPU", "CPU", "NPU"]

# Languages
LANGUAGES = [
    "English", "Spanish", "Hebrew", "French", "German", "Italian",
    "Portuguese", "Russian", "Chinese", "Japanese", "Korean", "Arabic"
]

# ============================================================================
# Utility Functions
# ============================================================================

def build_prompt_from_components(
    subject: str,
    action: str,
    setting: str,
    art_style: str,
    camera_angle: str,
    lighting: str,
    mood: str,
    color_palette: str,
    time_weather: str,
    quality: str,
    custom_additions: str = ""
) -> str:
    """
    Build a structured prompt from components.
    
    Returns:
        Formatted prompt string
    """
    prompt_parts = []
    
    # Subject and action
    if subject and action:
        prompt_parts.append(f"{subject} {action}")
    elif subject:
        prompt_parts.append(subject)
    
    # Setting
    if setting:
        prompt_parts.append(f"in {setting}")
    
    # Art style
    if art_style:
        prompt_parts.append(f"{art_style}")
    
    # Camera angle
    if camera_angle:
        prompt_parts.append(f"{camera_angle}")
    
    # Lighting
    if lighting:
        prompt_parts.append(f"{lighting} lighting")
    
    # Mood
    if mood:
        prompt_parts.append(f"{mood} atmosphere")
    
    # Color palette
    if color_palette:
        prompt_parts.append(f"{color_palette} color palette")
    
    # Time/Weather
    if time_weather:
        prompt_parts.append(time_weather)
    
    # Quality descriptors
    base_quality = "professional composition, sharp focus, highly detailed"
    if quality:
        prompt_parts.append(f"{quality}, {base_quality}")
    else:
        prompt_parts.append(base_quality)
    
    # Custom additions
    if custom_additions:
        prompt_parts.append(custom_additions)
    
    return ", ".join(prompt_parts)


def get_model_name_from_path(model_path: str) -> str:
    """Extract a friendly model name from the path."""
    if not model_path:
        return "Unknown Model"
    
    path_lower = model_path.lower()
    
    if "stable-diffusion" in path_lower:
        return "Stable Diffusion"
    elif "dreamlike" in path_lower and "anime" in path_lower:
        return "Dreamlike Anime"
    elif "dreamlike" in path_lower:
        return "Dreamlike"
    else:
        # Extract last directory name
        return os.path.basename(model_path.rstrip('/\\'))


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'generated_images' not in st.session_state:
        st.session_state.generated_images = []
    
    if 'service' not in st.session_state:
        st.session_state.service = None
    
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    
    if 'available_models' not in st.session_state:
        st.session_state.available_models = list_available_models()


def get_or_create_service(model_dir: Optional[str], device: str, output_dir: str) -> ImageGenerationService:
    """Get existing service or create new one if parameters changed."""
    if (st.session_state.service is None or 
        st.session_state.get('last_device') != device or
        st.session_state.get('last_model_dir') != model_dir):
        
        with st.spinner(f"Loading model on {device}..."):
            st.session_state.service = ImageGenerationService(
                model_dir=model_dir,  # Can be None for auto-detection
                device=device,
                output_dir=output_dir
            )
            st.session_state.last_device = device
            st.session_state.last_model_dir = st.session_state.service.model_dir  # Store actual path used
    
    return st.session_state.service


def create_download_zip(image_paths: List[str]) -> bytes:
    """Create a ZIP file containing multiple images."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for img_path in image_paths:
            if os.path.exists(img_path):
                zip_file.write(img_path, os.path.basename(img_path))
    
    return zip_buffer.getvalue()


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Image Generation Studio",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🎨 Image Generation Studio")
    st.markdown("### Create stunning AI-generated images with guided prompt building")
    
    # Check for available models
    if not st.session_state.available_models:
        st.error(
            "❌ **No image generation models found!**\n\n"
            "Please ensure models are in the parent directory (..):\n"
            "- `../stable-diffusion-ov`\n"
            "- `../dreamlike_anime_1_0_ov/FP16`\n\n"
            "See the documentation for setup instructions."
        )
        st.stop()
    
    # Display available models
    with st.expander("📦 Available Models", expanded=False):
        for name, path in st.session_state.available_models:
            st.success(f"✅ **{name}**\n`{path}`")
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model settings
        st.subheader("Model Settings")
        
        # Model selection
        model_options = ["Auto-detect (Recommended)"] + [
            f"{name}" for name, _ in st.session_state.available_models
        ]
        
        selected_model = st.selectbox(
            "Model",
            options=model_options,
            index=0,
            help="Select a model or use auto-detection"
        )
        
        # Get actual model path
        if selected_model == "Auto-detect (Recommended)":
            model_dir = None  # Will auto-detect
            st.info("🔍 Model will be auto-detected from available options")
        else:
            # Find the selected model's path
            for name, path in st.session_state.available_models:
                if name == selected_model:
                    model_dir = path
                    break
        
        device = st.selectbox(
            "Device",
            options=DEVICES,
            index=0,
            help="Hardware device for inference (GPU recommended)"
        )
        
        output_dir = st.text_input(
            "Output Directory",
            value="generated_images",
            help="Directory to save generated images"
        )
        
        st.divider()
        
        # Generation parameters
        st.subheader("Generation Parameters")
        
        use_advanced = st.checkbox(
            "Use Advanced Mode",
            value=True,
            help="Enable advanced generation parameters"
        )
        
        if use_advanced:
            num_inference_steps = st.slider(
                "Inference Steps",
                min_value=10,
                max_value=100,
                value=50,
                step=5,
                help="Higher = better quality but slower"
            )
            
            guidance_scale = st.slider(
                "Guidance Scale",
                min_value=1.0,
                max_value=20.0,
                value=7.5,
                step=0.5,
                help="How closely to follow the prompt (7.5 is typical)"
            )
            
            enhance_prompt = st.checkbox(
                "Auto-enhance Prompt",
                value=True,
                help="Automatically add quality enhancement phrases"
            )
        else:
            num_inference_steps = st.slider(
                "Inference Steps",
                min_value=10,
                max_value=50,
                value=20,
                step=5
            )
            guidance_scale = None
            enhance_prompt = False
        
        # Image dimensions
        st.subheader("Image Dimensions")
        col1, col2 = st.columns(2)
        with col1:
            width = st.selectbox(
                "Width",
                options=[256, 384, 512, 640, 768, 1024],
                index=2
            )
        with col2:
            height = st.selectbox(
                "Height",
                options=[256, 384, 512, 640, 768, 1024],
                index=2
            )
        
        st.divider()
        
        # Multilingual support
        st.subheader("🌍 Language Support")
        use_translation = st.checkbox("Enable Translation")
        
        if use_translation:
            prompt_language = st.selectbox(
                "Prompt Language",
                options=LANGUAGES,
                index=0
            )
        else:
            prompt_language = "English"
    
    # Main content area - Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Prompt Builder",
        "✍️ Free Prompt",
        "🖼️ Gallery",
        "📊 History"
    ])
    
    # ========================================================================
    # Tab 1: Prompt Builder
    # ========================================================================
    with tab1:
        st.header("Guided Prompt Builder")
        st.markdown("Build your prompt step by step using the components below.")
        
        # Prompt components in columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Subject
            subject_option = st.selectbox("👤 Subject", options=SUBJECTS)
            if subject_option == "Custom (enter below)":
                subject = st.text_input("Custom Subject", placeholder="e.g., A curious panda")
            else:
                subject = subject_option
            
            # Action
            action_option = st.selectbox("🎬 Action/Pose", options=ACTIONS)
            if action_option == "Custom (enter below)":
                action = st.text_input("Custom Action", placeholder="e.g., juggling colorful balls")
            else:
                action = action_option
            
            # Setting
            setting_option = st.selectbox("📍 Setting/Location", options=SETTINGS)
            if setting_option == "Custom (enter below)":
                setting = st.text_input("Custom Setting", placeholder="e.g., a bamboo forest")
            else:
                setting = setting_option
            
            # Art Style
            art_style_option = st.selectbox("🎨 Art Style", options=ART_STYLES)
            if art_style_option == "Custom (enter below)":
                art_style = st.text_input("Custom Art Style", placeholder="e.g., impressionist painting")
            else:
                art_style = art_style_option
            
            # Camera Angle
            camera_option = st.selectbox("📷 Camera Angle", options=CAMERA_ANGLES)
            if camera_option == "Custom (enter below)":
                camera_angle = st.text_input("Custom Camera Angle", placeholder="e.g., dynamic angle")
            else:
                camera_angle = camera_option
        
        with col2:
            # Lighting
            lighting_option = st.selectbox("💡 Lighting", options=LIGHTING)
            if lighting_option == "Custom (enter below)":
                lighting = st.text_input("Custom Lighting", placeholder="e.g., bioluminescent glow")
            else:
                lighting = lighting_option
            
            # Mood
            mood_option = st.selectbox("😊 Mood/Atmosphere", options=MOODS)
            if mood_option == "Custom (enter below)":
                mood = st.text_input("Custom Mood", placeholder="e.g., whimsical and dreamy")
            else:
                mood = mood_option
            
            # Color Palette
            color_option = st.selectbox("🎨 Color Palette", options=COLOR_PALETTES)
            if color_option == "Custom (enter below)":
                color_palette = st.text_input("Custom Color Palette", placeholder="e.g., turquoise and coral")
            else:
                color_palette = color_option
            
            # Time/Weather
            time_option = st.selectbox("🌤️ Time/Weather", options=TIME_WEATHER)
            if time_option == "Custom (enter below)":
                time_weather = st.text_input("Custom Time/Weather", placeholder="e.g., misty morning")
            else:
                time_weather = time_option
            
            # Quality
            quality_option = st.selectbox("⭐ Quality/Style", options=QUALITY_STYLES)
            if quality_option == "Custom (enter below)":
                quality = st.text_input("Custom Quality", placeholder="e.g., trending on artstation")
            else:
                quality = quality_option
        
        # Custom additions
        st.subheader("✨ Additional Details")
        custom_additions = st.text_area(
            "Add any extra details",
            placeholder="e.g., with sparkles, wearing a tiny hat, surrounded by butterflies",
            height=100
        )
        
        # Negative prompt
        with st.expander("🚫 Negative Prompt (Advanced)", expanded=False):
            custom_negative = st.text_area(
                "Things to avoid",
                placeholder="blurry, low quality, distorted, dark...",
                height=100
            )
        
        # Build and preview prompt
        st.divider()
        
        built_prompt = build_prompt_from_components(
            subject, action, setting, art_style, camera_angle,
            lighting, mood, color_palette, time_weather, quality,
            custom_additions
        )
        
        st.subheader("📝 Generated Prompt Preview")
        st.info(built_prompt)
        
        # Generation controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            generate_btn = st.button("🎨 Generate Image", type="primary", width="stretch")
        
        with col2:
            num_variations = st.number_input("Variations", min_value=1, max_value=5, value=1)
        
        with col3:
            child_friendly = st.checkbox("Child-Friendly")
        
        # Generate images
        if generate_btn:
            try:
                service = get_or_create_service(model_dir, device, output_dir)
                
                # Get the actual model being used
                actual_model_path = service.model_dir
                actual_model_name = get_model_name_from_path(actual_model_path)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                generated_batch = []
                
                for i in range(num_variations):
                    status_text.text(f"Generating variation {i+1}/{num_variations}...")
                    
                    if child_friendly:
                        image, path = service.generate_child_friendly_image(
                            prompt=built_prompt,
                            age_group="5-7",
                            width=width,
                            height=height,
                            num_inference_steps=num_inference_steps
                        )
                    elif use_advanced:
                        negative = custom_negative if custom_negative else None
                        image, path = service.generate_image_advanced(
                            prompt=built_prompt,
                            negative_prompt=negative,
                            width=width,
                            height=height,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            enhance_prompt=enhance_prompt
                        )
                    else:
                        image, path = service.generate_image_basic(
                            prompt=built_prompt,
                            width=width,
                            height=height,
                            num_inference_steps=num_inference_steps
                        )
                    
                    generated_batch.append({
                        'image': image,
                        'path': path,
                        'prompt': built_prompt,
                        'timestamp': time.time(),
                        'model_name': actual_model_name,
                        'model_path': actual_model_path,
                        'device': device,
                        'dimensions': f"{width}x{height}",
                        'steps': num_inference_steps,
                        'guidance_scale': guidance_scale if use_advanced else None
                    })
                    
                    progress_bar.progress((i + 1) / num_variations)
                
                # Store in session state
                st.session_state.generated_images.extend(generated_batch)
                st.session_state.generation_history.append({
                    'prompt': built_prompt,
                    'count': num_variations,
                    'timestamp': time.time(),
                    'paths': [item['path'] for item in generated_batch],
                    'model_name': actual_model_name,
                    'model_path': actual_model_path,
                    'device': device,
                    'dimensions': f"{width}x{height}",
                    'steps': num_inference_steps,
                    'guidance_scale': guidance_scale if use_advanced else None
                })
                
                status_text.text("✅ Generation complete!")
                st.success(f"Successfully generated {num_variations} image(s) using **{actual_model_name}**!")
                
                # Display generated images
                st.subheader("Generated Images")
                cols = st.columns(min(num_variations, 3))
                for idx, item in enumerate(generated_batch):
                    with cols[idx % 3]:
                        st.image(item['image'], width="stretch")
                        st.caption(f"🤖 Model: {actual_model_name}")
                        st.download_button(
                            "⬇️ Download",
                            data=open(item['path'], 'rb').read(),
                            file_name=os.path.basename(item['path']),
                            mime="image/png",
                            key=f"download_builder_{idx}_{item['timestamp']}"
                        )
                
            except Exception as e:
                st.error(f"❌ Generation failed: {str(e)}")
                with st.expander("Error Details"):
                    st.exception(e)
    
    # ========================================================================
    # Tab 2: Free Prompt
    # ========================================================================
    with tab2:
        st.header("Free Prompt Mode")
        st.markdown("Enter your own custom prompt directly.")
        
        free_prompt = st.text_area(
            "Enter your prompt",
            placeholder="Describe the image you want to generate...",
            height=150,
            key="free_prompt_input"
        )
        
        with st.expander("🚫 Negative Prompt", expanded=False):
            free_negative = st.text_area(
                "Things to avoid",
                placeholder="blurry, low quality, distorted...",
                height=100,
                key="free_negative_input"
            )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            free_generate_btn = st.button("🎨 Generate", type="primary", width="stretch", key="free_gen_btn")
        with col2:
            free_child_friendly = st.checkbox("Child-Friendly", key="free_child")
        
        if free_generate_btn and free_prompt:
            try:
                service = get_or_create_service(model_dir, device, output_dir)
                
                # Get the actual model being used
                actual_model_path = service.model_dir
                actual_model_name = get_model_name_from_path(actual_model_path)
                
                with st.spinner(f"Generating your image using {actual_model_name}..."):
                    if free_child_friendly:
                        image, path = service.generate_child_friendly_image(
                            prompt=free_prompt,
                            width=width,
                            height=height,
                            num_inference_steps=num_inference_steps
                        )
                    elif use_advanced:
                        negative = free_negative if free_negative else None
                        image, path = service.generate_image_advanced(
                            prompt=free_prompt,
                            negative_prompt=negative,
                            width=width,
                            height=height,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            enhance_prompt=enhance_prompt
                        )
                    else:
                        image, path = service.generate_image_basic(
                            prompt=free_prompt,
                            width=width,
                            height=height,
                            num_inference_steps=num_inference_steps
                        )
                    
                    result = {
                        'image': image,
                        'path': path,
                        'prompt': free_prompt,
                        'timestamp': time.time(),
                        'model_name': actual_model_name,
                        'model_path': actual_model_path,
                        'device': device,
                        'dimensions': f"{width}x{height}",
                        'steps': num_inference_steps,
                        'guidance_scale': guidance_scale if use_advanced else None
                    }
                    
                    st.session_state.generated_images.append(result)
                    st.session_state.generation_history.append({
                        'prompt': free_prompt,
                        'count': 1,
                        'timestamp': time.time(),
                        'paths': [path],
                        'model_name': actual_model_name,
                        'model_path': actual_model_path,
                        'device': device,
                        'dimensions': f"{width}x{height}",
                        'steps': num_inference_steps,
                        'guidance_scale': guidance_scale if use_advanced else None
                    })
                    
                    st.success(f"✅ Image generated successfully using **{actual_model_name}**!")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.image(image, width="stretch")
                    with col2:
                        st.info(f"**Model:** {actual_model_name}\n\n**Device:** {device}\n\n**Size:** {width}x{height}\n\n**Steps:** {num_inference_steps}")
                        st.download_button(
                            "⬇️ Download Image",
                            data=open(path, 'rb').read(),
                            file_name=os.path.basename(path),
                            mime="image/png"
                        )
            
            except Exception as e:
                st.error(f"❌ Generation failed: {str(e)}")
                with st.expander("Error Details"):
                    st.exception(e)
    
    # ========================================================================
    # Tab 3: Gallery
    # ========================================================================
    with tab3:
        st.header("Image Gallery")
        
        if not st.session_state.generated_images:
            st.info("No images generated yet. Use the Prompt Builder or Free Prompt tab to create images.")
        else:
            st.markdown(f"**Total Images:** {len(st.session_state.generated_images)}")
            
            # Download all button
            if len(st.session_state.generated_images) > 1:
                all_paths = [item['path'] for item in st.session_state.generated_images]
                zip_data = create_download_zip(all_paths)
                st.download_button(
                    "📦 Download All Images (ZIP)",
                    data=zip_data,
                    file_name="generated_images.zip",
                    mime="application/zip"
                )
            
            st.divider()
            
            # Display gallery
            cols_per_row = 3
            for idx in range(0, len(st.session_state.generated_images), cols_per_row):
                cols = st.columns(cols_per_row)
                for col_idx, col in enumerate(cols):
                    img_idx = idx + col_idx
                    if img_idx < len(st.session_state.generated_images):
                        item = st.session_state.generated_images[img_idx]
                        with col:
                            st.image(item['image'], width="stretch")
                            with st.expander("📋 Details"):
                                st.caption(f"**Prompt:** {item['prompt'][:100]}...")
                                st.caption(f"**Model:** {item.get('model_name', 'Unknown')}")
                                st.caption(f"**Device:** {item.get('device', 'Unknown')}")
                                st.caption(f"**Dimensions:** {item.get('dimensions', 'Unknown')}")
                                st.caption(f"**Steps:** {item.get('steps', 'Unknown')}")
                                if item.get('guidance_scale'):
                                    st.caption(f"**Guidance Scale:** {item.get('guidance_scale')}")
                                st.caption(f"**Time:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(item['timestamp']))}")
                                st.caption(f"**Path:** `{item.get('model_path', 'Unknown')[:60]}...`")
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.download_button(
                                    "⬇️ Download",
                                    data=open(item['path'], 'rb').read(),
                                    file_name=os.path.basename(item['path']),
                                    mime="image/png",
                                    key=f"gallery_download_{img_idx}"
                                )
                            with col_b:
                                if st.button("🗑️ Delete", key=f"gallery_delete_{img_idx}"):
                                    try:
                                        if os.path.exists(item['path']):
                                            os.remove(item['path'])
                                        st.session_state.generated_images.pop(img_idx)
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Delete failed: {e}")
            
            # Clear all button
            st.divider()
            if st.button("🗑️ Clear All Images", type="secondary"):
                for item in st.session_state.generated_images:
                    try:
                        if os.path.exists(item['path']):
                            os.remove(item['path'])
                    except:
                        pass
                st.session_state.generated_images = []
                st.rerun()
    
    # ========================================================================
    # Tab 4: History
    # ========================================================================
    with tab4:
        st.header("Generation History")
        
        if not st.session_state.generation_history:
            st.info("No generation history yet.")
        else:
            st.markdown(f"**Total Generations:** {len(st.session_state.generation_history)}")
            
            for idx, entry in enumerate(reversed(st.session_state.generation_history)):
                with st.expander(
                    f"🎨 Generation {len(st.session_state.generation_history) - idx}: "
                    f"{entry['count']} image(s) - "
                    f"{entry.get('model_name', 'Unknown Model')} - "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry['timestamp']))}"
                ):
                    # Generation details
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.text_area("Prompt", value=entry['prompt'], height=100, disabled=True, key=f"hist_prompt_{idx}")
                    
                    with col2:
                        st.markdown("**Generation Settings:**")
                        st.caption(f"🤖 Model: {entry.get('model_name', 'Unknown')}")
                        st.caption(f"💻 Device: {entry.get('device', 'Unknown')}")
                        st.caption(f"📐 Dimensions: {entry.get('dimensions', 'Unknown')}")
                        st.caption(f"🔄 Steps: {entry.get('steps', 'Unknown')}")
                        if entry.get('guidance_scale'):
                            st.caption(f"🎯 Guidance: {entry.get('guidance_scale')}")
                        st.caption(f"🖼️ Count: {entry['count']} image(s)")
                    
                    # Model path
                    if entry.get('model_path'):
                        st.caption(f"📂 Model Path: `{entry['model_path']}`")
                    
                    st.divider()
                    
                    # Display thumbnails
                    if entry['paths']:
                        st.markdown("**Generated Images:**")
                        cols = st.columns(min(len(entry['paths']), 4))
                        for img_idx, path in enumerate(entry['paths']):
                            if os.path.exists(path):
                                with cols[img_idx % 4]:
                                    img = Image.open(path)
                                    st.image(img, width="stretch")
                                    st.download_button(
                                        "⬇️",
                                        data=open(path, 'rb').read(),
                                        file_name=os.path.basename(path),
                                        mime="image/png",
                                        key=f"hist_download_{idx}_{img_idx}"
                                    )


# Run the application if executed directly
if __name__ == "__main__":
    main()