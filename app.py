import os
import streamlit as st
from clarifai.client.model import Model
import base64
from dotenv import load_dotenv

load_dotenv()

CLARIFAI_PAT = os.getenv("CLARIFAI_PAT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def generate_image(user_description, api_key):
    prompt = f"You are a professional medical doctor. Based on the below user's description and content," \
             f" create a proper visualization to enable your patient understand your diagnosis: {user_description}"
    inference_params = dict(quality="standard", size="1792x1024")
    model_prediction = Model(f"https://clarifai.com/openai/dall-e/models/dall-e-3?api_key={api_key}") \
        .predict_by_bytes(
        prompt.encode(), input_type="text", inference_params=inference_params
    )
    output_base64 = model_prediction.outputs[0].data.image.base64
    with open("generated_image.png", "wb") as file:
        file.write(output_base64)
    return "generated_image.png"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def understand_image(base64_image, api_key):
    prompt = "Analyze the content of this image and write an informative, educative description of the diagnosis" \
             " given. Describe the tools used, the methods of procedure that would  make the patient to understand" \
             " the message fully"
    inference_params = dict(temperature=0.2, image_base64=base64_image, api_key=api_key)
    model_prediction = Model(f"https://clarifai.com/openai/chat-completion/models/gpt-4-vision") \
        .predict_by_bytes(
        prompt.encode(), input_type="text", inference_params=inference_params
    )
    return model_prediction.outputs[0].data.text.raw


# from clarifai.client.input import Inputs
# prompt = "Explain this picture as though you were a medical doctor talking to a patient"
# image_url = "https://samples.clarifai.com/metro-north.jpg"
# inference_params = dict(temperature=0.2, max_tokens=100)
#
# model_prediction = Model("https://clarifai.com/openai/chat-completion/models/openai-gpt-4-vision")
# .predict(inputs = [Inputs.get_multimodal_input(input_id="",image_url=image_url, raw_text=prompt)]
# ,inference_params=inference_params)
#
# print(model_prediction.outputs[0].data.text.raw)


def text_to_speech(input_text, api_key):
    inference_params = dict(voice='alloy', speed=1.0, api_key=api_key)
    model_prediction = Model(f"https://clarifai.com/openai/tts/models/openai-tts-1") \
        .predict_by_bytes(
        input_text.encode(), input_type="text", inference_params=inference_params
    )
    audio_base64 = model_prediction.outputs[0].data.audio.base64
    return audio_base64


def main():
    st.set_page_config(page_title="Interactive Media Creator", layout="wide")
    st.title("Interactive Media Creator")

    with st.sidebar:
        st.header("Controls")
        image_description = st.text_area("Description for Image Generation", height=100)
        generate_image_btn = st.button("Generate Image")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Comic Art")
        if generate_image_btn and image_description:
            with st.spinner("Generating image..."):
                image_path = generate_image(user_description=image_description, api_key=CLARIFAI_PAT)
                if image_path:
                    st.image(
                        image_path,
                        caption="Generated Comic Image",
                        use_column_width=True,
                    )
                    st.success("Image generated!")
                else:
                    st.error("Failed to generate image.")

    with col2:
        st.header("Story")
        if generate_image_btn and image_description:
            with st.spinner("Creating a story..."):
                base64_image = encode_image(image_path=image_path)
                understood_text = understand_image(base64_image=base64_image, api_key=OPENAI_API_KEY)
                audio_base64 = text_to_speech(input_text=understood_text, api_key=OPENAI_API_KEY)
                st.audio(audio_base64, format="audio/mp3")
                st.success("Audio generated from image understanding!")


if __name__ == "__main__":
    main()
