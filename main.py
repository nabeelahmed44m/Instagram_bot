import os
import random
import json
import base64
import requests
import io
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv
from google import genai
import openai
from config import LOCATIONS, DRESS_STYLES, SETTINGS

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BUFFER_API_KEY = os.getenv("BUFFER_API_KEY")
BUFFER_CHANNEL_ID = os.getenv("BUFFER_CHANNEL_ID")

gemini = genai.Client(api_key=GEMINI_API_KEY)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

REFERENCE_PHOTOS = ["1.png", "2.png", "3.png", "4.png"]

ACTIVITIES = [
    "walking casually along a path, hair flowing naturally, glancing back at the camera with a big candid smile",
    "sitting on stone steps, leaning forward with elbows on knees, laughing naturally with a big smile",
    "standing at a railing overlooking the view, one hand on the railing, smiling at the camera warmly",
    "walking through a busy street market, looking at items on display, smiling to herself happily",
    "sitting at an outdoor café table with a cup of chai, smiling warmly while looking at the camera",
    "leaning against an old wall with arms loosely crossed, smiling naturally at the camera",
    "holding a dupatta gently as wind blows it, laughing and smiling with joy",
    "walking down a staircase, one hand on the railing, caught mid-step with a bright smile",
    "sitting on a bench under a tree, smiling softly while looking off to the side",
    "standing and looking back over her shoulder at the camera with a playful smile",
]

BUFFER_API_URL = "https://api.buffer.com/graphql"
BUFFER_MUTATION = """
mutation createPost($input: CreatePostInput!) {
  createPost(input: $input) {
    ... on PostActionSuccess { post { id status } }
    ... on InvalidInputError { message }
    ... on UnexpectedError { message }
    ... on RestProxyError { message code }
    ... on LimitReachedError { message }
  }
}
"""


def load_all_reference_photos():
    refs = []
    for path in REFERENCE_PHOTOS:
        img = Image.open(path).convert("RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        refs.append((f"ref_{path}", buf.getvalue(), "image/png"))
    return refs


def generate_model_image(location, dress, setting, activity, ref_photos):
    prompt = (
    f"I am providing 4 reference photos of the same real person. "
    f"Generate a new photograph of this exact person while preserving her facial identity. "
    f"Use only the uploaded reference photos to determine her appearance. "
    f"Do not redesign, reinterpret, or alter her face in any way. "
    f"Keep her facial structure, eyes, eyebrows, nose, lips, jawline, hairline, hairstyle, skin tone, and age exactly the same as shown in the reference photos. "
    f"Only change the clothing, pose, and environment. "
    f"She is wearing {dress['description']}, {dress['color']} color with {dress['pattern']}. "
    f"She is {activity} at {location['name']}, Pakistan. "
    f"Time of day: {setting}. "
    f"The scene should look like a real everyday location with natural surroundings and authentic details. "
    f"The photograph should appear to have been taken casually on a modern smartphone. "
    f"Use natural lighting with realistic shadows and accurate colors. "
    f"Preserve natural skin texture with slight smartphone camera noise. "
    f"No beauty filters, skin smoothing, airbrushing, glamour retouching, studio lighting, HDR, cinematic color grading, lens flare, bloom, or artificial lighting effects. "
    f"The final image should be indistinguishable from a genuine smartphone photograph of the same person in a different place and outfit."
)

    result = openai_client.images.edit(
        model="gpt-image-1",
        image=ref_photos,
        prompt=prompt,
        size="1024x1536",
        quality="high",
        n=1,
    )

    img_bytes = base64.b64decode(result.data[0].b64_json)
    return img_bytes


def upload_image_to_fal(img_bytes, filename):
    import fal_client
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    try:
        # Convert to JPEG before uploading
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.save(tmp.name, format="JPEG", quality=95)
        url = fal_client.upload_file(tmp.name)
        return url
    finally:
        os.unlink(tmp.name)


def generate_caption(location, dress, setting):
    prompt = (
        f"Write an engaging Instagram caption for a fashion post.\n"
        f"A Pakistani woman is wearing {dress['description']} at {location['name']}, Pakistan during {setting}.\n\n"
        f"Rules:\n"
        f"- One single sentence only, aesthetic and poetic\n"
        f"- Add 1-2 relevant emojis naturally\n"
        f"- Use one Urdu word or phrase for authenticity\n"
        f"- End with 20-25 hashtags covering Pakistani fashion, travel and culture\n"
        f"- Must sound like a real person wrote it\n\n"
        f"Return only the caption, nothing else."
    )

    response = gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


def post_to_buffer(image_urls, caption):
    if not BUFFER_API_KEY or not BUFFER_CHANNEL_ID:
        print("Buffer credentials not set — skipping posting.")
        return

    assets = [{"image": {"url": url}} for url in image_urls]

    payload = {
        "query": BUFFER_MUTATION,
        "variables": {
            "input": {
                "channelId": BUFFER_CHANNEL_ID,
                "text": caption,
                "assets": assets,
                "schedulingType": "automatic",
                "mode": "shareNow",
                "metadata": {
                    "instagram": {
                        "type": "post",
                        "shouldShareToFeed": True,
                    }
                },
            }
        },
    }

    resp = requests.post(
        BUFFER_API_URL,
        json=payload,
        headers={"Authorization": f"Bearer {BUFFER_API_KEY}"},
    )
    resp.raise_for_status()

    result = resp.json()
    data = result.get("data", {}).get("createPost", {})

    if "post" in data:
        print(f"  Posted via Buffer! Post ID: {data['post']['id']} | Status: {data['post']['status']}")
    else:
        raise Exception(f"Buffer post failed: {data.get('message', result)}")


def main():
    location = random.choice(LOCATIONS)
    dress = random.choice(DRESS_STYLES)
    setting = random.choice(SETTINGS)

    print(f"\nToday's post:")
    print(f"  Location : {location['name']}")
    print(f"  Dress    : {dress['color']} — {dress['pattern']}")
    print(f"  Setting  : {setting}\n")

    ref_photos = load_all_reference_photos()
    activity1, activity2 = random.sample(ACTIVITIES, 2)

    print("Generating model image 1...")
    print(f"  Activity: {activity1}")
    img1_bytes = generate_model_image(location, dress, setting, activity1, ref_photos)
    print("  Uploading image 1...")
    url1 = upload_image_to_fal(img1_bytes, "image1.jpg")
    print(f"  Image 1: {url1}")

    print("\nGenerating model image 2...")
    print(f"  Activity: {activity2}")
    img2_bytes = generate_model_image(location, dress, setting, activity2, ref_photos)
    print("  Uploading image 2...")
    url2 = upload_image_to_fal(img2_bytes, "image2.jpg")
    print(f"  Image 2: {url2}")

    print("\nGenerating caption...")
    caption = generate_caption(location, dress, setting)
    print(f"\n--- CAPTION ---\n{caption}\n---------------\n")

    all_images = [url1, url2]

    with open("last_post.json", "w") as f:
        json.dump({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "location": location["name"],
            "dress": dress["description"],
            "setting": setting,
            "images": all_images,
            "caption": caption,
        }, f, indent=2)
    print("Saved to last_post.json")

    print("\nPosting to Instagram via Buffer...")
    post_to_buffer(all_images, caption)


if __name__ == "__main__":
    main()
