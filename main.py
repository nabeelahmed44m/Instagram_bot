import os
import random
import json
import requests
import io
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types
from config import LOCATIONS, DRESS_STYLES, SETTINGS

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BUFFER_API_KEY = os.getenv("BUFFER_API_KEY")
BUFFER_CHANNEL_ID = os.getenv("BUFFER_CHANNEL_ID")

gemini = genai.Client(api_key=GEMINI_API_KEY)

IMAGE_MODEL = "gemini-2.5-flash-image"

REFERENCE_PHOTOS = [
    "refs/ref_01.png",  # frontal neutral close-up, window light — core identity anchor
    "refs/ref_04.png",  # harsh midday sun, waist-up, dark kurta
    "refs/ref_07.png",  # right profile, cool morning light
    "refs/ref_08.png",  # night restaurant, warm low light
    "refs/ref_10.png",  # candid open laugh, messy bun
    "refs/ref_11.png",  # golden-hour frontal
    "refs/ref_14.png",  # three-quarter view, dramatic side shadow
    "refs/ref_20.png",  # hair fully tied back — hairline/ears/jaw anchor
]

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
    return [Image.open(path).convert("RGB") for path in REFERENCE_PHOTOS]


def generate_model_image(location, dress, setting, activity, ref_photos, outfit_ref=None):
    prompt = (
        f"I am providing {len(ref_photos)} reference photos of the same person. "
        f"Generate a new photograph of this exact person, preserving her facial identity precisely: "
        f"same facial structure, eyes, eyebrows, nose, lips, jawline, hairline, hairstyle, skin tone, and age as in the reference photos. "
        f"Only the clothing, pose, and environment change. "
        f"She is wearing {dress['description']}, {dress['color']} color with {dress['pattern']}. "
        f"She is {activity} at {location['name']}, Pakistan. "
        f"Time of day: {setting}. "
        f"A casual candid photo taken on an iPhone 16 Pro main camera in standard Photo mode, natural handheld framing, "
        f"framed so her face is clearly visible and detailed. "
        f"Render her skin the way a phone camera actually captures it: visible pores, faint vellus hair on the cheeks, "
        f"a slight natural oil sheen on the nose and forehead, subtle under-eye texture, a few flyaway hairs, "
        f"mildly uneven skin tone, and light smartphone sensor noise. "
        f"Natural ambient lighting with realistic shadows, slightly imperfect white balance, true-to-life colors. "
        f"The scene is a real everyday location with natural surroundings and authentic details."
    )

    contents = list(ref_photos)
    if outfit_ref is not None:
        prompt += (
            f" The very last image I provided is another photo from this exact same photoshoot: "
            f"the same person, in the same outfit, at the same location, at the same time of day. "
            f"Her dress must be exactly identical to that photo — same garment, same color, same pattern, "
            f"same fabric, same fit — and the location, weather, and lighting must match it too. "
            f"Only her pose, activity, and the camera angle are different."
        )
        contents.append(outfit_ref)
    contents.append(prompt)

    response = gemini.models.generate_content(
        model=IMAGE_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            image_config=types.ImageConfig(aspect_ratio="2:3"),
        ),
    )

    for part in response.candidates[0].content.parts:
        if getattr(part, "inline_data", None) is not None:
            return part.inline_data.data
    raise RuntimeError(f"Model returned no image: {response.candidates[0].content}")


def add_phone_realism(img):
    # Soften over-sharp AI edges and add faint sensor noise so the
    # image reads like a real smartphone photo
    w, h = img.size
    img = img.resize((int(w * 0.92), int(h * 0.92)), Image.LANCZOS)
    img = img.resize((w, h), Image.LANCZOS)
    noise = Image.effect_noise((w, h), 14).convert("RGB")
    return Image.blend(img, noise, 0.025)


def upload_image_to_fal(img_bytes, filename):
    import fal_client
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = add_phone_realism(img)
        # phone-typical JPEG compression, not pristine quality
        img.save(tmp.name, format="JPEG", quality=86)
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
    num_images = random.randint(2, 4)
    activities = random.sample(ACTIVITIES, num_images)
    print(f"  Images   : {num_images}\n")

    all_images = []
    outfit_ref = None
    for i, activity in enumerate(activities, 1):
        print(f"Generating model image {i}/{num_images}...")
        print(f"  Activity: {activity}")
        img_bytes = generate_model_image(location, dress, setting, activity, ref_photos, outfit_ref)
        if outfit_ref is None:
            # first image becomes the outfit/location reference for the rest
            outfit_ref = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        print(f"  Uploading image {i}...")
        url = upload_image_to_fal(img_bytes, f"image{i}.jpg")
        print(f"  Image {i}: {url}\n")
        all_images.append(url)

    print("Generating caption...")
    caption = generate_caption(location, dress, setting)
    print(f"\n--- CAPTION ---\n{caption}\n---------------\n")

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
