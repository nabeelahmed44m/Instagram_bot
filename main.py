import os
import random
import json
import requests
import io
from PIL import Image, ImageEnhance
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types
from config import LOCATIONS, DRESS_STYLES, SETTINGS, CAROUSEL_SLIDES

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
    "refs/ref_fullbody_01.png",  # full-body frontal, plain background — body proportion anchor
    "refs/ref_fullbody_02.png",  # full-body 45-degree stance — height/build anchor
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


def generate_model_image(location, dress, setting, slide, pose, ref_photos, outfit_ref=None):
    prompt = (
        f"I am providing {len(ref_photos)} reference photos of the same person. "
        f"Generate a new photograph of this exact person, preserving her facial identity precisely: "
        f"same facial structure, eyes, eyebrows, nose, lips, jawline, hairline, hairstyle, skin tone, and age as in the reference photos. "
        f"Her body proportions, height and build must match the full-body reference photos. "
        f"Only the clothing, pose, and environment change. "
        f"She is wearing {dress['description']}, {dress['color']} color with {dress['pattern']}. "
        f"She is {pose}, at {location['name']}, Pakistan. "
        f"Time of day: {setting}. "
        f"A photo taken on an iPhone 16 Pro main camera in standard Photo mode, "
        f"{slide['framing']}, her face still clearly visible and recognizable. "
        f"CRITICAL for realism — this must look like a casual photo a friend took on a phone, never an editorial or fashion shoot: "
        f"deep depth of field with the background almost fully in focus, absolutely no cinematic bokeh or creamy background blur; "
        f"slightly imperfect composition with the subject a little off-center, framed a bit rushed like a real snapshot; "
        f"lighting must behave like the real time of day — hard cast shadows on walls and ground in sunlight, "
        f"mildly blown highlights, a faint smartphone HDR haze. "
        f"True-to-life mixed colors with slightly imperfect white balance — never uniformly warm, never color graded. "
        f"Render her skin the way a phone camera actually captures it: visible pores, faint vellus hair on the cheeks, "
        f"a slight natural oil sheen on the nose and forehead, subtle under-eye texture, a few flyaway hairs, "
        f"mildly uneven skin tone, and light smartphone sensor noise. "
        f"The backdrop is one simple authentic corner of the location — a textured wall, a plant, a doorway, a chair — "
        f"with real imperfect details, not a sweeping postcard vista."
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


def apply_feed_grade(img):
    # Fixed warm Portra-style grade applied identically to every image so
    # the whole grid reads as one consistently edited feed
    r, g, b = img.split()
    r = r.point(lambda v: min(255, int(v * 1.02 + 1)))
    b = b.point(lambda v: int(v * 0.98))
    img = Image.merge("RGB", (r, g, b))
    # lift shadows slightly for the soft filmic look
    img = img.point(lambda v: int(v + 10 * ((1 - v / 255) ** 2)))
    img = ImageEnhance.Color(img).enhance(0.97)
    img = ImageEnhance.Brightness(img).enhance(1.02)
    return img


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
        img = apply_feed_grade(img)
        img = add_phone_realism(img)
        # phone-typical JPEG compression, not pristine quality
        img.save(tmp.name, format="JPEG", quality=86)
        url = fal_client.upload_file(tmp.name)
        return url
    finally:
        os.unlink(tmp.name)


def generate_caption(location, dress, setting):
    prompt = (
        f"Write an Instagram caption for a post by a young Pakistani fashion micro-influencer.\n"
        f"Context for the vibe only — never describe it literally: she is at {location['name']} "
        f"during {setting}, wearing {dress['description']}.\n\n"
        f"Copy this exact structure:\n"
        f"- Line 1: one tiny casual vibe line, 2-8 words max, with 1-2 emojis at the end "
        f"(real examples: 'Just a GOOD night🖤✨' / 'Def a long drive & good music person🤎'). "
        f"Once in a while it can be just two emojis and no words at all.\n"
        f"- Then 3-5 lines that each contain only a single dot '.'\n"
        f"- Last line: exactly 5-6 generic viral hashtags chosen from "
        f"#instagood #explore #trending #fyp #viral #reels #instagram #relatable #aesthetic\n"
        f"- Never mention the outfit, fabric, brand or location by name\n"
        f"- No questions, no long sentences, no poetic words, no Urdu\n"
        f"- It must feel effortless, like she typed it in five seconds\n\n"
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
    slides = CAROUSEL_SLIDES[:num_images]
    print(f"  Images   : {num_images}\n")

    all_images = []
    outfit_ref = None
    for i, slide in enumerate(slides, 1):
        pose = random.choice(slide["poses"])
        print(f"Generating model image {i}/{num_images} ({slide['role']})...")
        print(f"  Pose: {pose}")
        img_bytes = generate_model_image(location, dress, setting, slide, pose, ref_photos, outfit_ref)
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
