import os
import asyncio
import aiohttp
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()

AZURE_TTS_KEY = os.getenv("AZURE_TTS_KEY")
AZURE_TTS_ENDPOINT = os.getenv("AZURE_TTS_ENDPOINT")
AZURE_TTS_DEPLOYMENT = os.getenv("AZURE_TTS_DEPLOYMENT", "tts")
AZURE_TTS_API_VERSION = os.getenv("AZURE_TTS_API_VERSION", "2025-03-01-preview")

async def synthesize_azure_tts(text, voice, output_file):
    url = f"{AZURE_TTS_ENDPOINT}/openai/deployments/{AZURE_TTS_DEPLOYMENT}/audio/speech?api-version={AZURE_TTS_API_VERSION}"
    headers = {
        "api-key": AZURE_TTS_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "model": AZURE_TTS_DEPLOYMENT,
        "input": text,
        "voice": voice,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            content = await resp.read()
            with open(output_file, "wb") as f:
                f.write(content)
    print(f"Azure TTS audio saved to: {output_file}")

async def generate_podcast(conversation, output_file="podcast_output_azure.mp3"):
    temp_files = [f".podcast_turn_{idx}.mp3" for idx in range(len(conversation))]
    print("Generating podcast turns with Azure TTS in parallel...")
    tasks = []
    for idx, (speaker, text, voice) in enumerate(conversation):
        turn_text = f"{speaker}: {text}"
        temp_file = temp_files[idx]
        print(f"  Scheduling turn {idx+1} with voice '{voice}'...")
        tasks.append(synthesize_azure_tts(turn_text, voice, temp_file))
    await asyncio.gather(*tasks)
    combined = None
    for temp_file in temp_files:
        segment = AudioSegment.from_file(temp_file, format="mp3")
        if combined is None:
            combined = segment
        else:
            pause = AudioSegment.silent(duration=600)
            combined += pause + segment
    combined.export(output_file, format="mp3")
    print(f"Podcast audio generated successfully: {output_file}")
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except Exception:
            pass

if __name__ == "__main__":
    # Example usage: just pass the conversation
    conversation = [
        ("Alice", "Welcome to our podcast! Today, we're talking about public speaking tips and tricks.", "nova"),
        ("Bob", "Hi Alice! I'm excited to share my experiences. Let's start with the importance of preparation.", "fable"),
        ("Alice", "Preparation is key. I always write an outline before drafting my speech.", "nova"),
        ("Bob", "That's a great point. I also rehearse in front of friends to get feedback.", "fable"),
        ("Alice", "Do you ever get nervous before speaking?", "nova"),
        ("Bob", "Absolutely! I think everyone does. Deep breathing helps me calm down.", "fable"),
        ("Alice", "I like to visualize success before stepping on stage. It boosts my confidence.", "nova"),
        ("Bob", "What about engaging the audience? Any tips?", "fable"),
        ("Alice", "I ask questions and use stories. People love relatable examples.", "nova"),
        ("Bob", "That's true. Humor can also break the ice and make the talk memorable.", "fable"),
        ("Alice", "How do you handle unexpected situations, like technical issues?", "nova"),
        ("Bob", "I try to stay calm and improvise. Sometimes, those moments make the speech more authentic.", "fable"),
        ("Alice", "Great advice! Any final thoughts for our listeners?", "nova"),
        ("Bob", "Just remember, every speaker improves with practice. Keep learning and don't be afraid to make mistakes.", "fable"),
        ("Alice", "Thank you, Bob! And thank you to everyone for tuning in. See you next time!", "nova"),
        # Extended conversation
        ("Bob", "Before we wrap up, let's talk about handling Q&A sessions.", "fable"),
        ("Alice", "Good idea! I always repeat the question for the audience before answering.", "nova"),
        ("Bob", "And if you don't know the answer, it's okay to admit it. Offer to follow up later.", "fable"),
        ("Alice", "What about using visual aids?", "nova"),
        ("Bob", "Visuals can be powerful, but keep them simple and relevant.", "fable"),
        ("Alice", "I agree. Too many slides can distract from your message.", "nova"),
        ("Bob", "Do you have a favorite speech you've given?", "fable"),
        ("Alice", "Yes, my graduation speech. It was emotional and I received great feedback.", "nova"),
        ("Bob", "That's wonderful! My favorite was a talk on innovation at a tech conference.", "fable"),
        ("Alice", "Any advice for overcoming mistakes during a speech?", "nova"),
        ("Bob", "Don't dwell on them. Move forward and keep your composure.", "fable"),
        ("Alice", "Thanks for sharing your insights, Bob.", "nova"),
        ("Bob", "Thank you, Alice. And thanks to our listeners for joining us today!", "fable"),
        ("Alice", "We'll be back next week with more tips. Goodbye!", "nova"),
    ]
    asyncio.run(generate_podcast(conversation))
