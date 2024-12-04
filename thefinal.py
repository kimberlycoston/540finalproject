import os  # Always import os before using it
from dotenv import load_dotenv  # dotenv is loaded to manage environment variables
from openai import OpenAI  # Import OpenAI after supporting modules
import requests
import git
from diffusers import StableDiffusionPipeline  # For image generation
import torch  # For running Stable Diffusion on GPU
from jinja2 import Template
from flask import Flask, render_template, request, jsonify
import gspread
import time
from oauth2client.service_account import ServiceAccountCredentials


# Load environment variables from the api.env file
load_dotenv(dotenv_path="api.env")

# Debugging: Print the content of api.env (optional, for development only)
from dotenv import dotenv_values
print(dotenv_values("api.env"))

# Check if the API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found! Ensure it is set in the api.env file.")

print(f"API Key Loaded: {api_key[:10]}...")

# Access Google Sheets credentials path from environment variables
credentials_file = os.getenv("GOOGLE_SHEET_CREDENTIALS")
if not credentials_file:
    raise ValueError("Google Sheets credentials file path not found in environment variables!")

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)
    
def connect_to_google_sheet(credentials_file, sheet_name):
    """
    Connect to a Google Sheet using the Sheets API.
    Args:
        credentials_file (str): Path to the JSON credentials file.
        sheet_name (str): Name of the Google Sheet.
    Returns:
        gspread.Sheet: The connected sheet.
    """
    # Define the scope for accessing Google Sheets and Drive
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
    client = gspread.authorize(creds)

    # Open the Google Sheet
    sheet = client.open(sheet_name).sheet1
    return sheet

def get_latest_form_responses(sheet):
    """
    Get the latest responses from a Google Sheet.
    Args:
        sheet (gspread.Sheet): The connected Google Sheet.
    Returns:
        dict: A dictionary of the latest form responses (headers as keys).
    """
    # Get all rows from the sheet
    rows = sheet.get_all_values()
    headers = rows[0]  # First row contains the headers
    latest_response = rows[-1]  # Last row contains the latest response
    return dict(zip(headers, latest_response))

def load_prompt(file_path):
    """Load a prompt from a text file."""
    with open(file_path, "r") as file:
        return file.read()

def generate_narrative(form_responses, narrative_template):
    """Generate a narrative based on form responses and a prompt template."""
    # Combine the prompt template with form responses
    prompt = f"{narrative_template}\n\nForm Responses:\n{form_responses}"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content

def identify_signals(narrative, signals_template):
    """
    Identify signals based on the narrative and the signals template.
    """
    prompt = f"{signals_template}\n\nNarrative:\n{narrative}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content

def identify_trends(narrative, trends_template):
    """
    Identify trends based on the narrative and the trends template.
    """
    prompt = f"{trends_template}\n\nNarrative:\n{narrative}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content

def identify_themes(narrative, themes_template):
    """
    Identify themes based on the narrative and the themes template.
    """
    prompt = f"{themes_template}\n\nNarrative:\n{narrative}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content

def generate_dalle3_image(prompt, output_path):
    """
    Generate a single image based on the overall narrative using DALL-E or similar.
    Args:
        prompt (str): The description of the image.
        narrative (str): The narrative text to use as input for image generation.
        output_path (str): Path to save the generated image.
    """
    try:
        print(f"Generating image for prompt: {narrative}") # changed this from prompt to narrative, see if it works
        # Generate the image
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,  # Number of images to generate
            size="1024x1024"  # Image resolution
        )
        # Extract the image URL from the response
        image_url = response.data[0].url
        
        # Download and save the image locally
        image_data = requests.get(image_url).content
        with open(output_path, 'wb') as f:
            f.write(image_data)
        
        print(f"Image saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

def text_to_speech_openai(text, output_file="audio/output.mp3", voice="fable", model="tts-1", speed=1.0):
    """
    Convert text to speech using OpenAI's TTS API.
    Args:
        text (str): Text to convert into audio.
        output_file (str): Path to save the audio file.
        voice (str): Voice ID ('fable' for this example).
        model (str): Model type ('tts-1' for real-time, 'tts-1-hd' for high-quality).
        speed (float): Speed of the speech (1.0 = normal, <1.0 = slower, >1.0 = faster).
    """
    print(f"Generating audio for the text: {text[:50]}... at speed {speed}")
    try:
        # Create speech request
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            speed=speed
        )
        # Stream audio directly to a file
        response.stream_to_file(output_file)
        print(f"Audio saved to {output_file}")
    except Exception as e:
        print(f"Error generating audio: {e}")
        
def push_to_github(commit_message):
    """
    Push the generated files to the GitHub repository.
    Args:
        commit_message (str): Commit message for the changes.
    """
    try:
        # Initialize the repository (use the path to your local GitHub repo)
        repo_path = "/path/to/your/540finalproject"  # Adjust this path
        repo = git.Repo(repo_path)

        # Stage all changes
        repo.git.add(all=True)

        # Commit the changes
        repo.index.commit(commit_message)

        # Push to GitHub
        origin = repo.remote(name="origin")
        origin.push()

        print("Changes pushed to GitHub successfully!")
    except Exception as e:
        print(f"Error pushing to GitHub: {e}")
        
def generate_html_storyboard(narrative, signals_trends_themes, image_path, audio_path, output_file, name):
    """
    Generate an HTML storyboard combining narrative, signals, trends, themes, images, and audio.
    Args:
        narrative (str): The generated narrative.
        signals_trends_themes (dict): Dictionary containing signals, trends, and themes.
        image_path (str): Path to the image file.
        audio_path (str): Path to the audio file.
        output_file (str): Output HTML file name.
        name (str): Participant's name.
    """
    if not name.strip():
        name = "Anonymous"

    # Format the narrative to include paragraph tags
    formatted_narrative = "".join(f"<p>{para.strip()}</p>" for para in narrative.split("\n") if para.strip())

    # Extract signals, trends, and themes from the dictionary
    signals = signals_trends_themes.get("signals", "No signals provided.")
    trends = signals_trends_themes.get("trends", "No trends provided.")
    themes = signals_trends_themes.get("themes", "No themes provided.")

    # Generate new content for the HTML file
    new_content = f"""
        <html>
    <head>
        <title>{name}'s Storyboard Entry</title>
        <!-- Include Google Fonts for Audiowide -->
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Audiowide&display=swap" rel="stylesheet">
        <style>
            body {{
                font-family: 'Roboto Condensed', sans-serif; /* Default body font */
                margin: 20px;
                color: #FFFFFF; /* Sets the default text color to white */
                background-color: #00009c; /* Set background color to dark blue */
            }}
            .banner {{
                width: 100%; /* Full width of the page */
                max-height: 300px; /* Optional: Limit banner height */
                object-fit: cover; /* Maintain aspect ratio */
            }}
            h1, h2, h3 {{
                font-family: 'Audiowide', sans-serif;
                color: #FFFFFF; /* white */
            }}
            .section {{
                margin-bottom: 20px;
            }}
            .narrative-image {{
                display: block;
                margin: 20px auto; /* Centers the image */
                max-width: 100%; /* Ensures it doesn't overflow */
                border: 2px solid #2E86C1; /* Optional border for emphasis */
                border-radius: 8px; /* Optional rounded corners */
            }}
        </style>
    </head>
    <body>
        <img src="/Users/kimberlycoston/Desktop/banner.png" alt="Banner Image" class="banner">
        <div class="entry">
            <hr>
            <h1>{name}'s Storyboard Entry</h1>
            <img class="narrative_image" src="{image_path}" alt="Generated Illustration for Narrative">
            {formatted_narrative}
            <hr>
            <h2>Signals, Trends, and Themes</h2>
            <div class="section">
                <h3>Signals</h3>
                <p>{signals_trends_themes['signals']}</p>
            </div>
            <div class="section">
                <h3>Trends</h3>
                <p>{signals_trends_themes['trends']}</p>
            </div>
            <div class="section">
                <h3>Themes</h3>
                <p>{signals_trends_themes['themes']}</p>
            </div>
            <hr>
            <h2>Listen to the Narrative</h2>
            <audio controls>
                <source src="{audio_path}" type="audio/mpeg">
                Your browser does not support the audio element.
            </audio>
        </div><hr>"""

    # Check if the output file exists
    if not os.path.exists(output_file):
        with open(output_file, "w") as file:
            file.write(f"""
            <html>
            <head>
                <title>All Storyboard Entries</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .entry {{ margin-bottom: 40px; }}
                    .section {{ margin-bottom: 20px; }}
                    h3 {{ color: #333; }}
                    audio {{ width: 100%; margin-top: 20px; }}
                </style>
            </head>
            <body>
            {new_content}
            </body>
            </html>
            """)
        print(f"New HTML file created: {output_file}")
    else:
        print(f"{output_file} exists. Appending content...")
        with open(output_file, "r+", encoding="utf-8") as file:
            content = file.read()
            if "</body>" not in content:
                print("Warning: </body> tag not found. Adding it to the file.")
                content += "</body>\n</html>"
            updated_content = content.replace("</body>", f"{new_content}</body>")
            print("Updated Content to Write:")  # Debug
            print(updated_content)
            file.seek(0)
            file.write(updated_content)
            file.truncate()
        print(f"Appended new content to {output_file}")

if __name__ == "__main__":
    # Step 1: # Load credentials file path from environment variables
    credentials_file = "GOGGLE_SHEET_CREDENTIALS"  # Replace with your credentials file path

    # Use the credentials file to connect to Google Sheets
    sheet_name = "Final_Responses"  # Replace with your Google Sheet name
    sheet = connect_to_google_sheet(credentials_file, sheet_name)

    # Step 2: Fetch the latest form responses
    form_responses = get_latest_form_responses(sheet)
    name = form_responses.get("What's your first name?", "Anonymous").strip()
    if not name:
        name = "Anonymous"
    print("Latest Form Responses:", form_responses)

    # Step 3: Load the prompt template
    narrative_template = load_prompt("assets/text/task1_narrative.txt")  # Adjust file path if necessary

    # Step 4: Generate the narrative using Google Form responses
    narrative = generate_narrative(form_responses, narrative_template)
    print("Generated Narrative:", narrative)

    # Step: Load the signals template
    signals_template = load_prompt("/assets/text/task2_signals.txt")  # Adjust file path if necessary

    # Step: Generate signals using the narrative and the signals template
    signals = identify_signals(narrative, signals_template)
    print("Generated Signals:", signals)

    # Step: Load the trends template
    trends_template = load_prompt("assets/text/task3_trends.txt")  # Adjust file path if necessary

    # Step: Generate trends using the narrative and the trends template
    trends = identify_trends(narrative, trends_template)
    print("Generated Trends:", trends)

    # Step: Load the themes template
    themes_template = load_prompt("assets/text/task4_themes.txt")  # Adjust file path if necessary

    # Step: Generate themes using the narrative and the themes template
    themes = identify_themes(narrative, themes_template)
    print("Generated Themes:", themes)
    
    # Generate timestamp for file names
    timestamp = int(time.time())  # Use the current time for uniqueness

    # Step: Generate images for the narratives
    # Generate a unique file name for the new narrative image
    output_path = f"assets/images/narrative_image_{timestamp}.png"
    image_path = generate_dalle3_image(narrative, output_path=output_path)

    # Step 9: Print the generated image paths
    print("\nGenerated Image:")
    print(f"Narrative Image Path: {image_path}")

    # Step 10: Convert the narrative to audio
    audio_path = f"assets/audio/narrative_{timestamp}.mp3"
    text_to_speech_openai(narrative, output_file=audio_path, voice="fable", model="tts-1", speed=0.95)

    #Step 11:
    # Combine signals, trends, and themes into a dictionary
    signals_trends_themes = {
        "signals": signals,
        "trends": trends,
        "themes": themes
    }
    # Step 12: Generate the HTML storyboard
    generate_html_storyboard(
        narrative=narrative,
        signals_trends_themes=signals_trends_themes,  # Single dictionary for related data
        image_path=image_path,
        audio_path=audio_path,
        output_file="index.html",
        name=name
    )

    # Generate the HTML, images, and audio (already in your script)

    # Push the changes to GitHub
    commit_message = "Update generated storyboard and assets"
    push_to_github("Update generated storyboard and assets")
