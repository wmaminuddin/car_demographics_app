from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from google import genai
import base64
from io import BytesIO
from PIL import Image
import traceback
from google.genai import types
import os

app = Flask(__name__)

# Load models
age_model = joblib.load('models/age_model.pkl')
gender_model = joblib.load('models/gender_model.pkl')
race_model = joblib.load('models/race_model.pkl')
marital_model = joblib.load('models/marital_model.pkl')

# Load encoders and scaler
gender_encoder = joblib.load('models/gender_encoder.pkl')
race_encoder = joblib.load('models/race_encoder.pkl')
marital_encoder = joblib.load('models/marital_encoder.pkl')
scaler = joblib.load('models/scaler.pkl')

# Get API key from environment variable with optional fallback
api_key = os.getenv("GEMINI_API_KEY")

# Optional validation to prevent runtime errors
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
    
# Create a client
client = genai.Client(api_key=api_key)

def get_age_descriptor(age):
    age_ranges = {
        (18, 25): "young first-time buyers",
        (26, 35): "young professionals",
        (36, 50): "family-oriented buyers",
        (51, 65): "experienced commuters",
        (66, 100): "mature drivers"
    }
    for rng, descriptor in age_ranges.items():
        if rng[0] <= age <= rng[1]:
            return descriptor
    return "various drivers"

def get_gemini_commentary(results):
    """Get AI commentary based on prediction results"""
    prompt = f"""
    Analyze these car specifications and predicted customer demographics:
    
    Car Specifications:
    - Dimensions: {results['car_specs']['height']}mm (H) x {results['car_specs']['width']}mm (W) x {results['car_specs']['length']}mm (L)
    - Weight: {results['car_specs']['weight']}kg
    - Seats: {results['car_specs']['seats']}
    - Cargo: {results['car_specs']['cargo']}L
    - Price: RM {results['car_specs']['price']}
    
    Predicted Demographics:
    - Average Age: {results['age']['prediction']}
    - Most Likely Gender: {results['gender']['prediction']} ({max(results['gender']['confidence'].values())}% confidence)
    - Predicted Race: {results['race']['prediction']}
    - Marital Status: {results['marital_status']['prediction']}
    
    Provide a brief marketing analysis commentary focusing on:
    1. Why these demographics might be interested in this vehicle
    2. Suggested marketing strategies based on the predictions
    3. Any notable correlations between car specs and demographics
    4. What car brand and model are they most likely to buy, perodua and any other brands

    Context: this is a machine learning model, trained on Perodua's customers database, use by Perodua internal users
    
    Keep it under 150 words and use simple business language.
    """
    
    print("[DEBUG] Generating AI commentary...")
    try:
        # Use the same client approach that worked for images
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        print("[DEBUG] AI commentary generated successfully")
        return response.text
    except Exception as e:
        print(f"[DEBUG] Error getting AI commentary: {str(e)}")
        return "AI commentary unavailable at this time."

# Define process_image function outside to avoid scope issues
def process_image(response):
    try:
        print(f"[DEBUG] Processing response type: {type(response)}")
        print(f"[DEBUG] Response attributes: {dir(response)}")
        
        if not hasattr(response, 'parts'):
            print("[DEBUG] Response has no 'parts' attribute")
            print(f"[DEBUG] Looking for other attributes: candidates={hasattr(response, 'candidates')}, text={hasattr(response, 'text')}")
            
            # Try alternative attribute paths
            if hasattr(response, 'candidates'):
                print(f"[DEBUG] Found candidates: {len(response.candidates)}")
                if len(response.candidates) > 0 and hasattr(response.candidates[0], 'content'):
                    if hasattr(response.candidates[0].content, 'parts'):
                        parts = response.candidates[0].content.parts
                        print(f"[DEBUG] Found parts in candidates[0].content: {len(parts)}")
                    else:
                        print("[DEBUG] No parts in candidates[0].content")
                        return None
                else:
                    print("[DEBUG] No valid candidates or content")
                    return None
            else:
                print("[DEBUG] No usable structure found in response")
                return None
        else:
            parts = response.parts
            print(f"[DEBUG] Response has {len(parts)} parts")
        
        # Inspect all parts
        for i, part in enumerate(parts):
            print(f"[DEBUG] Examining part {i}, attributes: {dir(part)}")
            
            # Check for inline_data attribute (snake_case)
            if hasattr(part, 'inline_data'):
                print(f"[DEBUG] Part {i} has inline_data")
                inline_data = part.inline_data
                print(f"[DEBUG] inline_data attributes: {dir(inline_data)}")
                
                if hasattr(inline_data, 'mime_type'):
                    print(f"[DEBUG] MIME type: {inline_data.mime_type}")
                    
                    if inline_data.mime_type.startswith('image/'):
                        print("[DEBUG] Found image data!")
                        
                        if hasattr(inline_data, 'data'):
                            print(f"[DEBUG] Data length: {len(inline_data.data)}")
                            encoded = base64.b64encode(inline_data.data).decode("utf-8")
                            print(f"[DEBUG] Encoded base64 length: {len(encoded)}")
                            print(f"[DEBUG] First 50 chars: {encoded[:50]}")
                            return encoded
                        else:
                            print("[DEBUG] inline_data has no 'data' attribute")
                else:
                    print("[DEBUG] inline_data has no 'mime_type' attribute")
            
            # Check for inlineData attribute (camelCase)
            elif hasattr(part, 'inlineData'):
                print(f"[DEBUG] Part {i} has inlineData (camelCase)")
                inline_data = part.inlineData
                
                if hasattr(inline_data, 'mimeType'):
                    print(f"[DEBUG] MIME type: {inline_data.mimeType}")
                    
                    if inline_data.mimeType.startswith('image/'):
                        print("[DEBUG] Found image data!")
                        
                        if hasattr(inline_data, 'data'):
                            encoded = base64.b64encode(inline_data.data).decode("utf-8")
                            print(f"[DEBUG] Encoded base64 length: {len(encoded)}")
                            return encoded
            
            # If no image data, check for text
            elif hasattr(part, 'text'):
                print(f"[DEBUG] Part {i} has text: {part.text[:30]}...")
        
        print("[DEBUG] No image data found in any part")
        return None
    except Exception as e:
        print(f"[DEBUG] Image processing error: {str(e)}")
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return None

def generate_images(car_specs, demographics):
    print("\n[DEBUG] === STARTING IMAGE GENERATION ===")
    print(f"[DEBUG] Car specs: {car_specs}")
    print(f"[DEBUG] Demographics: {demographics}")
    
    # Create car prompt with demographic-indirect features - now requesting descriptions
    car_prompt = f"""Generate a realistic image of a car with:
    - {car_specs['seats']} seats
    Show it in a setting suitable for {"urban" if car_specs['price'] < 60000 else "suburban"} use.
    
    Also provide a brief description of the type of car shown and its key features."""

    # Demographic prompt using lifestyle indicators (policy compliant) - now requesting descriptions
    demo_prompt = f"""Generate an image showing:
- A lifestyle scene representing {demographics['marital_status'].lower()} consumers
- The setting should be {"urban city background" if car_specs['price'] < 60000 else "suburban neighborhood"}
- Show transportation context without specific people
- Include design elements that appeal to {"younger" if demographics['age'] < 40 else "mature"} consumers
- Professional stock photo style

Also provide a brief description of who is shown in the image and what they're doing
"""

    # Debug the prompts
    print(f"[DEBUG] Car prompt: {car_prompt}")
    print(f"[DEBUG] Demo prompt: {demo_prompt}")

    try:
        # Use the correct client method to generate content
        print("[DEBUG] Generating car image...")
        car_response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=car_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["Text", "Image"]
            )
        )
        print("[DEBUG] Car image response received")
        print(f"[DEBUG] Car response type: {type(car_response)}")
        print(f"[DEBUG] Car response attributes: {dir(car_response)[:10]}...")
        
        print("[DEBUG] Generating demographic image...")
        demo_response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=demo_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["Text", "Image"]
            )
        )
        print("[DEBUG] Demographic image response received")
        
        # Set default descriptions
        car_description = "Generated car image"
        demo_description = "Generated demographic representation"
        
        # Extract text from complete response
        print("[DEBUG] Dumping full car response structure...")
        if hasattr(car_response, 'candidates') and car_response.candidates:
            print(f"[DEBUG] Car response has {len(car_response.candidates)} candidates")
            
            # Try to get the full text from the response first
            if hasattr(car_response, 'text') and car_response.text:
                print(f"[DEBUG] Found car text at root: {car_response.text[:50]}...")
                car_description = car_response.text[:150]
            
            # Otherwise look through parts
            else:
                for i, candidate in enumerate(car_response.candidates):
                    print(f"[DEBUG] Examining candidate {i}")
                    if hasattr(candidate, 'content') and candidate.content:
                        print(f"[DEBUG] Candidate {i} has content")
                        
                        # Try to get text from candidate's text property
                        if hasattr(candidate, 'text') and candidate.text:
                            print(f"[DEBUG] Found car text in candidate: {candidate.text[:50]}...")
                            car_description = candidate.text[:150]
                            break
                            
                        # Otherwise check all parts
                        elif hasattr(candidate.content, 'parts'):
                            print(f"[DEBUG] Candidate {i} has {len(candidate.content.parts)} parts")
                            
                            # Look for text parts
                            text_parts = []
                            for j, part in enumerate(candidate.content.parts):
                                if hasattr(part, 'text') and part.text:
                                    print(f"[DEBUG] Found text in part {j}: {part.text[:50]}...")
                                    text_parts.append(part.text)
                            
                            # Use the longest text part (likely the description)
                            if text_parts:
                                longest_text = max(text_parts, key=len)
                                car_description = longest_text[:150]
                                print(f"[DEBUG] Using longest car text: {car_description}")
        
        # Extract demographic image text using same approach
        print("[DEBUG] Extracting demographic text...")
        if hasattr(demo_response, 'candidates') and demo_response.candidates:
            print(f"[DEBUG] Demo response has {len(demo_response.candidates)} candidates")
            
            # Try to get the full text from the response first
            if hasattr(demo_response, 'text') and demo_response.text:
                print(f"[DEBUG] Found demo text at root: {demo_response.text[:50]}...")
                demo_description = demo_response.text[:150]
            
            # Otherwise look through parts
            else:
                for i, candidate in enumerate(demo_response.candidates):
                    if hasattr(candidate, 'content') and candidate.content:
                        # Try to get text from candidate's text property
                        if hasattr(candidate, 'text') and candidate.text:
                            print(f"[DEBUG] Found demo text in candidate: {candidate.text[:50]}...")
                            demo_description = candidate.text[:150]
                            break
                            
                        # Otherwise check all parts
                        elif hasattr(candidate.content, 'parts'):
                            # Look for text parts
                            text_parts = []
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    text_parts.append(part.text)
                            
                            # Use the longest text part (likely the description)
                            if text_parts:
                                longest_text = max(text_parts, key=len)
                                demo_description = longest_text[:150]
                                print(f"[DEBUG] Using longest demo text: {demo_description}")
        
        # Process images using the global process_image function
        print("[DEBUG] Processing car image...")
        car_image = process_image(car_response)
        print(f"[DEBUG] Car image processed: {'Success' if car_image else 'Failed'}")
        
        print("[DEBUG] Processing demographic image...")
        demo_image = process_image(demo_response)
        print(f"[DEBUG] Demographic image processed: {'Success' if demo_image else 'Failed'}")
        
        result = {
            'car_image': car_image,
            'demographic_image': demo_image,
            'car_description': car_description,
            'demo_description': demo_description
        }
        print(f"[DEBUG] Images generated: {car_image is not None}, {demo_image is not None}")
        print(f"[DEBUG] Descriptions found: {car_description != 'Generated car image'}, {demo_description != 'Generated demographic representation'}")
        return result
    except Exception as e:
        print(f"[DEBUG] Image generation error: {str(e)}")
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return {
            'car_image': None,
            'demographic_image': None,
            'car_description': "Image description unavailable",
            'demo_description': "Image description unavailable"
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("\n[DEBUG] === STARTING PREDICTION ===")
    
    # Get input values from form
    height = float(request.form.get('height'))
    length = float(request.form.get('length'))
    width = float(request.form.get('width'))
    weight = float(request.form.get('weight'))
    seats = float(request.form.get('seats'))
    cargo = float(request.form.get('cargo'))
    price = float(request.form.get('price'))
    
    print(f"[DEBUG] Form values: height={height}, length={length}, width={width}, weight={weight}, seats={seats}, cargo={cargo}, price={price}")
    
    # Calculate volume
    volume = height * length * width
    
    # Prepare input features
    input_features = np.array([[height, length, width, volume, weight, seats, cargo, price]])
    input_scaled = scaler.transform(input_features)
    
    print("[DEBUG] Making predictions...")
    # Predict age
    age_pred = age_model.predict(input_scaled)[0]
    
    # Predict gender with confidence levels
    gender_proba = gender_model.predict_proba(input_scaled)[0]
    gender_confidence = {gender_encoder.inverse_transform([i])[0]: round(prob*100, 2) 
                        for i, prob in enumerate(gender_proba)}
    gender_pred = gender_encoder.inverse_transform([gender_proba.argmax()])[0]
    
    # Predict race with confidence levels
    race_proba = race_model.predict_proba(input_scaled)[0]
    race_confidence = {race_encoder.inverse_transform([i])[0]: round(prob*100, 2) 
                      for i, prob in enumerate(race_proba)}
    race_pred = race_encoder.inverse_transform([race_proba.argmax()])[0]
    
    # Predict marital status with confidence levels
    marital_proba = marital_model.predict_proba(input_scaled)[0]
    marital_confidence = {marital_encoder.inverse_transform([i])[0]: round(prob*100, 2) 
                         for i, prob in enumerate(marital_proba)}
    marital_pred = marital_encoder.inverse_transform([marital_proba.argmax()])[0]
    
    print("[DEBUG] Predictions complete")
    print(f"[DEBUG] Age: {age_pred}, Gender: {gender_pred}, Race: {race_pred}, Marital: {marital_pred}")
    
    # Prepare results for display
    results = {
        'car_specs': {
            'height': height,
            'length': length,
            'width': width,
            'volume': volume,
            'weight': weight,
            'seats': seats,
            'cargo': cargo,
            'price': price
        },
        'age': {
            'prediction': round(age_pred, 1)
        },
        'gender': {
            'prediction': gender_pred,
            'confidence': gender_confidence
        },
        'race': {
            'prediction': race_pred,
            'confidence': race_confidence
        },
        'marital_status': {
            'prediction': marital_pred,
            'confidence': marital_confidence
        }
    }
    
    # Prepare demographic data for image generation
    demographics = {
        'age': round(age_pred, 1),
        'gender': gender_pred,
        'race': race_pred,
        'marital_status': marital_pred
    }
    
    # Get AI commentary
    print("[DEBUG] Getting AI commentary...")
    try:
        commentary = get_gemini_commentary(results)
        print("[DEBUG] AI commentary received successfully")
    except Exception as e:
        print(f"[DEBUG] Error getting AI commentary: {str(e)}")
        commentary = "AI commentary unavailable at this time."
    
    # Generate images based on results
    print("[DEBUG] Generating images...")
    try:
        images = generate_images(results['car_specs'], demographics)
        print(f"[DEBUG] Images generated: {images.keys()}")
    except Exception as e:
        print(f"[DEBUG] Error in image generation: {str(e)}")
        images = {
            'car_image': None,
            'demographic_image': None
        }
    
    # Add commentary and images to results
    results['ai_commentary'] = commentary
    results['generated_images'] = images
    
    print(f"[DEBUG] Final results keys: {results.keys()}")
    print(f"[DEBUG] Generated images present: {'car_image' in results['generated_images']}")
    print(f"[DEBUG] Has car image: {results['generated_images'].get('car_image') is not None}")
    print(f"[DEBUG] Has demo image: {results['generated_images'].get('demographic_image') is not None}")
    
    # Return the template with results
    return render_template('result.html', results=results)
   
if __name__ == '__main__':
    app.run(debug=True)
