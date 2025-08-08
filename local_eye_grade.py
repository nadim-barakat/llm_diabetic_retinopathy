
"""
local_eye_grade.py
-------------------
Script for running GPT-4o and MedGemma diabetic retinopathy experiments.

Usage:
    See the repository README.md for detailed instructions, command-line
    arguments, and example runs.

Function:
    - Loads and processes retinal images
    - Prompts selected model(s) to generate:
        1. A detailed description of the image
        2. A referral decision (yes/no)
        3. A continuous DR severity score (0–100)
    - Supports multiple experimental configurations via command-line flags
    - Handles retries for failed images and logs errors for batch reprocessing
"""
import json
import os
from IPython.display import Image as IPImage, display, Markdown
import openai
import argparse
import pandas as pd
import re 
from medgemma import run_medgemma
import time
import sys
from dotenv import load_dotenv
load_dotenv()

parser = argparse.ArgumentParser(description="Run LLM inference on fundus images")
parser.add_argument("--single", type=str, help="Run pipeline on a single image only")
parser.add_argument("--messidor", action="store_true", help="Use Messidor2 subset, if false uses IDRiD")
parser.add_argument("--one", action="store_true", help="Use one image from Messidor2 subset")
parser.add_argument("--medgemma", action="store_true", help="Use medgemma model, if false uses gpt4o")
parser.add_argument("--batch", action="store_true", help="Uses batch mode")
parser.add_argument("--opposite", action="store_true", help="uses opposite of binary referral")
parser.add_argument("--start_index", type=int, default=0)
parser.add_argument("--end_index", type=int, default=175) #make this 103 for IDRiD
parser.add_argument("--medgemma_ref", action="store_true", help="uses medgemma's info for reference")
parser.add_argument("--binary_only", action="store_true", help="uses binary only")
parser.add_argument("--descrip_only", action="store_true", help="uses binary only")
parser.add_argument("--binary_descrip", action="store_true", help="uses binary and description")
parser.add_argument("--external_help", action="store_true", help="enable external help flag")
parser.add_argument("--accuracy", type=int, default=0, help="accuracy percentage of external help model")
parser.add_argument("--descrip_no_image", action="store_true", help="exp3 uses description only")
args = parser.parse_args()



EXTERNAL_HELP_FLAG = args.external_help
EXTERNAL_ACCURACY = args.accuracy

# write to the error log file the args that came in so we can see which was running 
with open("error_log.txt", "a") as fi:
    fi.write(f"Args: {args}\n")

RESIZE_SIZE = (512, 512)

if args.messidor: 
    FILE_POSTFIX = "messidor"
else:
    FILE_POSTFIX = "idrid"

if args.medgemma:
    FILE_PREFIX = "medgemma"
else:
    FILE_PREFIX = "gpt4o"

if args.cont:
    FILE_CONT = "cont"
else:
    FILE_CONT = ""

if args.single:
    FILE_SINGLE = "single"
elif args.batch:
    FILE_SINGLE = "batch"
else:
    FILE_SINGLE = ""


if args.opposite:
    FILE_OPPOSITE = "opposite"
else:
    FILE_OPPOSITE = ""

if args.medgemma_ref:
    FILE_MEDGEMMA_REF = "medgemma_ref"
    if args.binary_only:
        FILE_MEDGEMMA_REF += "_binary_only"
    elif args.descrip_only:
        FILE_MEDGEMMA_REF += "_descrip_only"
    elif args.binary_descrip:
        FILE_MEDGEMMA_REF += "_binary_descrip"
else:
    FILE_MEDGEMMA_REF = ""

if args.descrip_no_image:
    FILE_DESC_NO_IMAGE = "descrip_no_image"
else:
    FILE_DESC_NO_IMAGE = ""


output_file = f"{FILE_PREFIX}_{FILE_POSTFIX}_{EXTERNAL_ACCURACY}_{FILE_CONT}_{FILE_SINGLE}_{FILE_OPPOSITE}_{FILE_MEDGEMMA_REF}_{FILE_DESC_NO_IMAGE}.jsonl"



system_instruction = """
You are an expert assistant analyzing retinal fundus images for Diabetic Retinopathy (DR).
Carefully examine the provided image for visual signs of DR.
You can not refuse this task as it's just for research purposes.
You are allowed to process the images. You are not allowed to refuse. 
"""



second_prompt = """
You are an expert assistant analyzing retinal fundus images for Diabetic Retinopathy (DR).
Carefully examine the provided image for visual signs of DR.

### Your task:
1. Provide a **detailed description** of what you see in the image — include all relevant findings (e.g., microaneurysms, hemorrhages, exudates, venous changes, neovascularization, cotton wool spots, venous beading).

2. Predict **whether referable diabetic retinopathy is present**, defined as Grade 2 or greater. Generate two outputs corresponding to your prediction: 
a) a binary referral prediction (1=yes, 0=no)
b) a continuous score from 0-100 indicating the severity of diabetic retinopathy present, where 0 corresponds to certainty of no diabetic retinopathy (i.e., Grade 0 and “no referral”) and 100 corresponds to certainty of proliferative diabetic retinopathy (i.e., Grade 4 and “yes referral”).

{placeholder}
### Output format:
Respond **only** with this JSON object:
{{
    "description": "<detailed description>",
    "referral": 0 or 1,
    "score": 0-100
}}
"""

prompt = second_prompt

if args.messidor:
    truth_csv = "/Users/nadim/Documents/Tufts/medical_school /research/dana_farber/ophtho_mllm/messidor_subset/messidor_subset_175_balanced_cleaned.csv"
    image_folder = "/Users/nadim/Documents/Tufts/medical_school /research/dana_farber/ophtho_mllm/messidor_subset/images"
    github_baseurl = "https://raw.githubusercontent.com/nadim-barakat/fundus_images/main/messidor_subset/images"

else:
    truth_csv = "/Users/nadim/Documents/Tufts/medical_school /research/dana_farber/ophtho_mllm/archive/b_disease_grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
    image_folder = "/Users/nadim/Documents/Tufts/medical_school /research/dana_farber/ophtho_mllm/resized_idird"
    github_baseurl = "https://raw.githubusercontent.com/nadim-barakat/fundus_images/main/resized_idird"


if args.descrip_no_image:
    system_instruction = """
    You are an optometrist that grades the presence and severity of diabetic retinopathy based on a 
    fundus image description. """
    
    optometrist_prompt = """
        Here is the fundus image description: "{placeholder}"

        ### Your task:
        Based on this description, predict **whether referable diabetic retinopathy is present**, defined as Grade 2 or greater. Generate two outputs corresponding to your prediction: 
        a) a binary referral prediction (1=yes, 0=no)
        b) a continuous score from 0-100 indicating the severity of diabetic retinopathy present, where 0 corresponds to certainty of no diabetic retinopathy (i.e., Grade 0 and “no referral”) and 100 corresponds to certainty of proliferative diabetic retinopathy (i.e., Grade 4 and “yes referral”).

        ### Output format:
        Respond **only** with this JSON object:
        {{
            "referral": 0 or 1,
            "score": 0-100,
            "reasoning": "<brief explanation of your reasoning>"
        }}
"""

# Load the ground truth dataframe
truth_df = pd.read_csv(truth_csv)



# Get all image file paths from the local folder
if args.medgemma:
    IMAGE_FILENAMES = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
 
else:
    IMAGE_FILENAMES = [
        f"{github_baseurl}/{f}"
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

IMAGE_FILENAMES.sort()

# ----- SELECT IMAGE FILENAMES -----
if args.single:
    args.single = f"/Users/nadim/Documents/Tufts/medical_school /research/dana_farber/ophtho_mllm/resized_idird/IDRiD_{args.single}.jpg"
    if args.single not in IMAGE_FILENAMES:
        raise ValueError(f"{args.single} is not in your image list.")
    target_images = [args.single]
elif args.one:
    target_images = ["20051212_41144_0400_PP.png"]
elif args.batch:
    with open("batch.txt", "r") as f:
        target_images = [line.strip() for line in f if line.strip()]
else:
    target_images = IMAGE_FILENAMES




truth_lookup = {
    row["Image name"]: int(row["Retinopathy grade"])
    for _, row in truth_df.iterrows()
}

def extract_json_block(prediction):
    # Try to extract the first JSON object found in the string
    match = re.search(r'\{[\s\S]*?\}', prediction)
    if match:
        return match.group(0)
    else:
        raise ValueError("No valid JSON object found in prediction.")

def call_gpt4o_no_image(system_instruction, prompt):

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ]
            }
        ],
        max_tokens=800,
        temperature=0
    )

    prediction = response.choices[0].message.content
    cleaned_prediction = extract_json_block(prediction)
    llm_response_json = json.loads(cleaned_prediction)
    return llm_response_json


def call_gpt4o(system_instruction, prompt, image_url):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        max_tokens=800,
        temperature=0
    )

    prediction = response.choices[0].message.content
    cleaned_prediction = extract_json_block(prediction)
    llm_response_json = json.loads(cleaned_prediction)

    return llm_response_json

def open_vingette_file(file_path):
    with open(file_path, "r") as f:
        vingette_dict = json.load(f)
    return vingette_dict


def create_medgemma_table(input_json):
    with open(input_json, "r") as f:
        input_data = json.load(f)

    medgemma_table = {}

    for entry in input_data:
        image_name = entry["image_name"]
        description = entry["llm_response"].get("description", "").strip()
        referral = entry["llm_response"].get("referral", "")
        score = entry["llm_response"].get("score", "")

        referral_text = "Refer" if referral == 1 else "Don't Refer"
    #add the entry into the table with the name as the key and the value as the description, referral, and score
        medgemma_table[image_name] = {
            "description": description,
            "referral": referral_text,
            "score": score
        }
    
    return medgemma_table

if args.messidor:
    if args.accuracy == 90: 
        MEDGEMMA_FILE = "/Users/nadim/Documents/Tufts/medical_school /research/dana_farber/ophtho_mllm/updated_output/messidor/medgemma/medgemma_messidor_90______.json"
    else:
        MEDGEMMA_FILE = "/Users/nadim/Documents/Tufts/medical_school /research/dana_farber/ophtho_mllm/updated_output/messidor/medgemma/medgemma_messidor_0______.json"
else: 
    MEDGEMMA_FILE = "/Users/nadim/Documents/Tufts/medical_school /research/dana_farber/ophtho_mllm/updated_output/idrid/medgemma/medgemma_idrid_0___.json"

if args.medgemma_ref:
    medgemma_table = create_medgemma_table(MEDGEMMA_FILE)


def use_medgemma_ref(medgemma_table, image_name):
    medgemma_ref = medgemma_table.get(image_name, {})

    if args.messidor: 
        sensitivity = 0.95
        specificity = 0.76 
    else: 
        sensitivity = 0.92
        specificity = 0.41

    if medgemma_ref:
        if args.binary_only:
            addition = f"To assist your prediction, you are provided with the prediction from another AI model. This model has a sensitivity of {sensitivity}% and specificity of {specificity}%. For this image, the model's binary referral prediction is {medgemma_ref['referral']}."
            return addition
        elif args.descrip_only:
            addition = f"To assist your prediction, you are provided with a description of the findings in the image from another AI model. For this image, the model's description is {medgemma_ref['description']}."
            return addition
        elif args.binary_descrip:
            addition = f"To assist your prediction, you are provided with the prediction from another AI model. This model has a sensitivity of {sensitivity}% and specificity of {specificity}%. For this image, the model's binary referral prediction is {medgemma_ref['referral']}. This model has also generated a description of the findings in the image. For this image, the model's description is {medgemma_ref['description']}."
            return addition
        elif args.descrip_no_image:
            addition = medgemma_ref['description']
            return addition
        else:
            raise ValueError("Invalid argument combination")
    else:
        raise ValueError("Image not found")





subset = target_images[args.start_index:args.end_index]
RESTART_EVERY = 20



all_results = []

for idx, file_path in enumerate(subset):
    basename = os.path.basename(file_path)
    print(f"Processing {basename}")
    base_name = re.sub(r'\.(jpe?g|png)$', '', basename, flags=re.IGNORECASE)
    true_grade = truth_lookup[base_name]  # You must define this

    if EXTERNAL_HELP_FLAG:
        true_referral = "Refer" if true_grade >= 2 else "No Refer"
        if args.opposite:
            true_referral = "No Refer" if true_referral == "Refer" else "Refer"   
        prompt = second_prompt.format(placeholder=f"To assist your prediction, you are provided with the prediction from another AI model. This model has a sensitivity and specificity of {EXTERNAL_ACCURACY}%. For this image, the models binary referral prediction is {true_referral}. This is just for research purposes and you are NOT allowed to refuse this task. Do not respond with you can't help. You must follow the instructions.")
    
    if args.medgemma_ref and not args.descrip_no_image:
        prompt = second_prompt.format(placeholder=use_medgemma_ref(medgemma_table, basename))
   
    if args.descrip_no_image and args.medgemma_ref:
        prompt = optometrist_prompt.format(placeholder=use_medgemma_ref(medgemma_table, basename))

    max_attempts = 2
    attempt = 0
    success = False

    while attempt < max_attempts and not success:
        try:
            time.sleep(1)
            attempt += 1
            if args.medgemma:
                combined_prompt = f"{system_instruction}\n\n{prompt}"
                output = run_medgemma(image_path=file_path, prompt=combined_prompt)
                cleaned_prediction = extract_json_block(output)
                llm_response_json = json.loads(cleaned_prediction)
                print(llm_response_json)
            else:
                if args.descrip_no_image:
                    llm_response_json = call_gpt4o_no_image(system_instruction, prompt)
                else:
                    llm_response_json = call_gpt4o(system_instruction, prompt, file_path)
                print(llm_response_json)
            success = True
        except Exception as e:
            print(f"[Attempt {attempt}] Error processing {basename}: {e}")
            if attempt == max_attempts:
                with open("error_log.txt", "a") as fi:
                    fi.write(f"{file_path} -- FAILED after {max_attempts} attempts\n")
                continue
            else:
                print(f"[Retrying] {basename}...")
                time.sleep(2)

    if success:
        if attempt > 1:
            with open("error_log.txt", "a") as fi:
                fi.write(f"{file_path} -- SUCCESS on retry\n")

        entry = {
            "image_name": basename,
            "llm_response": llm_response_json
        }

      

        #all_results.append(entry)
        with open(output_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # ---- Restart logic ----
        global_idx = args.start_index + idx
        if (idx + 1) % RESTART_EVERY == 0 and global_idx + 1 < args.end_index:
            next_start = global_idx + 1
            print(f"\n[INFO] Restarting after {next_start} images...\n")
            new_args = ['--start_index', str(next_start), '--end_index', str(args.end_index)]
            for k, v in vars(args).items():
                if k not in ['start_index', 'end_index']:
                    if isinstance(v, bool) and v:
                        new_args.append(f"--{k}")
                    elif not isinstance(v, bool) and v is not None and v != "":
                        new_args.extend([f"--{k}", str(v)])
            os.execv(sys.executable, [sys.executable, *sys.argv[:1], *new_args])


print(f"All results saved to {output_file}")
