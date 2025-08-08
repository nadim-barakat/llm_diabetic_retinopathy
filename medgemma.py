import subprocess
from pathlib import Path
import subprocess
import re
from pathlib import Path

def run_medgemma(
    image_path: str,
    prompt: str,
    model_path: str = "/Users/nadim/Documents/Tufts/medical_school /research/dana_farber/ophtho_mllm/llama.cpp/models/medgemma/medgemma-4b-it-Q4_1.gguf",
    mmproj_path: str = "/Users/nadim/Documents/Tufts/medical_school /research/dana_farber/ophtho_mllm/llama.cpp/models/medgemma/mmproj-F16.gguf"
) -> str:

    bin_path = Path("/Users/nadim/Documents/Tufts/medical_school /research/dana_farber/ophtho_mllm/llama.cpp/build/bin/llama-mtmd-cli").expanduser()

    if not Path(image_path).expanduser().is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not Path(model_path).expanduser().is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not Path(mmproj_path).expanduser().is_file():
        raise FileNotFoundError(f"mmproj file not found: {mmproj_path}")
    if not bin_path.is_file():
        raise FileNotFoundError(f"Llama binary not found: {bin_path}")

    cmd = [
        str(bin_path),
        "-m", str(Path(model_path).expanduser()),
        "--mmproj", str(Path(mmproj_path).expanduser()),
        "--image", str(Path(image_path).expanduser()),
        "-p", prompt,
    ]

    result = subprocess.run(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,  # capture stderr just in case
    text=True
)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed:\n{result.stderr}")

    # Extract JSON block if it exists
    match = re.search(r"```json\n(.*?)\n```", result.stdout, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return result.stdout.strip()





if __name__ == "__main__":
    system_instruction = "You are a retinal disease specialist. Provide concise but accurate diagnostic impressions of fundus images."
    user_prompt = """
            You are an expert assistant analyzing retinal fundus images for Diabetic Retinopathy (DR).
            Carefully examine the provided image for visual signs of DR.

            ### Your task:
            1 Provide a **detailed description** of what you see in the image — include all relevant findings (e.g., microaneurysms, hemorrhages, exudates, venous changes, neovascularization).

            2 Determine **if the patient needs referral to an ophthalmologist** for possible vision-threatening diabetic retinopathy.

            ### Referral criteria:
            Respond `1` (referral needed) if **any** of the following are present:
            - Severe non-proliferative diabetic retinopathy (Severe NPDR, Grade 3), which includes:
                - Dense or widespread intraretinal hemorrhages (often >20 per quadrant)
                - Definite venous beading in 2 or more quadrants
                - Moderate to severe intraretinal microvascular abnormalities (IRMA)
            - Proliferative diabetic retinopathy (PDR, Grade 4), which includes:
                - Neovascularization (new abnormal blood vessels, on the disc or elsewhere)
                - Vitreous or preretinal hemorrhage (bleeding into the eye)

            Respond `0` (no referral needed) if:
            - There are **only** microaneurysms, small hemorrhages, hard exudates, or cotton wool spots, without severe features (Grades 0-2, mild or moderate NPDR).

            3 Finally, classify the severity using the **International Clinical Diabetic Retinopathy Scale (0-4)**:
            - Grade 0 (No DR): No abnormalities.
            - Grade 1 (Mild NPDR): Only microaneurysms.
            - Grade 2 (Moderate NPDR): More than microaneurysms (e.g., hemorrhages, exudates, cotton wool spots), but no severe NPDR signs.
            - Grade 3 (Severe NPDR): Severe hemorrhages/microaneurysms (>20 per quadrant), venous beading in ≥2 quadrants, or moderate IRMA in ≥1 quadrant, but no proliferative signs.
            - Grade 4 (PDR): Neovascularization or vitreous/preretinal hemorrhage.

            ---

            ### Output format:
            Respond **only** with this JSON object:
            {
                "description": "<detailed description>",
                "referral": 0 or 1,
                "grade": 0-4
            }

            Do not include any other text, explanation, or formatting.
"""

    combined_prompt = f"{system_instruction}\n\n{user_prompt}"

    output = run_medgemma(
        image_path="/Users/nadim/Desktop/image-full-1.jpg",
        prompt=combined_prompt,
        model_path="/Users/nadim/Documents/Tufts/medical_school /research/dana_farber/ophtho_mllm/llama.cpp/models/medgemma/medgemma-4b-it-Q4_1.gguf",
        mmproj_path="/Users/nadim/Documents/Tufts/medical_school /research/dana_farber/ophtho_mllm/llama.cpp/models/medgemma/mmproj-F16.gguf"
    )
    
    print(output)
