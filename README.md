**# SmartScan — Document Scanner and Image Enhancement**



**\*\*Repository:\*\*** \`D_Team_13\`  

**\*\*Project:\*\*** SmartScan — Document Detection, Perspective Correction, Image Enhancement and OCR  

**\*\*Technology:\*\*** Python · OpenCV · NumPy · Streamlit · Tesseract OCR · Pytest



**## Overview**



SmartScan converts photographs of paper documents into clean, readable digital scans. It detects the page boundary, corrects perspective, enhances readability, optionally extracts text using OCR, and supports downloading scanned pages and extracted text.



The document detection and enhancement stages use classical computer-vision techniques; the application does **\*\*not\*\*** train a machine-learning model. OCR uses the separately installed, pretrained **\*\*Tesseract\*\*** engine.



**### Key features**



\- Upload document photos through a Streamlit web interface.

\- Detect document boundaries and estimate four page corners using multiple image-processing strategies.

\- Correct perspective with a homography and optionally adjust corners manually.

\- Improve scans with illumination normalization, noise reduction and optional CLAHE.

\- Produce **\*\*enhanced color\*\***, **\*\*grayscale\*\*** and **\*\*black-and-white\*\*** output variants.

\- Detect potential blur, low contrast and low resolution and show quality warnings.

\- Extract text with Tesseract OCR, including confidence information and word-box visualization.

\- Export scanned images as **\*\*PNG\*\***, documents as **\*\*PDF\*\***, and recognized text as **\*\*TXT\*\***.

\- Run automated tests and evaluate detection, enhancement, OCR and pipeline reliability.



**## Repository contents**



\`\`\`text

D_Team_13/

├── app.py                         # Streamlit user interface

├── requirements.txt               # Python dependencies

├── README.md                      # This guide

├── CONTRIBUTIONS.md               # Team contribution record

├── pytest.ini                     # Test configuration

├── conftest.py

├── .streamlit/config.toml          # Streamlit settings

├── .github/workflows/tests.yml    # CI tests

├── backend/

│   ├── contracts.py               # Shared data types and structured errors

│   ├── input_handler.py           # File decoding, validation, EXIF and resizing

│   ├── detection.py               # Page boundary and corner detection

│   ├── geometry.py                # Perspective transform and rotation

│   ├── deskew\.py                  # Skew correction

│   ├── text_region.py             # Text-region helpers and fallback

│   ├── enhancement.py             # Color/grayscale/BW enhancement and quality checks

│   ├── ocr.py                     # Tesseract OCR integration

│   ├── orientation_search.py      # Orientation helpers

│   ├── pipeline.py                # End-to-end orchestration

│   ├── export.py                  # Output export utilities

│   └── ui_support.py              # UI-related helper functions

├── evaluation/

│   ├── README.md                  # Evaluation methodology and dataset format

│   ├── synthetic.py               # Synthetic dataset generator

│   ├── annotate_corners.py        # Ground-truth corner annotation

│   ├── detection_metrics.py       # IoU and corner accuracy

│   ├── enhancement_metrics.py     # Contrast, sharpness, background uniformity

│   ├── ocr_metrics.py             # Character and word error rates

│   ├── reliability.py             # Failure and runtime analysis

│   ├── debug_run.py               # Intermediate pipeline images

│   └── dataset.py                 # Dataset loading and CSV helpers

├── sample_data/

│   ├── README.md                  # Dataset and annotation instructions

│   └── synthetic/                # 30 included synthetic JPG + JSON pairs

├── tests/                          # Pytest unit and integration tests

├── docs/

│   ├── ACCEPTANCE_TESTS.md        # Manual acceptance checks

│   ├── GIT_WORKFLOW\.md            # Git workflow notes

│   ├── presentation/              # Add final PPT/PPTX here

│   └── report/                    # Add final project report here

└── .gitignore

\`\`\`



**## Implementation details**



**### Processing pipeline**



\`\`\`text

Uploaded JPG/PNG

      ↓

Validate and decode image; correct EXIF orientation

      ↓

Create smaller working copy for detection

      ↓

Detect page boundary and four corners

      ↓

Perspective correction on the original-resolution image

      ↓

Optional deskew / orientation correction

      ↓

Enhancement and image-quality assessment

      ├── Color (LAB lightness enhancement + white balance)

      ├── Grayscale (illumination normalization, denoising, optional CLAHE)

      └── BW (Gaussian adaptive thresholding)

      ↓

Optional Tesseract OCR

      ↓

Preview, diagnostics and PNG/PDF/TXT downloads

\`\`\`



**\*\*Document detection (\`backend/detection.py\`):\*\*** Uses classical edge/threshold-based strategies and contour/corner candidate evaluation. The pipeline can use fallback strategies when a confident document boundary is unavailable.



**\*\*Perspective correction (\`backend/geometry.py\`):\*\*** Orders four corners and computes a homography with OpenCV to transform the photographed page into a top-down rectangular scan.



**\*\*Image enhancement (\`backend/enhancement.py\`):\*\*** Estimates illumination background with morphological closing and Gaussian smoothing, normalizes shadows by division, denoises with Non-Local Means (or bilateral filtering for large images), and applies CLAHE when selected or when automatic contrast assessment indicates it is needed. Gaussian adaptive thresholding generates the binary scan. The enhanced color variant works in LAB space to preserve chromatic information while improving lightness. Quality checks flag blur using Laplacian variance, low contrast using text–paper separation, and low resolution using image dimensions.



**\*\*OCR (\`backend/ocr.py\`, \`backend/pipeline.py\`):\*\*** Uses \`pytesseract\` to call the locally installed Tesseract executable. The pipeline can compare OCR attempts using different page-segmentation configurations and image variants. OCR failure does not prevent users from downloading an image scan.



**\*\*Interface and export (\`app.py\`, \`backend/export.py\`):\*\*** Streamlit provides upload, controls, previews, manual corner adjustment, OCR diagnostics and download options.



**## Dataset(s)**



The supplied project ZIP contains **\*\*30 synthetic document photos\*\*** in \`sample_data/synthetic/\`, each accompanied by a JSON annotation file (\`synthetic_00_dark.jpg\` through \`synthetic_29_wood.jpg\`, with matching \`.json\` files). The synthetic examples cover backgrounds and conditions such as dark, wood, cluttered and low-contrast scenes.



\- **\*\*Images:\*\*** \`sample_data/synthetic/\*.jpg\`

\- **\*\*Annotations:\*\*** \`sample_data/synthetic/\*.json\` (corner coordinates and ground-truth text for evaluation)

\- **\*\*Dataset guidance:\*\*** \`sample_data/README.md\` and \`evaluation/README.md\`

\- **\*\*Optional real-image dataset:\*\*** Add your own annotated images under \`sample_data/real/\` if available and permitted to share.



Generate or regenerate a synthetic dataset with:



\`\`\`bash

python -m evaluation.synthetic --out sample_data/synthetic --count 30

\`\`\`



Synthetic images are useful for repeatable testing, but results on synthetic data should not be presented as real-world accuracy measurements.



**## Step-by-step installation and execution**



**### Prerequisites**



1\. Install **\*\*Python 3.10–3.12\*\*** (the existing project documentation recommends this range).

2\. Install **\*\*Tesseract OCR\*\*** separately if you want text extraction. \`pytesseract\` is only a Python wrapper; it does not install the OCR executable. Confirm installation with \`tesseract --version\` and ensure the English language data is available.

3\. Download or clone this repository and open a terminal in the repository root (where \`app.py\` and \`requirements.txt\` are located).



**### Step 1 — Clone and enter the project**



\`\`\`bash

git clone https\://github.com/shruti240104/D_Team_13.git

cd D_Team_13

\`\`\`



**### Step 2 — Create a virtual environment**



\`\`\`bash

python -m venv .venv

\`\`\`



**### Step 3 — Activate the environment**



**\*\*Git Bash on Windows:\*\***



\`\`\`bash

source .venv/Scripts/activate

\`\`\`



**\*\*Windows PowerShell:\*\***



\`\`\`powershell

.\\.venv\Scripts\Activate.ps1

\`\`\`



**\*\*Windows Command Prompt:\*\***



\`\`\`bat

.venv\Scripts\activate.bat

\`\`\`



**\*\*Linux / macOS:\*\***



\`\`\`bash

source .venv/bin/activate

\`\`\`



**### Step 4 — Install Python dependencies**



\`\`\`bash

pip install -r requirements.txt

\`\`\`



**### Step 5 — Launch SmartScan**



\`\`\`bash

streamlit run app.py

\`\`\`



Open the local address shown by Streamlit (usually \`http\://localhost:8501\`).



**### Step 6 — Use the application**



1\. Upload a supported document photograph.

2\. Configure enhancement, OCR and orientation options as needed.

3\. Run the scan and inspect the detected page and corrected output.

4\. If needed, adjust document corners and rescan.

5\. Select the desired color, grayscale or BW version.

6\. Review OCR text and confidence indicators if Tesseract is available.

7\. Download the scan or extracted text as PNG, PDF or TXT.



**### Tesseract troubleshooting (Windows)**



If OCR reports that Tesseract cannot be found, install it and add its installation folder to \`PATH\`, or set its executable path in PowerShell:



\`\`\`powershell

$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"

\`\`\`



Check that the executable and language data are installed:



\`\`\`bash

tesseract --version

tesseract --list-langs

\`\`\`



**## Testing**



Run the complete test suite from the repository root:



\`\`\`bash

python -m pytest -v

\`\`\`



Run the tests that do not require Tesseract:



\`\`\`bash

python -m pytest -v -m "not requires_tesseract"

\`\`\`



Run enhancement tests specifically:



\`\`\`bash

python -m pytest tests/test_enhancement.py -v

\`\`\`



**## Evaluation**



\`\`\`bash

\# Document detection: IoU, corner error, detection success

python -m evaluation.detection_metrics sample_data/synthetic --csv detection.csv



\# Enhancement: contrast, Laplacian sharpness, background uniformity and montages

python -m evaluation.enhancement_metrics sample_data/synthetic --out enhancement_report



\# OCR: CER and WER against annotated ground-truth text (requires Tesseract)

python -m evaluation.ocr_metrics sample_data/synthetic --csv ocr.csv



\# Reliability: failure categories and processing time

python -m evaluation.reliability sample_data/synthetic --csv reliability.csv

\`\`\`



For enhancement, **\*\*higher text–paper contrast\*\***, **\*\*generally higher edge sharpness (with caution)\*\*** and **\*\*lower estimated background standard deviation\*\*** can indicate improvements. Sharpness alone is not proof of better OCR. **\*\*CER/WER\*\*** directly compare recognized text with ground truth; lower is better. See \`evaluation/README.md\` for methodology and limitations.



**## Team contributions**






| No. | Member | Module and contribution | Evidence |
|---|---|---|---|
| 1 | Srishti (20233278) | Page detection: blur/Canny/closing pipeline, contour filtering, corner ordering, candidate scoring, fallback | Annotated debug images, failure gallery, detection success table |
| 2 | Tanishqa Dhete (20233291) | Input validation, EXIF, scaling, homography warp, deskew, pipeline controller, error handling, synthetic benchmark | Geometry tests, error-handling matrix, benchmark generator |
| 3 | Shruti Singh (20233351) | Enhancement modes: grayscale, adaptive B&W, colour, CLAHE, denoising, illumination normalization, quality estimation | Ablation table, before/after gallery, shadow cases |
| 4 | Vishaka Meena (20233312) | Tesseract OCR, segmentation tuning, Streamlit UI, PNG/TXT/PDF export, CER/WER evaluation script | OCR experiment table, CER/WER results, UI demo |

## Presentation and Final Report

The project presentation (PPT) and final project report are added in the `docs/` folder.
