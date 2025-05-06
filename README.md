# 👗 Fashion Diffusion Studio

**AI-powered toolkit for generating and editing fashion images using state-of-the-art diffusion models.**

Fashion Diffusion Studio enables product designers, marketers, and developers to create and modify high-quality fashion visuals using text prompts, pose control, and inpainting. Built on top of Stable Diffusion, LoRA fine-tuning, and ControlNet, it allows for rapid prototyping of lookbooks, product mockups, and ad creatives—all without a traditional photoshoot.

---

## ✨ Features

- 🎨 **Text-to-Fashion Image Generation**  
  Generate outfits, models, and accessories using natural language prompts.

- 🧍 **Pose-Controlled Generation**  
  Use ControlNet to generate consistent human poses or clone campaign shots.

- ✂️ **Garment Inpainting**  
  Edit specific clothing items in a photo while preserving the original model.

- 🧠 **LoRA Fine-Tuning Pipeline**  
  Fine-tune lightweight diffusion adapters on your own fashion brand/style.

- 🖥️ **Streamlit Web App**  
  Try it out in an interactive UI for real-time generation and editing.

---

## 🖼️ Example Outputs

<!-- You can later replace these with real examples -->
| Prompt | Output |
|--------|--------|
| `a model wearing a red trench coat on a runway` | ![example1](app/assets/example1.jpg) |
| `change shirt to leather jacket` (inpainting) | ![example2](app/assets/example2.jpg) |

---

## 🧱 Folder Structure

```plaintext
fashion-diffusion-studio/
├── models/                         # Core model components
│   ├── base/                       # Base SD models (v1.5, inpainting, ControlNet)
│   ├── lora/                       # Fine-tuned LoRA weights/checkpoints
│   ├── dreambooth/                # DreamBooth weights, config
│   └── utils.py                   # Helper functions for model loading/running

├── data/                          # Dataset-related files
│   ├── raw/                       # Raw image and caption data
│   ├── processed/                 # Preprocessed datasets (resized, masked, tagged)
│   ├── annotations/              # Captions, pose maps, segmentation masks
│   └── metadata.json             # JSON to track versions, sources

├── training/                      # Training scripts and configs
│   ├── lora_train.py              # LoRA fine-tuning script
│   ├── dreambooth_train.py        # DreamBooth fine-tuning
│   ├── preprocess.py              # Image resizing, tagging, captioning
│   └── configs/                   # YAML configs for experiments

├── editing/                       # Image editing and inpainting logic
│   ├── inpaint.py                 # Garment editing via inpainting
│   ├── controlnet_edit.py        # Editing with ControlNet (pose, edge)
│   └── segmenter.py              # Garment segmentation or mask generator

├── generation/                    # Image generation pipeline
│   ├── text2img.py                # Prompt-based generation
│   ├── pose_guided_gen.py        # Generate images with pose control
│   └── prompt_templates.py       # Modular prompt templates for fashion items

├── app/                           # Web UI and demo app (Gradio or Streamlit)
│   ├── gradio_app.py              # Main app file
│   ├── assets/                    # Icons, default images
│   └── templates/                 # UI layout templates

├── backend/                       # Optional API backend (FastAPI)
│   ├── api.py                     # Inference endpoints
│   └── models.py                  # Pydantic schemas

├── infra/                         # Deployment & environment setup
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── install.sh                 # Dependency install script
│   └── e2e_launcher.py            # Launch automation (E2E Networks compatible)

├── notebooks/                     # Research, experimentation, and demos
│   ├── 01_lora_training.ipynb
│   ├── 02_inpainting_demo.ipynb
│   └── 03_prompt_experiments.ipynb

├── scripts/                       # Utility scripts (scraping, cleaning, etc.)
│   ├── scrape_dataset.py          # For collecting fashion images
│   └── generate_poses.py          # OpenPose or PoseNet extraction

├── tests/                         # Unit and integration tests
│   ├── test_generation.py
│   └── test_editing.py

├── configs/                       # Global config files
│   └── project_config.yaml

├── .env.example                   # Template for env vars (e.g., HF_TOKEN, API keys)
├── .gitignore
├── LICENSE
├── README.md                      # Project overview, instructions
├── requirements.txt               # Pip dependencies
└── environment.yml                # Conda environment file
```


## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/fashion-diffusion-studio.git
cd fashion-diffusion-studio
```

### 2. Set Up Environment
With Conda
```bash
conda env create -f environment.yml
conda activate fashion-diffusion
```

Or with pip
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app/streamlit_app.py
```

## ⚙️ Fine-Tuning (LoRA)
1. Prepare your dataset in `data/processed/`
2. Configure training in training/configs/lora_config.yaml
3. Run:
```bash 
python training/lora_train.py --config training/configs/lora_config.yaml
```

## 📦 Pretrained Models
| Model Type              | Source                                    |
| ----------------------- | ----------------------------------------- |
| Stable Diffusion 1.5    | `runwayml/stable-diffusion-v1-5`          |
| Inpainting              | `stabilityai/stable-diffusion-inpainting` |
| ControlNet (pose/canny) | `lllyasviel/ControlNet`                   |
| LoRA                    | Saved in `models/lora/` after training    |


## 🧪 Evaluation

- CLIP similarity scoring
- Attribute match (planned)
- Visual A/B comparison UI (coming soon)

## 📸 Use Cases

- Virtual try-on and style previews
- Social media asset generation
- Fashion product prototyping
- Brand lookbook automation

## 🛡️ License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributors
- Shreyas - Project Lead

## 📬 Contact
Questions or collaborations?\
📧 Email: shreyasdb99@gmail.com\
🌐 GitHub: shryzium
