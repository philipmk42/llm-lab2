# 📝 Comparative Story Generation using Transformer Models

This Streamlit-based app allows users to compare the story generation capabilities of three popular transformer models:

- **GPT-2** – A decoder-only model by OpenAI.
- **FLAN-T5 Base** – A fine-tuned encoder-decoder model by Google.
- **DistilGPT-2** – A smaller, faster, and lighter GPT-2 variant.

---

## 🚀 Features

- Prompt-based story generation.
- Comparison of output across three different transformer models.
- Evaluation using automatic metrics (BLEU, ROUGE-L).
- Manual rating on fluency, coherence, and creativity.
- Visualizations using bar charts.

---

## 🛠️ Installation

### Clone the Repository


git clone https://github.com/your-username/story-gen-comparator.git
cd story-gen-comparator
Install Required Dependencies
Install with pip:

bash
Copy
Edit
pip install -r requirements.txt
Or manually install:

bash
Copy
Edit
pip install streamlit torch transformers evaluate nltk pandas seaborn matplotlib
🔐 Hugging Face Authentication
Some models may require authentication to download. You can authenticate using:

bash
Copy
Edit
huggingface-cli login
Or set your token:

bash
Copy
Edit
export HUGGINGFACE_TOKEN=your_token_here
On Windows (CMD):

cmd
Copy
Edit
set HUGGINGFACE_TOKEN=your_token_here
▶️ Run the App
bash
Copy
Edit
streamlit run lab2.py
Then open your browser to http://localhost:8501

📌 Example Prompts
"A lone astronaut discovers a hidden planet."

"A child finds an ancient map in their attic."

"Two strangers meet on a train and realize their destinies are linked."

📊 Evaluation Metrics
Metric	Description
BLEU	Compares n-gram overlap with reference text.
ROUGE-L	Measures longest common subsequence.
Human	Manual scoring on fluency, coherence, creativity.

📂 Project Structure
bash
Copy
Edit
├── lab2.py              # Main Streamlit application
├── README.md            # Project documentation
├── requirements.txt     # Python dependency file
└── screenshots/         # (Optional) Screenshots of UI
📃 License
This project is licensed under the MIT License.
Feel free to fork, improve, and share.

🙏 Acknowledgements
Hugging Face Transformers

Streamlit

Hugging Face Evaluate

⭐ If you like this project, please consider giving it a star!

yaml
Copy
Edit









