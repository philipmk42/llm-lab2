# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import streamlit as st
import evaluate
import nltk
import pandas as pd

nltk.download('punkt')

# Model metadata
MODELS = {
    "GPT-2": {
        "model_name": "gpt2",
        "causal": True
    },
    "FLAN-T5": {
        "model_name": "google/flan-t5-base",
        "causal": False
    },
    "DistilGPT-2": {
        "model_name": "distilgpt2",
        "causal": True
    }
}

@st.cache_resource
def load_model(model_info):
    tokenizer = AutoTokenizer.from_pretrained(model_info["model_name"])

    # Set pad_token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_info["causal"]:
        model = AutoModelForCausalLM.from_pretrained(model_info["model_name"])
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_info["model_name"])

    return model, tokenizer

def preprocess_prompt(prompt, model_type):
    prompt = prompt.strip()
    if model_type == "GPT-2" or model_type == "DistilGPT-2":
        return f"Once upon a time, {prompt}"
    elif model_type == "FLAN-T5":
        return f"Write a story about: {prompt}"
    return prompt

def generate_story(model_name, prompt, max_length, temperature, top_p):
    model_info = MODELS[model_name]
    model, tokenizer = load_model(model_info)
    formatted_prompt = preprocess_prompt(prompt, model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    inputs = tokenizer(
        formatted_prompt, return_tensors="pt", truncation=True, padding=True
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if model_info["causal"]:
        return generated_text.replace(formatted_prompt, "").strip()[:500]
    else:
        return generated_text[:500]

# Evaluation metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

def evaluate_story(reference, generated):
    bleu_score = bleu.compute(predictions=[generated], references=[[reference]])["bleu"]
    rouge_score = rouge.compute(predictions=[generated], references=[reference])
    return {"bleu": bleu_score, "rouge": rouge_score}

# Main Streamlit App
def main():
    st.title("📝 Comparative Story Generation using Transformer Models")

    st.markdown("""
    ### 🚀 Project Overview
    This app compares **story generation** from three lightweight transformer models:
    - GPT-2
    - FLAN-T5 (Google)
    - DistilGPT-2 (Compact GPT-2)

    Evaluate each model using BLEU, ROUGE, and manual ratings.
    """)

    sample_prompts = [
        "A lone astronaut discovers a hidden planet.",
        "A child finds an ancient map in their attic.",
        "Two strangers meet on a train and realize their destinies are linked."
    ]

    selected_prompt = st.selectbox("Choose a sample prompt", sample_prompts)
    prompt = st.text_input("Or enter your own prompt", selected_prompt)

    max_length = st.slider("Max Story Length", 100, 500, 300)
    temperature = st.slider("Temperature", 0.5, 1.5, 0.7)
    top_p = st.slider("Top-p (nucleus sampling)", 0.5, 1.0, 0.9)

    if st.button("Generate and Compare"):
        st.subheader("📚 Generated Stories and Evaluation")
        auto_metrics = {"Model": [], "BLEU": [], "ROUGE-L": []}
        human_metrics = {"Model": [], "Fluency": [], "Coherence": [], "Creativity": [], "Average": []}

        # Reference story for auto-metric evaluation (mock reference for comparison)
        reference = "This is a placeholder reference story used for BLEU and ROUGE scoring."

        for model_name in MODELS:
            with st.spinner(f"Generating story with {model_name}..."):
                story = generate_story(model_name, prompt, max_length, temperature, top_p)
                metrics = evaluate_story(reference, story)
                auto_metrics["Model"].append(model_name)
                auto_metrics["BLEU"].append(metrics["bleu"])
                auto_metrics["ROUGE-L"].append(metrics["rouge"]["rougeL"])

                st.markdown(f"### 🔹 {model_name}")
                st.text(story)
                st.write(f"**BLEU Score:** {metrics['bleu']:.4f}")
                st.write(f"**ROUGE-L:** {metrics['rouge']['rougeL']:.4f}")

                with st.expander(f"Rate {model_name} (Human Evaluation)"):
                    fluency = st.slider(f"{model_name} - Fluency", 1, 5, 3, key=f"{model_name}_f")
                    coherence = st.slider(f"{model_name} - Coherence", 1, 5, 3, key=f"{model_name}_c")
                    creativity = st.slider(f"{model_name} - Creativity", 1, 5, 3, key=f"{model_name}_cr")
                    avg = (fluency + coherence + creativity) / 3
                    human_metrics["Model"].append(model_name)
                    human_metrics["Fluency"].append(fluency)
                    human_metrics["Coherence"].append(coherence)
                    human_metrics["Creativity"].append(creativity)
                    human_metrics["Average"].append(avg)

        # Visualization
        st.subheader("📊 Comparative Visualization")
        df_auto = pd.DataFrame(auto_metrics).set_index("Model")
        st.markdown("### 🤖 Automatic Evaluation: BLEU & ROUGE-L")
        st.bar_chart(df_auto)

        df_human = pd.DataFrame(human_metrics).set_index("Model")
        st.markdown("### 👤 Human Evaluation: Fluency, Coherence, Creativity")
        st.bar_chart(df_human[["Fluency", "Coherence", "Creativity"]])

        st.markdown("### 🌟 Average Human Ratings")
        st.bar_chart(df_human[["Average"]])

        with st.expander("🔍 Show Raw Metric Tables"):
            st.write("Automatic Metrics", df_auto)
            st.write("Human Evaluation", df_human)

if __name__ == "__main__":
    main()
