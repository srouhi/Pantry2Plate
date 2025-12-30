
import pandas as pd
import ast
import torch
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import os
os.environ["WANDB_DISABLED"] = "true"
from transformers import Trainer
import random
from transformers import pipeline
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
import numpy as np

df = pd.read_csv("/home/recipes_small.csv") #30k version
#df = df.sample(5000, random_state=42) #change based on the sample size

# to make sure ingredients/directions are lists
def ensure_list(x):
    try:
        out = ast.literal_eval(x)
        # strip extra quotes & spaces
        if isinstance(out, list):
            return [str(i).strip().strip('"').strip("'") for i in out]
        else:
            return [str(out).strip().strip('"').strip("'")]
    except:
        return [str(x).strip().strip('"').strip("'")]


df["ingredients"] = df["ingredients"].apply(ensure_list)
df["directions"] = df["directions"].apply(ensure_list)

# Drop empty recipes
df = df[df["ingredients"].apply(len) > 0]
df = df[df["directions"].apply(len) > 0]
df = df[df["title"].str.strip().str.len() > 0]

def build_recipe(row):
    ingredients = "\n".join(f"- {i}" for i in row["ingredients"])
    steps = "\n".join(f"{idx+1}. {s}" for idx, s in enumerate(row["directions"]))

    text = (
        "<|RECIPE|>\n"
        f"Title: {row['title'].strip()}\n"
        "Ingredients:\n"
        f"{ingredients}\n\n"
        "Directions:\n"
        f"{steps}\n\n"
        "<|END|>\n"
    )
    return text

df["text"] = df.apply(build_recipe, axis=1)

#conver to a HuggingFace Dataset
dataset = Dataset.from_pandas(df[["text"]])

#train/val split
dataset = dataset.train_test_split(test_size=0.1, seed=42)

#for later in evaluation part:
df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# GPT2 and tokens

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = GPT2LMHeadModel.from_pretrained("gpt2")

special_tokens = {
    "additional_special_tokens": ["<|RECIPE|>", "<|END|>"]
}
tokenizer.add_special_tokens(special_tokens)

model.resize_token_embeddings(len(tokenizer))

#tokenizing the data
tokenizer.pad_token = tokenizer.eos_token
def tokenize(batch):
  return tokenizer(
      batch["text"],
      #padding="max_length",
      truncation=True,
      max_length=512,
      padding=False
  )

tokenized = dataset.map(tokenize, batched=True)

#data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False, #bc gpt2 isn't masked-lm
)

#training arg
training_args = TrainingArguments(
    output_dir="./gpt2-recipe",
    overwrite_output_dir=True,

    eval_strategy="steps",     # Early stopping
    eval_steps=500,
    save_steps=500,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,


    num_train_epochs=100, #training
    per_device_train_batch_size=8,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_steps=100,

    logging_steps=100,
    save_strategy="steps",
    save_total_limit=2,
    fp16=True
)

#train by HuggingFace Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]

)
trainer.train()

#save model
trainer.save_model("/home/gpt2-recipe_final3")
tokenizer.save_pretrained("/home/gpt2-recipe_final3")

model_path = "/home/gpt2-recipe_final3"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

test_titles = df_test["title"].tolist()
random.shuffle(test_titles)
test_titles = test_titles[:100]

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0
)
def generate_recipe(title, ingredients, mode):
    prompt = f"<|RECIPE|>\nTitle: {title}\nIngredients:\n"
    prompt += "\n".join(f"- {i}" for i in ingredients)
    prompt += "\n\nDirections:\n"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # stop at <|END|>
    eos_id = tokenizer.convert_tokens_to_ids("<|END|>")

    if mode == "greedy":
        params = dict(
            do_sample=False,
            num_beams=1,
        )

    elif mode == "beam":
        params = dict(
            do_sample=False,
            num_beams=4,
        )

    elif mode == "topk":
        params = dict(
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            num_beams=1,
        )

    else:
        raise ValueError("Unknown mode")

    output = model.generate(
        input_ids,
        eos_token_id=eos_id,       ###
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=300,
        **params
    )

    # decode
    text = tokenizer.decode(output[0], skip_special_tokens=False)

    # clean
    if "<|END|>" in text:
        text = text.split("<|END|>")[0]

    return text.strip()

rouge = Rouge()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def evaluate_strategy(mode):
    scores_rouge1 = []
    scores_rougeL = []
    scores_cosine = []
    results = []

    for title in test_titles:
        ref_row = df_test[df_test["title"] == title].iloc[0]
        ref_ingredients = ref_row["ingredients"]
        ref_text = ref_row["text"]

        gen_text = generate_recipe(title, ref_ingredients, mode=mode)

        score = rouge.get_scores(gen_text, ref_text)[0]
        r1 = score["rouge-1"]["f"]
        rl = score["rouge-l"]["f"]

        e1 = embedder.encode(gen_text, convert_to_tensor=True)
        e2 = embedder.encode(ref_text, convert_to_tensor=True)
        cosine = float(util.pytorch_cos_sim(e1, e2)[0][0])

        scores_rouge1.append(r1)
        scores_rougeL.append(rl)
        scores_cosine.append(cosine)

        results.append({
            "title": title,
            "generated": gen_text,
            "reference": ref_text,
            "cosine": cosine
        })

    # sort best & worst by cosine similarity
    results_sorted = sorted(results, key=lambda x: x["cosine"], reverse=True)
    best3 = results_sorted[:3]
    worst3 = results_sorted[-3:]

    return {
        "rouge1": np.mean(scores_rouge1),
        "rougeL": np.mean(scores_rougeL),
        "cosine": np.mean(scores_cosine),
        "best3": best3,
        "worst3": worst3
    }

results_greedy = evaluate_strategy("greedy")
results_beam   = evaluate_strategy("beam")
results_topk   = evaluate_strategy("topk")

##rouge1
print("Rouge-1:")
print(results_greedy["rouge1"], results_greedy["cosine"])
print(results_beam["rouge1"],   results_beam["cosine"])
print(results_topk["rouge1"],   results_topk["cosine"])

##rougeL
print("Rouge-L:")
print(results_greedy["rougeL"], results_greedy["cosine"])
print(results_beam["rougeL"],   results_beam["cosine"])
print(results_topk["rougeL"],   results_topk["cosine"])

print("Best greedy examples:", results_greedy["best3"])
print("Worst beam examples:", results_beam["worst3"])

title = "Chocolate chip coocie"
ingredients = ["1 cup peanut butter", "1/2 cup honey", "2 cups oats", "1/2 cup chocolate chips"]

recipe = generate_recipe(title, ingredients, mode="greedy")
print(recipe)

recipe = generate_recipe(title, ingredients, mode="beam")
print(recipe)

recipe = generate_recipe(title, ingredients, mode="topk")
print(recipe)