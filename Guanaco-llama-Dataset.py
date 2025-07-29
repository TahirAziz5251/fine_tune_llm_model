from datasets import load_dataset
import re

#load the dataset
dataset = load_dataset("timdettmers/openassistant-guanaco")
# shuffle the dataset 
dataset = dataset["train"].shuffle(seed=42).select(range(1000))
 #define a function to transform the data

def transform_conversation(example):
    conversation_text = example["text"]
    segments = conversation_text.split('###')
    reformatted_segments = []
    # Iterate over pairs of segments
    for i in range(1, len(segments) - 1, 2):
        human_text = segments[i].strip().replace("Human:", "").strip()
    # Check if ther is corresponding assistant segment before processing
        if i + 1 < len(segments):
            assistant_text = segments[i+1].strip().replace("Assistant:", "").strip()
            # Apply the new template
            reformatted_segments.append(f"<s>[INST] {human_text}[/INST] {assistant_text}</s>")
        else:
            #Handle the case where there is no corresponding assistant segment
            reformatted_segments.append(f"<s>[INST] {human_text} [/INST]</s>")

    return {"text": "".join(reformatted_segments)}
#Apply the transformation
transformed_dataset = dataset.map(transform_conversation)