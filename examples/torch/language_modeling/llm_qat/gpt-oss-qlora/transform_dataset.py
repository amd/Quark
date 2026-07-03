#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from datasets import Dataset, DatasetDict, load_dataset

"""
    NOTE: the messages should in this format
    [
        {
            "content": "Can you show me the latest trends on Twitter right now?",
            "role": "user",
        },
        {   "content": "Hey there! While I can't check Twitter (X) in real-time or access live dat",
            "role": "assistant",
            "thinking": "The user asking for ...."
        }
    ]
"""
false_reject_dataset = load_dataset("AmazonScience/FalseReject", cache_dir="PATH_DIR/hug_data")
train_data = []
for each_data in false_reject_dataset["train"]:
    union_conversion = {}

    user_conv = {"content": each_data["prompt"], "role": "user", "thinking": None}

    assistant_conv = {
        "content": each_data["cot_response"]["solution"],
        "role": "assistant",
        "thinking": each_data["cot_response"]["reasoning_content"],
    }
    union_conversion["messages"] = [user_conv, assistant_conv]
    train_data.append(union_conversion)


train_ds = Dataset.from_list(train_data)

dataset = DatasetDict(
    {
        "train": train_ds,
    }
)

dataset.save_to_disk("./local_false_reject_dataset")
# you can also push to your own huggingface repo
# dataset.push_to_hub("YOUR_NAME/false_reject", private=True)
print(1)
