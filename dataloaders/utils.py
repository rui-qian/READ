import torch
from model.llava import conversation as conversation_lib
from model.llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from model.llava.mm_utils import tokenizer_image_token


def pad_sequence_to_max_length(sequence, max_length, padding_value=0):
    """Pad a sequence to the desired max length."""
    if len(sequence) >= max_length:
        return sequence
    return torch.cat(
        [
            torch.full(
                (max_length - len(sequence),), padding_value, dtype=sequence.dtype
            ),
            sequence,
        ]
    )


def replace_image_tokens(conversation_list):
    """
    Replace <image> tokens in the conversation list with start and end image tokens.
    """
    for i in range(len(conversation_list)):
        replace_token = DEFAULT_IMAGE_TOKEN
        replace_token = (
            DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
        )
        conversation_list[i] = conversation_list[i].replace(
            DEFAULT_IMAGE_TOKEN, replace_token
        )
    return conversation_list


def tokenize_and_pad(conversation_list, tokenizer, padding="right"):
    """
    Tokenize and pad the conversation list.
    Args:
        conversation_list: A list of conversation prompts to be tokenized.
        tokenizer: The tokenizer to use for tokenizing the prompts.
        padding: The direction of padding, either "right" or "left".
    Returns:
        Tuple of input_ids and attention_masks.
    """
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt").squeeze(0) for prompt in conversation_list]
    if padding == "left":
        max_len = max(len(seq) for seq in input_ids)
        input_ids = [pad_sequence_to_max_length(seq, max_len, tokenizer.pad_token_id) for seq in input_ids]
        input_ids = torch.stack(input_ids, dim=0)
    else:
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

    attention_masks = input_ids.ne(tokenizer.pad_token_id)
    return input_ids, attention_masks


def handle_conversation_specifics(input_ids, conversation_list, tokenizer, conv_type):
    """
    Generate targets for the model and handle conversation specifics.
    """
    # Create a copy of the default conversation structure
    conv = conversation_lib.default_conversation.copy()
    # Initialize targets with a clone of input_ids
    targets = input_ids.clone()
    # Define the separator based on conversation type
    sep = conv.sep + conv.roles[1] + ": " if conv_type == "llava_v1" else "[/INST] "
    
    # Iterate through each conversation in the list and update targets
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for rou in rounds:
            if rou == "":
                break

            parts = rou.split(sep)
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            # Mark the instruction part as IGNORE_INDEX in the target
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len
    return targets