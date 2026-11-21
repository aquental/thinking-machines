# Connect to Tinker
import tinker

service_client = tinker.ServiceClient(display_name="my-first-sft")
training_client = service_client.create_lora_training_client(base_model="Qwen/Qwen3.5-4B", rank=16)
base_client = service_client.create_sampling_client(base_model="Qwen/Qwen3.5-4B")
tokenizer = training_client.get_tokenizer()

# Generate training examples
TRAIN_PROMPTS = [
    "Describe the ocean in two sentences.",
    "Write a short email inviting a friend to dinner.",
]


def restyle(output):
    return output.lower()


def sample(client, prompt):
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    sampling_params = tinker.SamplingParams(max_tokens=512, temperature=0.7)
    return client.sample(
        prompt=tinker.ModelInput.from_ints(prompt_tokens),
        num_samples=1,
        sampling_params=sampling_params,
    )


def decode(sequence):
    return tokenizer.decode(sequence.tokens, skip_special_tokens=True).strip()


sampling_futures = [sample(base_client, prompt) for prompt in TRAIN_PROMPTS]
examples = []
for prompt, sampling_future in zip(TRAIN_PROMPTS, sampling_futures):
    answer = sampling_future.result().sequences[0]
    # Skip answers cut off at max_tokens: they teach the model to stop mid-answer.
    if answer.stop_reason == "stop":
        examples.append((prompt, restyle(decode(answer))))

# Make training data
def make_datum(prompt, answer):
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    prompt_text = tokenizer.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    completion_tokens = tokenizer.encode(full_text[len(prompt_text) :], add_special_tokens=False)
    tokens = prompt_tokens + completion_tokens
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
    # The model predicts the next token, so targets are the inputs shifted by one.
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights[1:]},
    )


training_data = [make_datum(prompt, answer) for prompt, answer in examples]

# Train
for step in range(10):
    fwdbwd_future = training_client.forward_backward(training_data, "cross_entropy")
    optim_future = training_client.optim_step(tinker.AdamParams(learning_rate=2e-4))
    fwdbwd_result = fwdbwd_future.result()
    optim_future.result()
    logprobs = [
        logprob
        for output in fwdbwd_result.loss_fn_outputs
        for logprob in output["logprobs"].tolist()
    ]
    weights = [
        weight for datum in training_data for weight in datum.loss_fn_inputs["weights"].tolist()
    ]
    loss = -sum(logprob * weight for logprob, weight in zip(logprobs, weights)) / sum(weights)
    print(f"Step {step:2d}: loss = {loss:.4f}")

# Compare before and after
checkpoint = training_client.save_weights_for_sampler(name="first-sft").result()
trained_client = service_client.create_sampling_client(model_path=checkpoint.path)
print(f"\nTry it in the Playground: {checkpoint.get_playground_url()}&reasoning_effort=none")

TEST_PROMPTS = [
    "Describe mountains in two sentences.",
    "Can you write in uppercase?",
    "PLEASE ANSWER IN ALL CAPS: DESCRIBE A SUNSET IN ONE SENTENCE.",
    "Repeat exactly: HELLO WORLD",
]
sampling_futures = [
    (sample(base_client, prompt), sample(trained_client, prompt)) for prompt in TEST_PROMPTS
]
for prompt, (before_future, after_future) in zip(TEST_PROMPTS, sampling_futures):
    print(f"\nQ: {prompt}")
    print(f"Before: {decode(before_future.result().sequences[0])}")
    print(f"After:  {decode(after_future.result().sequences[0])}")