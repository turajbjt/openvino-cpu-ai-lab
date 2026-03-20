def generate(model, tokenizer, prompt, max_tokens=100):
    tokens = tokenizer.encode(prompt, return_tensors="np")

    output_tokens = tokens.tolist()[0]

    for _ in range(max_tokens):
        outputs = model.infer({"input_ids": tokens})
        next_token = outputs[0][0, -1].argmax()

        output_tokens.append(int(next_token))
        tokens = tokenizer.encode(output_tokens, return_tensors="np")

    return tokenizer.decode(output_tokens)
