from cactus import cactus_init, cactus_complete, cactus_destroy

model = cactus_init("weights/functiongemma-270m-it")

messages = [{"role": "user", "content": "Compute 2+2. Reply with only the number."}]
raw = cactus_complete(model, messages, max_tokens=32)

print("RAW:", repr(raw))
cactus_destroy(model)