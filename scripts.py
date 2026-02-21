'''
import json
from cactus import cactus_init, cactus_complete, cactus_destroy

model = cactus_init("weights/functiongemma-270m-it")
messages = [{"role": "user", "content": "What is 2+2?"}]
response = json.loads(cactus_complete(model, messages))
print(response["response"])

cactus_destroy(model)
'''


from cactus import cactus_init, cactus_complete, cactus_destroy
import json

model = cactus_init("weights/functiongemma-270m-it")
messages = [{"role": "user", "content": "Compute 2+2 and reply with only the number."}]

raw = cactus_complete(model, messages)
print(raw)

obj = json.loads(raw)
print(obj)

cactus_destroy(model)