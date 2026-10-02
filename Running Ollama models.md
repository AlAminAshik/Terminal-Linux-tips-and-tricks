\############### Downloading model from HuggingFace ###############



**Method 1 (better):** 

&#x09;*Cloning repo*: Running commands such as: `git clone https://huggingface.co/Qwen/QwenXX-XX`

&#x09;		However, this method does not show the download progress bar. but you get direct access to the model files, unlike downloading using hf.



**Method 2:**

&#x09;*Downloading using hf*:

&#x09;	> Install git-xet: `winget install git-xet` #needed only the first time for Windows

&#x09;	> Install huggingface:  `pip install huggingface\_hub` #needed only first time

&#x09;	> Download model: `hf download lightx2v/XXXX-XXX-XXX` #may need to disable smart app control for windows.

&#x09;	>The model will be downloaded in the root directory for Windows, such as "C:\\Users\\Alamin\\.cache\\huggingface\\hub\\"







\############### **The following is for Banana Pi M1 1GB** ###############



The following shows how to download, set up, and run Ollama models from Hugging Face. The 32-bit system is for Banana Pi, and the other systems also follow a similar pattern.

The downloaded file is in .gguf format so the model can be run directly from terminal without any python script.



Linux (32-bit armv7l)

&#x09;1) Install llama.cpp in a new directory (since full llama is not supported on Banana Pi)

&#x09;	`git clone https://github.com/ggerganov/llama.cpp`

&#x09;2) Make sure a C++ compiler is installed

&#x09;	`apt install -y build-essential cmake`

&#x09;3) Go to the llama.cpp directory and build llama.cpp. -j2 means use 2 cores.

&#x09;	`cmake --build build -j2`

&#x09;	''if cmake is not installed: sudo apt install cmake -y''

&#x09;4) After a long wait of build check the files and look for "llama-cli" under "ls build/bin/"

&#x09;	`cd /build/bin`

&#x09;5) Download a model from https://huggingface.co/ select a model and copy its Download link. save on a directory on the Linux device.

&#x09;	`wget https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q2\_K.gguf`

&#x09;6) Make sure enough space is available

&#x09;	`free -h`

&#x09;	`swapon --show`

&#x09;7) Finally, run the model.

&#x09;	 `\~/Myfolder/Ollamatest/llama.cpp/build/bin/llama-cli -m \~/Myfolder/Ollamatest/deepseekModels/DeepSeek-R1-Distill-Qwen-1.5B-Q2\_K.gguf -p "Explain what an embedded system is in simple terms." -n 64 --ctx-size 256 --threads 2 --batch-size 16 --no-mmap`



Note:

What each flag means when running the downloaded model:

\--ctx-size 256 → saves RAM

\--batch-size 16 → avoids memory spikes

\--no-mmap → prevents filesystem crashes on low RAM

\-n 64 → limits output tokens

\--threads 2 → matches Banana Pi cores 2)







\############### **The following is for the Orange Pi 5 plus 16GB** ###############



\*\*\*Note: the following is useful if the downloaded model does not have a .gguf file. In that case, a Python script is very useful.



This is a 64-bit board with 8-core processors. Contains on-board 16GB RAM along with a 6 TOPS NPU.

&#x09;1)	Install llama.cpp as a new directory (since full llama is not supported on bananapi)

&#x09;	`git clone https://github.com/ggerganov/llama.cpp`

&#x09;2)	Move to the llama directory: 

&#x09;	`cd llama.cpp`

&#x09;3)	Configure for ARM optimization so that it utilizes hardware vector extensions and dot product:

&#x09;	```

&#x09;	cmake -B build \\

&#x09;	-DCMAKE\_BUILD\_TYPE=Release \\

&#x09;	-DCMAKE\_C\_FLAGS="-march=armv8.2-a+dotprod+fp16fml" \\

&#x09;	-DCMAKE\_CXX\_FLAGS="-march=armv8.2-a+dotprod+fp16fml"

&#x09;	```

&#x09;4)	Run the build on 8 threads; this may take some time:

&#x09;	`cmake --build build --config Release -j8`

&#x09;5)	Download the model from Hugging Face by cloning the repo: `git clone https://huggingface.co/Qwen/Qwen3-4B`

&#x09;6)	Ensure you have the following installed:

&#x09;	`python3 --version`	//must be 3.10+

&#x09;	`pip show torch`	//must be 2.6.x or newer

&#x09;	`pip show transformers`	//must be 4.51.x or newer

&#x09;	`pip show tokenizers`	//must be installed the latest

&#x09;	`python3 -m pip --version` //must be setup

&#x09;7)	Create a Python script to run the model:

```

import torch

from transformers import (

&#x20;   AutoTokenizer,

&#x20;   AutoModelForCausalLM,

&#x20;   TextStreamer

)



MODEL\_PATH = "/home/alaminorange/Downloads/Qwen3-0.6B"



print("Loading tokenizer...")

tokenizer = AutoTokenizer.from\_pretrained(MODEL\_PATH)



print("Loading model...")

model = AutoModelForCausalLM.from\_pretrained(

&#x20;   MODEL\_PATH,

&#x20;   torch\_dtype="auto"

)



model.to("cpu")

model.eval()



print("\\nModel loaded!")

print("Type 'exit' to quit.\\n")



while True:



&#x20;   prompt = input("You: ")



&#x20;   if prompt.lower() in \["exit", "quit"]:

&#x20;       break



&#x20;   messages = \[

&#x20;       {"role": "user", "content": prompt}

&#x20;   ]



&#x20;   text = tokenizer.apply\_chat\_template(

&#x20;       messages,

&#x20;       tokenize=False,

&#x20;       add\_generation\_prompt=True,

&#x20;       enable\_thinking=False

&#x20;   )



&#x20;   inputs = tokenizer(

&#x20;       \[text],

&#x20;       return\_tensors="pt"

&#x20;   )



&#x20;   print("Qwen: ", end="", flush=True)



&#x20;   streamer = TextStreamer(

&#x20;       tokenizer,

&#x20;       skip\_prompt=True,

&#x20;       skip\_special\_tokens=True

&#x20;   )



&#x20;   with torch.no\_grad():

&#x20;       model.generate(

&#x20;           \*\*inputs,

&#x20;           streamer=streamer,

&#x20;           max\_new\_tokens=256,

&#x20;           temperature=0.7,

&#x20;           top\_p=0.8,

&#x20;           top\_k=20

&#x20;       )



&#x20;   print("\\n")

```

&#x09;8) Run the python script and the chatbot should start running.







\*\*\*Troubleshooting:\*\*\*



> Upgrade pillow if you get error like 'Could not import module 'Qwen3ForCausalLM'' by: python3 -m pip install --user --upgrade Pillow



