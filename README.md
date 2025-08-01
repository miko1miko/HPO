## Directory Structure

The directory structure of this repository is organized as follows:

**Note**: For reproduction purposes, please refer to the specific sections below and follow the setup instructions for each component to ensure compatibility and proper functionality.

```
.
├── code_exec_server          # Code execution server
├── CoSec                     # CoSec related code
├── datasets                  # Datasets
│   ├── APPS                  # APPS dataset
│   ├── APPS_detect          # APPS test cases
│   └── hpo-pref             # HPO preference dataset
├── data-synthesis           # Data synthesis scripts
│   ├── functionality       # Functionality data synthesis
│   └── security            # Security data synthesis
├── instruction-generation   # Instruction generation scripts
│   ├── api-keys            # API keys configuration
│   ├── cwe-eclicitors      # CWE elicitors
│   ├── exact_question.py   # Exact question generation script
│   ├── exact_question.sh   # Exact question execution script
│   ├── gen_vul_inst.py     # Vulnerability instruction generation
│   ├── gen_vul_inst.sh     # Vulnerability instruction execution script
│   └── vul-inst            # Generated vulnerability instructions
├── model-train              # Model training scripts
│   ├── HPOconfig.py        # HPO configuration
│   ├── HPOtrainer.py       # HPO trainer implementation
│   ├── run_hpo.py          # HPO training script
│   ├── run_rpo.py          # RPO training script
│   ├── run_sft.py          # SFT training script
│   └── run.sh              # Training execution script
├── PurpleLlama              # PurpleLlama integration
├── test                     # Evaluation and testing scripts
│   ├── apps                # APPS evaluation
│   ├── cyberseceval        # CyberSecEval evaluation
│   ├── functionality      # Functionality evaluation
│   ├── humaneval          # HumanEval evaluation
│   └── security           # Security evaluation
└── vllm.sh                 # VLLM execution script
```

## 🔨 Setup

### Environment Setup
Create and activate a conda environment:

```bash
conda create -n hpo_env python==3.10
conda activate hpo_env
pip install -r requirements.txt
```

### Dependencies
- **PurpleLlama**: We use PurpleLlama from [meta-llama/PurpleLlama](https://github.com/meta-llama/PurpleLlama)
- **Code Execution Server**: We use the code execution server from [cassanof/code_exec_server](https://github.com/cassanof/code_exec_server)

#### Code Execution Server Setup
```bash
# Enter the container
docker exec -it container_name /bin/bash
cd ..
mkdir -p /tmp
cd /tmp
mkdir -p codeexec
cd codeexec
mkdir -p code_wait  # Store processed data files
pip install pyext
exit
docker cp datasets/APPS_detect/in_outs container_id:/tmp/codeexec
```

### Model Serving

All model generation in this project relies on vLLM for efficient inference serving. Start the model server using:

```bash
bash vllm.sh
```

This script will start a vLLM OpenAI-compatible API server that serves the model for all generation tasks throughout the pipeline.

**Note**: For reproduction purposes, please refer to the original implementations of these dependencies to ensure compatibility and proper setup.

## 🚀 Instruction Generation

Navigate to the `instruction-generation` directory to generate training instructions:

```bash
cd instruction-generation
# Generate vulnerability instructions
bash gen_vul_inst.sh
# Extract exact questions from responses
bash exact_question.sh
```

## 🔄 Data Synthesis

The `data-synthesis` module provides tools for synthesizing both functionality and security preference training data:

### Functionality

#### Generate Data
```bash
cd data-synthesis/functionality
# Generate rejected code
bash gen.sh
```

#### Functionality Evaluation
```bash
cd test/functionality
bash cor_check.sh
```

#### Repair Code
```bash
cd data-synthesis/functionality
# Generate chosen code
bash repair.sh
```

### Security

#### Generate Data
```bash
cd data-synthesis/security
# Generate rejected code
bash gen.sh
```

#### Security Evaluation
```bash
cd test/security
bash sec_check.sh
```

#### Repair Code
```bash
cd data-synthesis/security
# Generate chosen code
bash repair.sh
```


## 🎯 Model Training

The `model-train` directory contains scripts for different training approaches:

### HPO Training
```bash
cd model-train
python run_hpo.py --model_name "Qwen/Qwen2.5-Coder-7B-Instruct" --output_dir "./output/hpo"
```

### SFT Training
```bash
python run_sft.py --model_name "Qwen/Qwen2.5-Coder-7B-Instruct" --output_dir "./output/sft"
```

### RPO Training
```bash
python run_rpo.py --model_name "Qwen/Qwen2.5-Coder-7B-Instruct" --output_dir "./output/rpo"
```

## 🧪 Evaluation

The `test` directory provides comprehensive evaluation across multiple benchmarks:

### APPS

#### Generate Test Code
```bash
cd test/apps
bash gen_from_apps.sh
```

#### Run Evaluation
```bash
bash apps_check.sh
```

### CyberSecEval

#### Setup
1. Copy the language-specific subdatasets from `test/cyberseceval` to `PurpleLlama/CybersecurityBenchmarks/datasets/instruct/`
2. Modify `PurpleLlama/CybersecurityBenchmarks/benchmark/llm.py` in the `class OPENAI(LLM)` section by adding:
   ```python
   base_url=os.getenv("base_url")
   ```

#### Run Evaluation
```bash
cd test/cyberseceval
bash example.sh
```

For more detailed testing procedures, please refer to the [PurpleLlama repository](https://github.com/meta-llama/PurpleLlama).

### HumanEval(+)

#### Setup
Refer to the [EvalPlus repository](https://github.com/evalplus/evalplus) for setup instructions.

#### Run Evaluation
```bash
cd test/humaneval
bash example.sh
```

For more detailed testing procedures, please refer to the [EvalPlus repository](https://github.com/evalplus/evalplus).

## 🔧 CoSec Integration

Due to compatibility issues with older transformers versions in the original paper, we have updated the core code file `CustomizedGeneration.py` to work with newer transformers versions.

## 📚 Acknowledgments

We gratefully acknowledge the use of evaluation frameworks and datasets from the following sources:

- **PurpleLlama**: Security evaluation tools and benchmarks from [Meta's PurpleLlama project](https://github.com/meta-llama/PurpleLlama)
- **Code Execution Server**: Safe code execution environment from [cassanof/code_exec_server](https://github.com/cassanof/code_exec_server)
- **EvalPlus**: Enhanced HumanEval evaluation framework from [evalplus/evalplus](https://github.com/evalplus/evalplus)
