## Test lambda

sls invoke local -f promptOptimization -p test.json
sls invoke -f promptOptimization -p test.json

## Notes

Into context, take in

1. Wandb feedback of run
2. Conversation
3. Contents of files
4. Human Guidelines / Goals
5. Current suggestions / okrs on dashboard
6. Prompts being used
7. Previous results

Can do:

1. Open Github Issue
2. Update prompts

Show:

1. Evaluation results mapping to prompts

# Prompt Manager

A Streamlit application for managing and storing prompts using Weights & Biases (W&B) as a backend.

## Features

- Create and save prompts to W&B
- View and edit existing prompts
- Version control through W&B artifacts
- Real-time debug information

## Prerequisites

- Python 3.7+
- pip (Python package manager)
- Weights & Biases account

## Installation

1. Clone the repository and navigate to the project directory:

```bash
cd prompt-manager
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

Required packages:

- streamlit
- wandb
- pandas>=2.2.0
- python-dotenv>=1.0.1

3. Set up environment variables:

Option 1: Using environment variables directly:

```bash
export WANDB_API_KEY="your_wandb_api_key"
export WANDB_PROJECT="your_project_name"
export WANDB_ENTITY="your_wandb_username"
```

For Windows, use:

```cmd
set WANDB_API_KEY=your_wandb_api_key
set WANDB_PROJECT=your_project_name
set WANDB_ENTITY=your_wandb_username
```

Option 2: Using a .env file:
Create a `.env` file in the project root:

```env
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT=your_project_name
WANDB_ENTITY=your_wandb_username
```

## Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Usage

1. **Create Prompts**

   - Navigate to the "Create Prompt" tab
   - Enter a name and content for your prompt
   - Click "Save Prompt" to store it in W&B

2. **View/Edit Prompts**
   - Go to the "View Prompts" tab
   - Expand any prompt to view or edit its content
   - Click "Update" to save changes

## Debug Information

The application provides real-time debug information in the right sidebar, including:

- W&B connection status
- Environment variables
- Operation logs

## Environment Variables

- `WANDB_API_KEY`: Your W&B API key
- `WANDB_PROJECT`: W&B project name
- `WANDB_ENTITY`: W&B username/organization
- `WANDB_MODE`: (Optional) W&B mode (defaults to online)
