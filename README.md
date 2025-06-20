# NLP_TextSummarizer

## Project Overview
NLP_TextSummarizer is a machine learning project for text summarization using a pre-trained Pegasus model from Hugging Face. The project implements an end-to-end NLP pipeline, including data ingestion, transformation, model training, and evaluation, with a FastAPI-based web application for generating summaries. It leverages the SAMSum dataset for training and evaluation.

This project demonstrates skills in:
- **Data Manipulation**: Using Pandas and Hugging Face `datasets` for data processing.
- **Deep Learning**: Fine-tuning a Pegasus model with the Hugging Face `transformers` library.
- **MLOps**: Modular pipelines for scalability and deployment with FastAPI and Docker.
- **Web Development**: Serving predictions via a FastAPI app.

The dataset is sourced from [SAMSum](https://github.com/krishnaik06/datasets/raw/refs/heads/main/summarizer-data.zip).

## Repository Structure
```
NLP_TextSummarizer/
├── .gitignore                  # Ignored files
├── app.py                      # FastAPI web app for predictions
├── artifacts/                  # Model, data, and metrics storage
├── config/
│   └── config.yaml             # Configuration file
├── Dockerfile                  # Docker setup for deployment
├── LICENSE                     # License file
├── logs/                       # Log files
├── main.py                     # Main script for running pipelines
├── params.yaml                 # Hyperparameters
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── research/                   # Jupyter notebooks for experimentation
│   ├── 1_data_ingestion.ipynb
│   ├── 2_data_transformation.ipynb
│   ├── 3_model_trainer.ipynb
│   ├── 4_model_evaluation.ipynb
│   ├── research.ipynb
│   └── textsummarizer.ipynb
├── setup.py                    # Package setup script
├── src/textSummarizer/         # Source code
│   ├── __init__.py
│   ├── components/            # ML components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_evaluation.py
│   │   └── model_trainer.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── configuration.py
│   ├── constants/
│   │   └── __init__.py
│   ├── entity/
│   │   └── __init__.py
│   ├── logging/
│   │   └── __init__.py
│   ├── pipeline/              # ML pipelines
│   │   ├── __init__.py
│   │   ├── prediction_pipeline.py
│   │   ├── stage_1_data_ingestion_pipeline.py
│   │   ├── stage_2_data_transformation_pipeline.py
│   │   ├── stage_3_model_trainer_pipeline.py
│   │   └── stage_4_model_evaluation.py
│   └── utils/
│       ├── __init__.py
│       └── common.py
├── template.py                 # Template generation script
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Monish-Nallagondalla/NLP_TextSummarizer.git
   cd NLP_TextSummarizer
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   **Note**: Ensure `transformers` is updated to the latest version (e.g., >=4.38) to avoid errors like `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'`. Run:
   ```bash
   pip install --upgrade transformers
   ```
4. Download the SAMSum dataset as specified in `config.yaml`.

## Usage
1. **Exploratory Analysis**:
   - Open `research/textsummarizer.ipynb` to explore the dataset and model setup.
   - The SAMSum dataset contains dialogues and summaries for training a text summarization model.

2. **Pipeline Execution**:
   - Run the full pipeline using:
     ```bash
     python main.py
     ```
   - Stages:
     - **Data Ingestion**: Downloads and unzips the SAMSum dataset (`stage_1_data_ingestion_pipeline.py`).
     - **Data Transformation**: Preprocesses data using the Pegasus tokenizer (`stage_2_data_transformation_pipeline.py`).
     - **Model Training**: Fine-tunes the Pegasus model (`stage_3_model_trainer_pipeline.py`).
     - **Model Evaluation**: Evaluates performance using metrics like ROUGE (`stage_4_model_evaluation.py`).

3. **Prediction**:
   - Run the FastAPI app for text summarization:
     ```bash
     python app.py
     ```
   - Access the app at `http://localhost:8000` to input text and generate summaries.
   - Uses `prediction_pipeline.py` to load the trained model and tokenizer.

4. **Docker Deployment**:
   - Build and run the Docker container:
     ```bash
     docker build -t text-summarizer .
     docker run -p 8000:8000 text-summarizer
     ```

## Key Features
- **Data Ingestion**: Downloads and extracts the SAMSum dataset from a URL.
- **Data Transformation**: Uses Hugging Face `transformers` for tokenization and preprocessing.
- **Model Training**: Fine-tunes the Pegasus model for summarization using `TrainingArguments`.
- **Model Evaluation**: Computes ROUGE and other metrics, saved to `╭┬┤┘
- **Web Interface**: FastAPI app for real-time text summarization.
- **MLOps**: Modular pipelines and Docker support for scalable deployment.

## Requirements
Key dependencies (see `requirements.txt` for full list):
- `transformers` (Hugging Face library for NLP models)
- `datasets` (Hugging Face dataset handling)
- `sacrebleu`, `rouge_score` (evaluation metrics)
- `pandas` (data manipulation)
- `torch` (PyTorch for model training)
- `fastapi`, `uvicorn` (web app)
- `boto3`, `mypy-boto3-s3` (AWS integration)

## Notes
- **Dependency Management**: Ensure `transformers` is updated to avoid issues like the `evaluation_strategy` error. Use `pip install --upgrade transformers` for compatibility.
- **AWS Integration**: The project includes `boto3` for potential S3 storage of artifacts or models.
- **Metrics**: ROUGE scores are used to evaluate summarization quality, stored in `artifacts/model_evaluation/metrics.csv`.

## Contributing
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`.
3. Make changes and commit: `git commit -m "Add feature"`.
4. Push to the branch: `git push origin feature-branch`.
5. Create a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or suggestions, contact [Monish Nallagondalla](mailto:nsmonish@gmail.com) or open an issue on GitHub.
