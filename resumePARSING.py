import math
import PyPDF2
import os
from os.path import isfile, join
from io import StringIO
import pandas as pd
from collections import Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pdfplumber
from datetime import datetime

# Load spaCy model with embeddings
nlp = spacy.load("en_core_web_lg")

# Function to read PDFs and extract text
def pdfextract(file):
    fileReader = PyPDF2.PdfReader(open(file, 'rb'))
    text = []
    for page in fileReader.pages:
        t = page.extract_text()
        text.append(t)
    return " ".join(text)

# Extract years of experience from text
def extract_years_of_experience(resume_text):
    total_years = 0.0
    year_pattern = re.compile(
        r"(?P<years>\d+(?:\.\d+)?)(?:\+|-)?\s*(?:years?|yrs?\.?)?\s*(?:of\s*)?experience|"
        r"(?P<start_month>[a-zA-Z]{3,9})?\s*(?P<start_year>\d{4})\s*[-to]+\s*(?P<end_month>[a-zA-Z]{3,9})?\s*(?P<end_year>\d{4})\s*(?:experience)?",
        re.IGNORECASE
    )
    for match in year_pattern.finditer(resume_text):
        if match.group("years"):
            total_years += float(match.group("years"))
        elif match.group("start_year") and match.group("end_year"):
            start_year = int(match.group("start_year"))
            end_year = int(match.group("end_year"))
            start_month = match.group("start_month") if match.group("start_month") else "Jan"
            end_month = match.group("end_month") if match.group("end_month") else "Dec"
            try:
                start_date = datetime.strptime(f"{start_month} {start_year}", "%b %Y")
                end_date = datetime.strptime(f"{end_month} {end_year}", "%b %Y")
                difference_in_years = (end_date - start_date).days / 365.25
                total_years += difference_in_years
            except ValueError:
                pass
    return total_years


# Get text and TF-IDF vectorizer for job description and resumes
def extract_job_description_keywords():
    job_description = """
        machine learning, clustering, logistic regression, classification, scikit learn, data science methodology, model selection, model evaluation,
        pyspark, hadoop, big data pipelines, deep learning, tensorflow, keras, pytorch, neural networks, cnn, rnn, gans, model deployment, nlp,
        natural language processing, nltk, spacy, beautiful soup, transformers, nlp architectures, deploying nlp models, computer vision,
        object detection, ocr, opencv, cnn, python, numpy, pandas, matplotlib, seaborn, sql, mongodb, nosql, graphql, aws, azure,
        google cloud platform, firebase, django, phd, mtech, mca, bca, b.tech, bsc, B.CA
    """
    return job_description

# Score resumes based on TF-IDF and embeddings
def score_resume_with_embeddings(resume_text, job_description):
    print("Resume Text:", resume_text)
    print("Job Description:", job_description)
    # Get TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    vectors = tfidf.fit_transform([resume_text, job_description])
    print("TF-IDF Vectors (dense):", vectors.toarray()) # Convert sparse matrix to dense for easier reading

    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2]).flatten()[0]
    print("TF-IDF Cosine Similarity:", cosine_sim)

    # Compute embeddings
    resume_embedding = nlp(resume_text).vector
    jd_embedding = nlp(job_description).vector
    embedding_similarity = cosine_similarity([resume_embedding], [jd_embedding])[0][0]
    print("Resume Embedding Shape:", resume_embedding.shape)
    print("Job Description Embedding Shape:", jd_embedding.shape)

    print("Embedding Cosine Similarity:", embedding_similarity)

    # Combine TF-IDF and embedding similarity into score
    final_score = (cosine_sim + embedding_similarity) / 2
    print("Final Score:", final_score)
    return final_score

# Main function to process resumes and rank candidates
def main():
    resumes_folder = input("Enter the path to your resumes folder: ")
    if not os.path.isdir(resumes_folder):
        print(f"Error: The path '{resumes_folder}' is not a valid directory.")
        return

    # Get all resume files in the specified directory
    onlyfiles = [os.path.join(resumes_folder, f) for f in os.listdir(resumes_folder) if os.path.isfile(os.path.join(resumes_folder, f))]

    # Extract job description keywords
    job_description = extract_job_description_keywords()

    scores = []
    for file in onlyfiles:
        resume_text = pdfextract(file)
        filename = os.path.basename(file)

        # Get combined score with TF-IDF and embeddings
        score = score_resume_with_embeddings(resume_text, job_description)
        years_of_experience = extract_years_of_experience(resume_text)

        # Additional score boost based on experience
        experience_score = 0
        if years_of_experience >= 5:
            experience_score = 1
        elif years_of_experience >= 3:
            experience_score = 0.5
        score += experience_score

        scores.append((filename, score))

    # Sort and display top 5 candidates
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 Candidates:")
    for filename, sc in sorted_scores:
        print(f"{filename}: {sc:.2f}")

if __name__ == '__main__':
    main()
