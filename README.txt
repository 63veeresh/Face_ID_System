You are a senior Python/AI/ML engineer helping me complete a technical assessment for an AI/ML internship.

I need you to BUILD A COMPLETE, WORKING, LOCAL FACE RECOGNITION IDENTIFICATION SYSTEM from scratch.

IMPORTANT:
- The project must cost $0 / ₹0.
- Use only free/open-source libraries and pretrained models that can run locally.
- Do NOT use paid APIs, cloud APIs, Firebase, AWS, Azure, Google Cloud, or any paid service.
- Do NOT fabricate accuracy, evaluation metrics, benchmark results, or test results.
- Run actual tests and report only results that were genuinely obtained.
- The code must be understandable because I will have to explain the implementation in an interview.
- Prefer a simple, reliable, explainable implementation over unnecessary complexity.
- Target Windows + Python.
- The application should work on CPU if CUDA/GPU is unavailable.
- Avoid unnecessarily complicated infrastructure such as Docker unless it is genuinely useful.
- Do not copy a complete existing GitHub project.
- Write original, clean, well-structured code.

==================================================
ASSIGNMENT REQUIREMENTS
==================================================

The system must support:

1. Face detection
2. Face embeddings
3. Similarity-based face matching
4. Enrollment of individuals
5. Identification of new faces against the enrolled database
6. An "UNKNOWN" rejection mechanism
7. Basic evaluation
8. Documentation of:
   - model used
   - matching method
   - threshold
   - failure cases
   - possible improvements
9. A complete README
10. A project that can be uploaded directly to a public GitHub repository

The system should be able to:

Example:

Registered database:
- Person A
- Person B
- Person C

Input:
- Image containing Person B

Output:
- Detected face
- Predicted identity: Person B
- Similarity score
- Recognition status

For an unregistered person:

Output:
- Predicted identity: UNKNOWN
- Similarity score
- Recognition rejected because similarity is below the configured threshold

==================================================
RECOMMENDED TECHNICAL APPROACH
==================================================

Use a pretrained face-recognition pipeline rather than training a face recognition network from scratch.

Preferred initial stack:

Python
PyTorch
facenet-pytorch
MTCNN for face detection
InceptionResnetV1 pretrained on VGGFace2 for face embeddings
NumPy
scikit-learn where useful
OpenCV/Pillow for image processing
Streamlit for a simple local web interface
pytest for basic tests

Use cosine similarity for matching.

IMPORTANT:
Before implementing, verify that the selected libraries and pretrained model actually work together in the current Python environment.

If facenet-pytorch creates compatibility problems with the current environment, choose another FREE, OPEN-SOURCE, LOCALLY RUNNABLE face recognition solution and explain the reason in README.

Do not switch technologies randomly.

==================================================
SYSTEM ARCHITECTURE
==================================================

Design the system approximately as:

Input Image
    |
    v
Face Detection
    |
    v
Face Alignment / Preprocessing if supported
    |
    v
Face Embedding Model
    |
    v
Embedding Vector
    |
    v
Compare Against Enrolled Embeddings
    |
    v
Cosine Similarity
    |
    +--------------------------+
    |                          |
 similarity >= threshold       similarity < threshold
    |                          |
    v                          v
Recognized Person              UNKNOWN