# Face Matching Criminal Forensics Backend

A Django REST API application for face recognition and matching in criminal forensics. This system allows you to upload face images, extract their unique facial features, store them in a database, and search for similar faces to identify suspects or match evidence from crime scenes.

## Table of Contents

- [Project Overview](#project-overview)
- [Database Schema](#database-schema)
- [API Endpoints](#api-endpoints)
- [Project Architecture](#project-architecture)

## Project Overview

### Technology Stack

- **Backend Framework**: Django REST Framework (Python)
- **Database**: PostgreSQL
- **Image Storage**: AWS S3 (Amazon Simple Storage Service)
- **Face Recognition**: face_recognition library (using deep learning neural networks)
- **API Format**: RESTful JSON API

### Key Components

- **Database Layer**: PostgreSQL stores face metadata and 128-dimensional face encodings
- **Storage Layer**: AWS S3 stores actual face images
- **Processing Layer**: face_recognition library extracts and compares facial features
- **API Layer**: Django REST Framework provides HTTP endpoints for all operations

## Project Architecture

The system follows a three-tier architecture:

1. **Data Access Layer** - Models (faces/models.py)
2. **Business Logic Layer** - Views and Utilities (faces/views.py, faces/utils.py)
3. **Presentation Layer** - REST API Endpoints

## Database Schema

### FaceImage Table

The system uses a single main database table to store all face-related information:

| Column                | Type                         | Description                                                  |
| --------------------- | ---------------------------- | ------------------------------------------------------------ |
| **id**                | Integer (Primary Key)        | Unique identifier for each face record                       |
| **image_url**         | URL (500 chars)              | Link to the image stored in AWS S3                           |
| **original_filename** | String (255 chars)           | Original filename of the uploaded image                      |
| **encoding**          | Text (JSON)                  | 128-dimensional face encoding (unique facial feature vector) |
| **name**              | String (255 chars, Optional) | Name of the person in the image                              |
| **tags**              | JSON Array                   | Labels for categorization (e.g., ["suspect", "wanted"])      |
| **notes**             | Text (Optional)              | Additional information about the person                      |
| **uploaded_at**       | DateTime                     | Automatic timestamp when record is created                   |
| **updated_at**        | DateTime                     | Automatic timestamp when record is modified                  |

### What is Face Encoding?

A **face encoding** is a mathematical representation of a person's unique facial features. The face_recognition library:

1. Detects the face in the image
2. Extracts key facial landmarks (eyes, nose, mouth, face shape, etc.)
3. Converts these features into a 128-dimensional vector (128 numbers)
4. Stores this as JSON in the database

**Example:** [-0.123, 0.456, 0.789, ..., -0.234] (128 values representing unique facial characteristics)

Two faces with similar encodings belong to the same person. The system calculates a **similarity percentage** (0-100%) by comparing encodings.

## API Endpoints

All endpoints are RESTful and use JSON format for requests and responses. Base URL: `http://localhost:8000/api/faces/`

### 1. Upload Face Image

**Endpoint:** `POST /upload/`

**Purpose:** Upload a face image, extract its facial features (encoding), and store in the database.

**What happens:**

1. Receives an image file from the user
2. Detects if there's a face in the image
3. Extracts the 128-dimensional face encoding
4. Uploads the image to AWS S3
5. Stores the encoding and metadata in PostgreSQL database

**Input Parameters:**

- `image` (Required) - The image file containing a face
- `name` (Optional) - Person's name
- `tags` (Optional) - Categories like "suspect", "wanted", "missing"
- `notes` (Optional) - Additional context

**Response Success:** Returns the stored face record with ID and S3 URL

**Response Error:** Returns error if no face is detected in the image

---

### 2. Search for Similar Faces

**Endpoint:** `POST /search/`

**Purpose:** Upload a photo and find all matching faces in the database based on facial similarity.

**What happens:**

1. Receives a search image
2. Extracts the face encoding from the search image
3. Compares it with all stored face encodings in the database
4. Calculates similarity percentage (0-100%) for each match
5. Returns matches sorted by highest similarity first

**Input Parameters:**

- `image` (Required) - The image to search for
- `min_similarity` (Optional) - Minimum match percentage (0-100, default: 0)
- `max_results` (Optional) - Maximum number of results to return (1-50, default: 10)

**Response:** Returns a list of matching faces with:

- Similarity percentage
- Person's name (if available)
- Tags and notes
- Links to the images
- Upload date

**Use Case:** Identify a person from a crime scene photo or surveillance image

---

### 3. List All Faces

**Endpoint:** `GET /list/`

**Purpose:** Retrieve all face records stored in the database.

**What happens:**

1. Queries the database for all face records
2. Returns them in reverse chronological order (newest first)

**Input Parameters:** None

**Response:** Returns:

- Total count of faces in the database
- Complete details of all stored faces

**Use Case:** View the entire face database, audit stored records

---

### 4. Delete a Face

**Endpoint:** `DELETE /delete/<face_id>/`

**Purpose:** Remove a face record from the database.

**What happens:**

1. Finds the face record by ID
2. Deletes it from the database
3. Image remains in S3 storage (for audit trail)

**Input Parameters:**

- `face_id` - The ID of the face to delete

**Response Success:** Confirmation message

**Response Error:** "Face not found" if the ID doesn't exist

**Use Case:** Remove incorrect entries, privacy requests, case closures

---

### 5. Health Check

**Endpoint:** `GET /health/`

**Purpose:** Verify the API is running and database is connected.

**What happens:**

1. Checks API server status
2. Verifies database connection
3. Returns total count of faces

**Input Parameters:** None

**Response:** Returns:

- API status (healthy/unhealthy)
- Database connection status
- Total faces in database

**Use Case:** Monitoring, deployment verification, diagnostics

---

## AI Sketch Provider Configuration

The sketch AI endpoints (`/api/faces/sketch/` with `use_ai=true` and `/api/faces/sketch/compose-face/`) now support provider failover.

Environment variables:

- `AI_IMAGE_PROVIDER` - Preferred provider. Values: `huggingface` (default) or `pollinations`
- `HF_API_TOKEN` - Required only for Hugging Face provider
- `AI_IMAGE_HF_MODEL` - Hugging Face model id (default: `stabilityai/stable-diffusion-xl-base-1.0`)
- `AI_IMAGE_POLLINATIONS_MODEL` - Pollinations model name (default: `flux`)

Behavior:

- If preferred provider fails, the API automatically falls back to the other provider.
- If both providers fail, the endpoint returns a 500 with combined provider errors.
