# OrthofinixAI Backend

Production-ready FastAPI backend for the OrthofinixAI Android application.

## Requirements
- Python 3.9+
- Firebase Project with Firestore and Authentication enabled

## Local Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate a Service Account Key from your Firebase Console (`Project Settings > Service Accounts > Generate new private key`) and save it as `firebase-adminsdk.json` in this `backend` directory.
3. Set up the `.env` file (copy from `.env.example` if available).
4. Run the server locally:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
5. Check out the Swagger API Docs at [http://localhost:8000/docs](http://localhost:8000/docs)

## Android Integration (Retrofit)

### Base URL
If testing on an Android Emulator, use:
`http://10.0.2.2:8000/`

For a physical device on the same Wi-Fi, use your local IP address:
`http://192.168.x.x:8000/`

For production, use the Render deployment URL.

### Retrofit Interface Example

```kotlin
import okhttp3.MultipartBody
import retrofit2.Call
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.Path

interface OrthofinixApi {
    @Multipart
    @POST("analysis/predict")
    fun predictImage(
        @Part file: MultipartBody.Part
    ): Call<PredictionResponse>

    @POST("analysis/save-analysis")
    fun saveAnalysis(
        @Body record: AnalysisRecordCreate
    ): Call<AnalysisRecordResponse>

    @GET("analysis/history")
    fun getHistory(): Call<List<AnalysisRecordResponse>>

    @GET("analysis/{id}")
    fun getAnalysisById(@Path("id") id: String): Call<AnalysisRecordResponse>
}
```

### JSON Request / Response format

**Save Analysis Request Body (`AnalysisRecordCreate`)**:
```json
{
  "patient_name": "John Doe",
  "age": 25,
  "symptoms": ["Crowding", "Overbite"],
  "image_url": "/uploads/example.jpg",
  "prediction": "Class II Malocclusion",
  "confidence_score": 0.95,
  "recommendations": ["Consider Class II elastics or distalization."]
}
```

## Deployment

### Render Deployment Steps
1. Push this repository to GitHub.
2. Go to [Render](https://render.com/) and create a new **Web Service**.
3. Connect your GitHub repository.
4. Set the build command: `pip install -r requirements.txt`
5. Set the start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
6. Add an Environment Variable for the `FIREBASE_CREDENTIALS_PATH`. You can either:
   - Base64 encode your `firebase-adminsdk.json` and decode it at runtime.
   - Use Render's Secret Files feature to upload `firebase-adminsdk.json` and point the `FIREBASE_CREDENTIALS_PATH` environment variable to its location.
