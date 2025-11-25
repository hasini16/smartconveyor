# # # import uvicorn
# # # from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
# # # from fastapi.middleware.cors import CORSMiddleware
# # # from pydantic import BaseModel
# # # from sqlalchemy import create_engine, text
# # # from sqlalchemy.orm import sessionmaker, Session
# # # import pandas as pd
# # # import numpy as np
# # # import tensorflow as tf
# # # from tensorflow.keras.models import Model
# # # from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
# # # from sklearn.model_selection import train_test_split
# # # import pytesseract
# # # from PIL import Image
# # # import io
# # # import random
# # # import string
# # # import os
# # # import dotenv
# # # # ==========================================
# # # # 1. CONFIGURATION & DATABASE
# # # # ==========================================

# # # # Your Supabase Connection URL
# # # DATABASE_URL = os.getenv("DATABASE_URL")   

# # # # CSV File Path
# # # CSV_FILE = "parcels_10000.csv"
# # # MODEL_PATH = "routing_model.h5"
# # # MAX_CITY_ID = 10000  # Adjust if city IDs exceed this

# # # # Database Setup
# # # engine = create_engine(DATABASE_URL)
# # # SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# # # # FastAPI App Setup
# # # app = FastAPI(title="Smart Conveyor Belt System")

# # # # CORS - Allow connections from React Frontend
# # # app.add_middleware(
# # #     CORSMiddleware,
# # #     allow_origins=["*"],
# # #     allow_credentials=True,
# # #     allow_methods=["*"],
# # #     allow_headers=["*"],
# # # )

# # # # ==========================================
# # # # 2. DATA MODELS (Pydantic)
# # # # ==========================================

# # # class LoginRequest(BaseModel):
# # #     username: str
# # #     password: str

# # # class NewUserRequest(BaseModel):
# # #     email: str

# # # class UpdateProfileRequest(BaseModel):
# # #     user_id: int
# # #     new_username: str
# # #     new_password: str

# # # class CityRequest(BaseModel):
# # #     city_name: str

# # # class DataPointRequest(BaseModel):
# # #     source_city_id: int
# # #     source_city_name: str
# # #     destination_city_id: int
# # #     destination_city_name: str
# # #     parcel_type: int
# # #     route_direction: int

# # # class PredictionRequest(BaseModel):
# # #     source_city_id: int
# # #     destination_city_id: int
# # #     parcel_type: int

# # # # ==========================================
# # # # 3. HELPER FUNCTIONS & DEPENDENCIES
# # # # ==========================================

# # # def get_db():
# # #     """Database session dependency."""
# # #     db = SessionLocal()
# # #     try:
# # #         yield db
# # #     finally:
# # #         db.close()

# # # def load_ai_model():
# # #     """Loads the trained Keras model if it exists."""
# # #     if os.path.exists(MODEL_PATH):
# # #         try:
# # #             return tf.keras.models.load_model(MODEL_PATH)
# # #         except Exception as e:
# # #             print(f"Error loading model: {e}")
# # #             return None
# # #     return None

# # # # Global Model Variable
# # # model = load_ai_model()

# # # def train_model_pipeline():
# # #     """
# # #     Reads data from CSV and Database, combines them,
# # #     trains a new Neural Network, and saves it.
# # #     """
# # #     print("Starting Model Retraining...")
    
# # #     # 1. Load CSV Data
# # #     if os.path.exists(CSV_FILE):
# # #         df_csv = pd.read_csv(CSV_FILE)
# # #     else:
# # #         df_csv = pd.DataFrame(columns=['source_city_ID', 'destination_city_ID', 'parcel_type', 'route'])

# # #     # 2. Load DB Data
# # #     db = SessionLocal()
# # #     try:
# # #         query = text("SELECT source_city_id, destination_city_id, parcel_type, route_direction FROM datapoints")
# # #         results = db.execute(query).fetchall()
        
# # #         if results:
# # #             df_db = pd.DataFrame(results, columns=['source_city_ID', 'destination_city_ID', 'parcel_type', 'route'])
# # #             full_df = pd.concat([df_csv, df_db], ignore_index=True)
# # #         else:
# # #             full_df = df_csv
# # #     except Exception as e:
# # #         print(f"Database read error during training: {e}")
# # #         full_df = df_csv
# # #     finally:
# # #         db.close()

# # #     if len(full_df) < 100:
# # #         print("Not enough data to train.")
# # #         return 0

# # #     # 3. Preprocess
# # #     X_src = full_df['source_city_ID'].values
# # #     X_dst = full_df['destination_city_ID'].values
# # #     X_type = full_df['parcel_type'].values
# # #     y = full_df['route'].values

# # #     # 4. Build Neural Network
# # #     src_input = Input(shape=(1,), name='src_in')
# # #     dst_input = Input(shape=(1,), name='dst_in')
# # #     type_input = Input(shape=(1,), name='type_in')

# # #     embedding = Embedding(input_dim=MAX_CITY_ID + 5000, output_dim=32)
# # #     src_emb = Flatten()(embedding(src_input))
# # #     dst_emb = Flatten()(embedding(dst_input))

# # #     merged = Concatenate()([src_emb, dst_emb, type_input])
# # #     x = Dense(64, activation='relu')(merged)
# # #     x = Dropout(0.2)(x)
# # #     x = Dense(32, activation='relu')(x)
# # #     output = Dense(3, activation='softmax')(x)

# # #     new_model = Model(inputs=[src_input, dst_input, type_input], outputs=output)
# # #     new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # #     # 5. Train
# # #     new_model.fit([X_src, X_dst, X_type], y, epochs=5, batch_size=32, verbose=0)
    
# # #     # 6. Save
# # #     new_model.save(MODEL_PATH)
# # #     print("Model saved.")
    
# # #     return len(full_df)

# # # # ==========================================
# # # # 4. STARTUP EVENT (Seeding)
# # # # ==========================================

# # # @app.on_event("startup")
# # # def startup_event():
# # #     """Runs when server starts. Seeds cities table if empty."""
# # #     db = SessionLocal()
# # #     try:
# # #         # Check if cities exist
# # #         try:
# # #             count = db.execute(text("SELECT COUNT(*) FROM cities")).scalar()
# # #         except:
# # #             # Table might not exist yet, skip seeding (assume user ran SQL script)
# # #             print("Cities table not found or error checking count.")
# # #             return

# # #         if count == 0 and os.path.exists(CSV_FILE):
# # #             print("Seeding cities from CSV...")
# # #             df = pd.read_csv(CSV_FILE)
            
# # #             # Extract unique source cities
# # #             src = df[['source_city', 'source_city_ID']].rename(columns={'source_city': 'name', 'source_city_ID': 'id'})
# # #             # Extract unique dest cities
# # #             dst = df[['destination_city', 'destination_city_ID']].rename(columns={'destination_city': 'name', 'destination_city_ID': 'id'})
            
# # #             # Combine and drop duplicates
# # #             cities = pd.concat([src, dst]).drop_duplicates(subset=['id'])
            
# # #             # Insert into DB
# # #             for _, row in cities.iterrows():
# # #                 clean_name = str(row['name']).replace("'", "''") # Escape single quotes
# # #                 sql = text(f"INSERT INTO cities (unique_id, cityname) VALUES ({row['id']}, '{clean_name}') ON CONFLICT DO NOTHING")
# # #                 db.execute(sql)
            
# # #             db.commit()
# # #             print(f"Seeded {len(cities)} cities.")
# # #     except Exception as e:
# # #         print(f"Startup error: {e}")
# # #     finally:
# # #         db.close()

# # # # ==========================================
# # # # 5. API ENDPOINTS
# # # # ==========================================

# # # # --- Auth ---

# # # @app.post("/login/{role}")
# # # def login(role: str, creds: LoginRequest, db: Session = Depends(get_db)):
# # #     table = "admin-logins" if role == "admin" else "employee-logins"
# # #     # Note: Using parameterized queries to prevent SQL Injection
# # #     query = text(f"SELECT id, username FROM \"{table}\" WHERE username=:u AND password=:p")
# # #     result = db.execute(query, {"u": creds.username, "p": creds.password}).fetchone()
    
# # #     if result:
# # #         return {"status": "success", "user_id": result[0], "username": result[1], "role": role}
# # #     raise HTTPException(status_code=401, detail="Invalid credentials")

# # # # --- User Management (Admin) ---

# # # @app.get("/users/{role}")
# # # def get_users(role: str, db: Session = Depends(get_db)):
# # #     table = "admin-logins" if role == "admin" else "employee-logins"
# # #     result = db.execute(text(f"SELECT id, username, email FROM \"{table}\"")).fetchall()
# # #     return [{"id": r[0], "username": r[1], "email": r[2]} for r in result]

# # # @app.post("/add-user/{role}")
# # # def add_user(role: str, req: NewUserRequest, db: Session = Depends(get_db)):
# # #     table = "admin-logins" if role == "admin" else "employee-logins"
    
# # #     # Generate random credentials
# # #     username = 'user_' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
# # #     password = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    
# # #     try:
# # #         query = text(f"INSERT INTO \"{table}\" (username, password, email) VALUES (:u, :p, :e)")
# # #         db.execute(query, {"u": username, "p": password, "e": req.email})
# # #         db.commit()
# # #         return {"status": "created", "username": username, "password": password}
# # #     except Exception as e:
# # #         raise HTTPException(status_code=400, detail=f"Error creating user: {str(e)}")

# # # @app.delete("/remove-user/{role}/{user_id}")
# # # def remove_user(role: str, user_id: int, db: Session = Depends(get_db)):
# # #     table = "admin-logins" if role == "admin" else "employee-logins"
# # #     db.execute(text(f"DELETE FROM \"{table}\" WHERE id=:id"), {"id": user_id})
# # #     db.commit()
# # #     return {"status": "deleted"}

# # # @app.put("/edit-profile/{role}")
# # # def edit_profile(role: str, req: UpdateProfileRequest, db: Session = Depends(get_db)):
# # #     table = "admin-logins" if role == "admin" else "employee-logins"
    
# # #     # Check uniqueness of new username
# # #     check_query = text(f"SELECT id FROM \"{table}\" WHERE username=:u AND id != :id")
# # #     existing = db.execute(check_query, {"u": req.new_username, "id": req.user_id}).fetchone()
# # #     if existing:
# # #         raise HTTPException(status_code=400, detail="Username already taken")
        
# # #     update_query = text(f"UPDATE \"{table}\" SET username=:u, password=:p WHERE id=:id")
# # #     db.execute(update_query, {"u": req.new_username, "p": req.new_password, "id": req.user_id})
# # #     db.commit()
# # #     return {"status": "updated"}

# # # # --- City Management ---

# # # @app.get("/cities")
# # # def get_cities(db: Session = Depends(get_db)):
# # #     res = db.execute(text("SELECT unique_id, cityname FROM cities ORDER BY cityname")).fetchall()
# # #     return [{"id": r[0], "name": r[1]} for r in res]

# # # @app.post("/add-city")
# # # def add_city(req: CityRequest, db: Session = Depends(get_db)):
# # #     # Calculate next ID
# # #     max_id = db.execute(text("SELECT MAX(unique_id) FROM cities")).scalar()
# # #     new_id = (max_id if max_id else 5000) + 1
    
# # #     try:
# # #         db.execute(text("INSERT INTO cities (unique_id, cityname) VALUES (:id, :name)"), 
# # #                    {"id": new_id, "name": req.city_name})
# # #         db.commit()
# # #         return {"status": "success", "new_id": new_id, "name": req.city_name}
# # #     except:
# # #         raise HTTPException(status_code=400, detail="City name likely exists")

# # # # --- Data Points & Training ---

# # # @app.get("/datapoints")
# # # def get_datapoints(db: Session = Depends(get_db)):
# # #     # Get DB data
# # #     db_res = db.execute(text("SELECT source_city_id, source_city_name, destination_city_id, destination_city_name, parcel_type, route_direction FROM datapoints ORDER BY id DESC LIMIT 500")).fetchall()
    
# # #     db_data = []
# # #     for r in db_res:
# # #         db_data.append({
# # #             "source_city_ID": r[0], "source_city": r[1],
# # #             "destination_city_ID": r[2], "destination_city": r[3],
# # #             "parcel_type": r[4], "route": r[5], "origin": "DB"
# # #         })
        
# # #     # Optional: Mix with some CSV data for display
# # #     # (Skipping heavy CSV load here for performance, showing mostly DB data)
# # #     return db_data

# # # @app.post("/add-datapoint")
# # # def add_datapoint(data: DataPointRequest, db: Session = Depends(get_db)):
# # #     query = text("""
# # #         INSERT INTO datapoints 
# # #         (source_city_id, source_city_name, destination_city_id, destination_city_name, parcel_type, route_direction)
# # #         VALUES (:sid, :sn, :did, :dn, :pt, :rd)
# # #     """)
# # #     db.execute(query, {
# # #         "sid": data.source_city_id, "sn": data.source_city_name,
# # #         "did": data.destination_city_id, "dn": data.destination_city_name,
# # #         "pt": data.parcel_type, "rd": data.route_direction
# # #     })
# # #     db.commit()
# # #     return {"status": "success"}

# # # @app.post("/retrain")
# # # def retrain_endpoint():
# # #     try:
# # #         total_samples = train_model_pipeline()
# # #         # Reload model globally
# # #         global model
# # #         model = load_ai_model()
# # #         return {"status": "success", "total_samples_trained": total_samples}
# # #     except Exception as e:
# # #         raise HTTPException(status_code=500, detail=str(e))

# # # # --- Prediction & Image ---

# # # @app.post("/predict")
# # # def predict_route(req: PredictionRequest):
# # #     if model is None:
# # #         return {"route_code": -1, "direction": "Error: Model not trained/loaded"}
    
# # #     src = np.array([req.source_city_id])
# # #     dst = np.array([req.destination_city_id])
# # #     typ = np.array([req.parcel_type])
    
# # #     pred = model.predict([src, dst, typ])
# # #     route_idx = int(np.argmax(pred[0]))
    
# # #     mapping = {0: "Straight", 1: "Left", 2: "Right"}
# # #     return {"route_code": route_idx, "direction": mapping.get(route_idx, "Unknown")}

# # # @app.post("/extract-from-image")
# # # async def extract_data(file: UploadFile = File(...)):
# # #     try:
# # #         contents = await file.read()
# # #         image = Image.open(io.BytesIO(contents))
        
# # #         # Perform OCR
# # #         # Note: Requires Tesseract installed on system
# # #         text_out = pytesseract.image_to_string(image)
        
# # #         # Simple parsing heuristic
# # #         data = {"source_id": "", "dest_id": "", "type": "0"}
# # #         lines = text_out.split('\n')
        
# # #         for line in lines:
# # #             line_lower = line.lower()
# # #             # Extract numbers from line
# # #             nums = [int(s) for s in line.split() if s.isdigit()]
# # #             if not nums: continue
            
# # #             if "source" in line_lower:
# # #                 data["source_id"] = nums[0]
# # #             elif "dest" in line_lower:
# # #                 data["dest_id"] = nums[0]
# # #             elif "type" in line_lower or "parcel" in line_lower:
# # #                 data["type"] = nums[0]
                
# # #         return {"extracted_text": text_out, "parsed_data": data}
# # #     except Exception as e:
# # #         raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

# # # if __name__ == "__main__":
# # #     uvicorn.run(app, host="0.0.0.0", port=8000)

# # # server/main.py
# # import os
# # from contextlib import asynccontextmanager
# # from typing import List, Dict, Union, Optional
# # import random
# # import string
# # import hashlib
# # import dotenv
# # # Database & ORM
# # from sqlalchemy import create_engine, Column, Integer, String, text
# # from sqlalchemy.orm import sessionmaker, declarative_base
# # from sqlalchemy.exc import IntegrityError, OperationalError
# # from sqlalchemy.engine import Connection

# # # FastAPI & Pydantic
# # from fastapi import FastAPI, Depends, HTTPException, status
# # from fastapi.middleware.cors import CORSMiddleware
# # from pydantic import BaseModel, Field, validator

# # # Model/Data handling (using joblib for scikit-learn model)
# # import joblib
# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.neural_network import MLPClassifier


# # # --- 1. CONFIGURATION & MAPPINGS ---

# # # Your Supabase Connection String
# # # DATABASE_URL = "postgresql://postgres.clgjswrlcwzmdxhhegtk:adhithya365@aws-1-ap-south-1.pooler.supabase.com:6543/postgres"
# # DATABASE_URL = os.getenv("DATABASE_URL")
# # MODEL_FILENAME = 'routing_model.pkl'
# # ROUTE_MAP = {0: 'straight', 1: 'left', 2: 'right'}
# # ROUTE_MAP_REV = {'straight': 0, 'left': 1, 'right': 2}
# # PARCEL_MAP = {'normal': 0, 'fast': 1}
# # PARCEL_MAP_REV = {0: 'normal', 1: 'fast'}
# # FEATURE_COLS = ['source_city_id', 'destination_city_id', 'parcel_type']
# # TARGET_COL = 'route_direction'


# # # --- 2. DATABASE SETUP ---

# # Base = declarative_base()

# # # Define Database Models (Tables)
# # class AdminLogin(Base):
# #     __tablename__ = 'admin-logins'
# #     id = Column(Integer, primary_key=True, index=True)
# #     username = Column(String, unique=True, index=True, nullable=False)
# #     hashed_password = Column(String, nullable=False)

# # class EmployeeLogin(Base):
# #     __tablename__ = 'employee-logins'
# #     id = Column(Integer, primary_key=True, index=True)
# #     username = Column(String, unique=True, index=True, nullable=False)
# #     hashed_password = Column(String, nullable=False)
# #     email = Column(String, unique=True, index=True, nullable=True) # Making email nullable for simplicity if old data lacks it

# # class City(Base):
# #     __tablename__ = 'cities'
# #     id = Column(Integer, primary_key=True)
# #     city_name = Column(String, unique=True, index=True, nullable=False)
# #     unique_id = Column(Integer, unique=True, index=True, nullable=False)

# # class DataPoint(Base):
# #     __tablename__ = 'datapoints'
# #     id = Column(Integer, primary_key=True, index=True)
# #     source_city_id = Column(Integer, index=True, nullable=False)
# #     destination_city_id = Column(Integer, index=True, nullable=False)
# #     parcel_type = Column(Integer, nullable=False) # 0: normal, 1: fast
# #     route_direction = Column(Integer, nullable=False) # 0: straight, 1: left, 2: right

# # # Create the engine
# # engine = create_engine(DATABASE_URL)
# # # Session Local
# # SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# # # Dependency to get the DB session
# # def get_db():
# #     db = SessionLocal()
# #     try:
# #         yield db
# #     finally:
# #         db.close()

# # # --- 3. HELPER FUNCTIONS ---

# # def hash_password(password: str) -> str:
# #     """Hashes a password using SHA256."""
# #     # Production apps should use bcrypt/Argon2
# #     return hashlib.sha256(password.encode()).hexdigest()

# # def verify_password(plain_password: str, hashed_password: str) -> bool:
# #     """Verifies a plain password against a hash."""
# #     return hash_password(plain_password) == hashed_password

# # def generate_credentials(email: str) -> tuple:
# #     """Generates a random username and one-time password."""
# #     # Basic unique username generation (can lead to clashes, but fine for prototype)
# #     username = email.split('@')[0] + "".join(random.choices(string.digits, k=4))
# #     password = "".join(random.choices(string.ascii_letters + string.digits, k=10))
# #     return username, password

# # def get_next_city_id(db) -> int:
# #     """Gets the next unique_id for a new city."""
# #     max_id = db.query(text("MAX(unique_id)")).select_from(City).scalar()
# #     return (max_id or 0) + 1
    
# # def prepare_retrain_data(datapoints: List[DataPoint]) -> tuple:
# #     """Prepares data from DB models for model retraining."""
# #     data_list = [{
# #         'source_city_id': dp.source_city_id,
# #         'destination_city_id': dp.destination_city_id,
# #         'parcel_type': dp.parcel_type,
# #         'route_direction': dp.route_direction
# #     } for dp in datapoints]
    
# #     df = pd.DataFrame(data_list)
    
# #     if len(df) == 0:
# #         raise ValueError("No data points found for retraining.")
        
# #     X = df[FEATURE_COLS].values.astype(np.float32)
# #     y = df[TARGET_COL].values.astype(np.int32)
    
# #     return X, y

# # # --- 4. DATA MODELS (Pydantic Schemas) ---

# # class CityBase(BaseModel):
# #     city_name: str = Field(..., min_length=1, description="Name of the city.")

# # class CityRead(CityBase):
# #     id: int
# #     unique_id: int

# #     class Config:
# #         from_attributes = True

# # class LoginRequest(BaseModel):
# #     username: str
# #     password: str

# # class ProfileUpdate(BaseModel):
# #     new_username: str
# #     new_password: str

# #     @validator('new_username')
# #     def validate_username(cls, v):
# #         if not v or len(v) < 3:
# #             raise ValueError('Username must be at least 3 characters long')
# #         return v

# # class NewUserRequest(BaseModel):
# #     email: str

# #     @validator('email')
# #     def validate_email(cls, v):
# #         if "@" not in v or "." not in v:
# #             raise ValueError('Invalid email format')
# #         return v

# # class DataPointBase(BaseModel):
# #     source_city_id: int
# #     destination_city_id: int
# #     parcel_type_text: str = Field(..., description="Parcel type: 'normal' or 'fast'")
# #     route_direction_text: Optional[str] = Field(None, description="Route: 'straight', 'left', 'right'")
    
# #     @validator('parcel_type_text')
# #     def validate_parcel_type(cls, v):
# #         if v.lower() not in PARCEL_MAP:
# #             raise ValueError(f"Parcel type must be one of: {list(PARCEL_MAP.keys())}")
# #         return v.lower()
    
# #     @validator('route_direction_text')
# #     def validate_route_direction(cls, v):
# #         if v is not None and v.lower() not in ROUTE_MAP_REV:
# #             raise ValueError(f"Route direction must be one of: {list(ROUTE_MAP_REV.keys())}")
# #         return v.lower() if v else None

# # class DataPointRead(DataPointBase):
# #     id: int
# #     source_city_name: Optional[str] = None
# #     destination_city_name: Optional[str] = None
    
# #     class Config:
# #         from_attributes = True

# # class PredictionRequest(BaseModel):
# #     source_city_id: int
# #     destination_city_id: int
# #     parcel_type_text: str = Field(..., description="Parcel type: 'normal' or 'fast'")
    
# #     @validator('parcel_type_text')
# #     def validate_parcel_type(cls, v):
# #         if v.lower() not in PARCEL_MAP:
# #             raise ValueError(f"Parcel type must be one of: {list(PARCEL_MAP.keys())}")
# #         return v.lower()

# # # --- 5. APPLICATION LIFESPAN & MODEL LOADING ---

# # smart_conveyor_model = None

# # @asynccontextmanager
# # async def lifespan(app: FastAPI):
# #     global smart_conveyor_model
    
# #     try:
# #         # NOTE: Rely on manual Supabase table creation.
# #         # Base.metadata.create_all(bind=engine) 
# #         print("Database structure check complete.")
# #     except OperationalError as e:
# #         print(f"Could not connect to database or check tables: {e}")

# #     # 2. Load the Model using joblib
# #     try:
# #         current_dir = os.path.dirname(os.path.abspath(__file__))
# #         model_path = os.path.join(current_dir, MODEL_FILENAME)
# #         if os.path.exists(model_path):
# #             smart_conveyor_model = joblib.load(model_path)
# #             print("Model loaded successfully using joblib.")
# #         else:
# #             print(f"WARNING: Model file not found at {model_path}. Run train_model.py first.")
# #     except Exception as e:
# #         print(f"ERROR: Could not load the model: {e}")
        
# #     yield
    
# # # --- 6. FASTAPI APP INITIALIZATION ---

# # app = FastAPI(lifespan=lifespan, 
# #               title="Smart Conveyor API",
# #               description="Backend for the smart conveyor belt system using FastAPI.")

# # origins = [
# #     "http://localhost:5173",  # Default Vite/React development server
# #     "http://127.0.0.1:5173",
# #     # Add your production frontend URL here
# # ]

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=origins,
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # --- 7. DEPENDENCIES (Authentication) ---

# # async def authenticate_user(login_data: LoginRequest, db, model_cls):
# #     """Authenticates a user against a given login table model."""
# #     user = db.query(model_cls).filter(model_cls.username == login_data.username).first()
# #     if not user or not verify_password(login_data.password, user.hashed_password):
# #         raise HTTPException(
# #             status_code=status.HTTP_401_UNAUTHORIZED,
# #             detail="Invalid credentials",
# #             headers={"WWW-Authenticate": "Bearer"},
# #         )
# #     # Return a basic dict instead of ORM object to avoid accidental exposure
# #     return {"id": user.id, "username": user.username}

# # async def get_current_admin(login_data: LoginRequest = Depends(), db: SessionLocal = Depends(get_db)):
# #     """Dependency for Admin authentication."""
# #     return await authenticate_user(login_data, db, AdminLogin)

# # async def get_current_employee(login_data: LoginRequest = Depends(), db: SessionLocal = Depends(get_db)):
# #     """Dependency for Employee authentication."""
# #     return await authenticate_user(login_data, db, EmployeeLogin)


# # # --- 8. ENDPOINTS ---

# # @app.get("/")
# # def read_root():
# #     return {"message": "Smart Conveyor API is running!"}

# # #
# # # --- AUTHENTICATION ENDPOINTS ---
# # # (Note: Authentication is handled by the dependency and returns user info)

# # @app.post("/auth/admin/login")
# # async def admin_login(authenticated_user: Dict = Depends(get_current_admin)):
# #     return {"message": "Admin login successful", "username": authenticated_user['username'], "user_id": authenticated_user['id']}

# # @app.post("/auth/employee/login")
# # async def employee_login(authenticated_user: Dict = Depends(get_current_employee)):
# #     return {"message": "Employee login successful", "username": authenticated_user['username'], "user_id": authenticated_user['id']}

# # #
# # # --- ADMIN ENDPOINTS (Assumes authenticated via token/session in a real app, placeholder used here) ---
# # #

# # # Placeholder dependency for required admin access (using a known/default user ID)
# # # In a real app, this would use a security token in the header.
# # async def check_admin_access(db: SessionLocal = Depends(get_db)):
# #     # For prototype: always allow access if credentials are valid (handled by frontend logic for now)
# #     return True 

# # @app.post("/admin/new_admin", status_code=status.HTTP_201_CREATED)
# # async def add_new_admin(new_user: NewUserRequest, db: SessionLocal = Depends(get_db)):
# #     username, password = generate_credentials(new_user.email)
# #     hashed_password = hash_password(password)
    
# #     new_admin = AdminLogin(username=username, hashed_password=hashed_password)
    
# #     try:
# #         db.add(new_admin)
# #         db.commit()
# #         return {
# #             "message": "New admin added successfully. **Save these credentials immediately.**",
# #             "username": username,
# #             "password": password, 
# #         }
# #     except IntegrityError:
# #         db.rollback()
# #         raise HTTPException(status_code=400, detail="Username already exists (try again)")

# # @app.post("/admin/new_employee", status_code=status.HTTP_201_CREATED)
# # async def add_new_employee(new_user: NewUserRequest, db: SessionLocal = Depends(get_db)):
# #     username, password = generate_credentials(new_user.email)
# #     hashed_password = hash_password(password)
    
# #     new_employee = EmployeeLogin(username=username, hashed_password=hashed_password, email=new_user.email)
    
# #     try:
# #         db.add(new_employee)
# #         db.commit()
# #         return {
# #             "message": "New employee added successfully. **Save these credentials immediately.**",
# #             "username": username,
# #             "password": password, 
# #             "email": new_user.email
# #         }
# #     except IntegrityError:
# #         db.rollback()
# #         raise HTTPException(status_code=400, detail="Username or email already exists")

# # @app.post("/admin/edit_profile/{admin_id}")
# # async def edit_admin_profile(admin_id: int, profile_data: ProfileUpdate, db: SessionLocal = Depends(get_db)):
# #     # NOTE: In a real app, admin_id should be extracted from a secure token, not path param.
# #     admin = db.query(AdminLogin).filter(AdminLogin.id == admin_id).first()
    
# #     if not admin:
# #         raise HTTPException(status_code=404, detail="Admin not found")
        
# #     if profile_data.new_username != admin.username:
# #         if db.query(AdminLogin).filter(AdminLogin.username == profile_data.new_username).first():
# #             raise HTTPException(status_code=400, detail="New username already taken")
# #         admin.username = profile_data.new_username
        
# #     admin.hashed_password = hash_password(profile_data.new_password)
    
# #     db.commit()
# #     return {"message": "Admin profile updated successfully", "new_username": admin.username}

# # #
# # # --- EMPLOYEE ENDPOINTS (Assumes authenticated via token/session in a real app) ---
# # #

# # @app.get("/employee/cities", response_model=List[CityRead])
# # async def get_all_cities(db: SessionLocal = Depends(get_db)):
# #     return db.query(City).order_by(City.unique_id).all()

# # @app.post("/employee/add_city", response_model=CityRead, status_code=status.HTTP_201_CREATED)
# # async def add_new_city(city_data: CityBase, db: SessionLocal = Depends(get_db)):
# #     next_id = get_next_city_id(db)
# #     new_city = City(city_name=city_data.city_name, unique_id=next_id)
    
# #     try:
# #         db.add(new_city)
# #         db.commit()
# #         db.refresh(new_city)
# #         return new_city
# #     except IntegrityError:
# #         db.rollback()
# #         raise HTTPException(status_code=400, detail="City name already exists")

# # @app.post("/employee/add_datapoint", response_model=DataPointRead, status_code=status.HTTP_201_CREATED)
# # async def add_new_datapoint(data_point: DataPointBase, db: SessionLocal = Depends(get_db)):
# #     try:
# #         parcel_type_int = PARCEL_MAP[data_point.parcel_type_text]
# #         route_direction_int = ROUTE_MAP_REV[data_point.route_direction_text]
# #     except KeyError as e:
# #         raise HTTPException(status_code=400, detail=f"Invalid mapping for {e}")
        
# #     new_dp = DataPoint(
# #         source_city_id=data_point.source_city_id,
# #         destination_city_id=data_point.destination_city_id,
# #         parcel_type=parcel_type_int,
# #         route_direction=route_direction_int
# #     )
    
# #     try:
# #         db.add(new_dp)
# #         db.commit()
# #         db.refresh(new_dp)
        
# #         # Fetch city names for a friendly response
# #         cities = db.query(City).all()
# #         city_map = {c.unique_id: c.city_name for c in cities}
        
# #         return DataPointRead(
# #             id=new_dp.id,
# #             source_city_id=new_dp.source_city_id,
# #             destination_city_id=new_dp.destination_city_id,
# #             parcel_type_text=PARCEL_MAP_REV.get(new_dp.parcel_type, 'unknown'),
# #             route_direction_text=ROUTE_MAP.get(new_dp.route_direction, 'unknown'),
# #             source_city_name=city_map.get(new_dp.source_city_id),
# #             destination_city_name=city_map.get(new_dp.destination_city_id)
# #         )
# #     except Exception as e:
# #         db.rollback()
# #         raise HTTPException(status_code=500, detail=f"Database error: {e}")

# # @app.get("/employee/datapoints", response_model=List[DataPointRead])
# # async def get_datapoints(db: SessionLocal = Depends(get_db)):
# #     datapoints = db.query(DataPoint).order_by(DataPoint.id).all()
# #     cities = db.query(City).all()
# #     city_map = {c.unique_id: c.city_name for c in cities}
    
# #     result = []
# #     for dp in datapoints:
# #         result.append(DataPointRead(
# #             id=dp.id,
# #             source_city_id=dp.source_city_id,
# #             destination_city_id=dp.destination_city_id,
# #             parcel_type_text=PARCEL_MAP_REV.get(dp.parcel_type, 'unknown'),
# #             route_direction_text=ROUTE_MAP.get(dp.route_direction, 'unknown'),
# #             source_city_name=city_map.get(dp.source_city_id),
# #             destination_city_name=city_map.get(dp.destination_city_id)
# #         ))
        
# #     return result

# # @app.put("/employee/update_datapoint/{datapoint_id}")
# # async def update_datapoint(datapoint_id: int, data_point: DataPointBase, db: SessionLocal = Depends(get_db)):
# #     db_dp = db.query(DataPoint).filter(DataPoint.id == datapoint_id).first()
    
# #     if not db_dp:
# #         raise HTTPException(status_code=404, detail="DataPoint not found")
        
# #     try:
# #         db_dp.route_direction = ROUTE_MAP_REV[data_point.route_direction_text]
# #         db.commit()
# #     except KeyError:
# #         raise HTTPException(status_code=400, detail="Invalid route direction")
# #     except Exception as e:
# #         db.rollback()
# #         raise HTTPException(status_code=500, detail=f"Database error: {e}")
        
# #     return {"message": "DataPoint updated successfully"}

# # @app.post("/employee/retrain_model")
# # async def retrain_model(db: SessionLocal = Depends(get_db)):
# #     global smart_conveyor_model
    
# #     datapoints = db.query(DataPoint).all()
# #     if not datapoints:
# #         raise HTTPException(status_code=400, detail="No data points found in the database for retraining.")
        
# #     try:
# #         X, y = prepare_retrain_data(datapoints)

# #         if len(X) < 10:
# #             X_train, y_train = X, y
# #             X_test, y_test = X, y
# #         else:
# #             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #         model = MLPClassifier(hidden_layer_sizes=(50,), 
# #                               max_iter=500, 
# #                               activation='relu', 
# #                               solver='adam', 
# #                               random_state=42)

# #         model.fit(X_train, y_train)
# #         accuracy = model.score(X_test, y_test)

# #         current_dir = os.path.dirname(os.path.abspath(__file__))
# #         model_path = os.path.join(current_dir, MODEL_FILENAME)
# #         joblib.dump(model, model_path)
# #         smart_conveyor_model = model 

# #         return {"message": "Model retrained successfully!", "test_accuracy": f"{accuracy:.4f}", "new_data_size": len(X)}

# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"Model retraining failed: {e}")

# # @app.post("/employee/edit_profile/{employee_id}")
# # async def edit_employee_profile(employee_id: int, profile_data: ProfileUpdate, db: SessionLocal = Depends(get_db)):
# #     # NOTE: In a real app, employee_id should be extracted from a secure token, not path param.
# #     employee = db.query(EmployeeLogin).filter(EmployeeLogin.id == employee_id).first()
    
# #     if not employee:
# #         raise HTTPException(status_code=404, detail="Employee not found")
        
# #     if profile_data.new_username != employee.username:
# #         if db.query(EmployeeLogin).filter(EmployeeLogin.username == profile_data.new_username).first():
# #             raise HTTPException(status_code=400, detail="New username already taken")
# #         employee.username = profile_data.new_username
        
# #     employee.hashed_password = hash_password(profile_data.new_password)
    
# #     db.commit()
# #     return {"message": "Employee profile updated successfully", "new_username": employee.username}


# # #
# # # --- USER ENDPOINTS (Prediction) ---
# # #

# # @app.post("/user/predict_route")
# # async def predict_route(request: PredictionRequest, db: SessionLocal = Depends(get_db)):
# #     global smart_conveyor_model

# #     if smart_conveyor_model is None:
# #         raise HTTPException(status_code=503, detail="Model is not loaded or trained yet.")
        
# #     try:
# #         parcel_type_int = PARCEL_MAP[request.parcel_type_text]
# #         features = np.array([[
# #             request.source_city_id, 
# #             request.destination_city_id, 
# #             parcel_type_int
# #         ]])
        
# #         prediction_int = smart_conveyor_model.predict(features)[0]
# #         predicted_route = ROUTE_MAP.get(prediction_int, 'unknown')
        
# #         # Optional: Fetch city names for a richer response
# #         cities = db.query(City).filter(City.unique_id.in_([request.source_city_id, request.destination_city_id])).all()
# #         city_map = {c.unique_id: c.city_name for c in cities}

# #         return {
# #             "source_id": request.source_city_id,
# #             "destination_id": request.destination_city_id,
# #             "source_city_name": city_map.get(request.source_city_id, 'ID Not Found'),
# #             "destination_city_name": city_map.get(request.destination_city_id, 'ID Not Found'),
# #             "parcel_type": request.parcel_type_text,
# #             "predicted_route": predicted_route
# #         }

# #     except KeyError:
# #         raise HTTPException(status_code=400, detail="Invalid parcel type in request.")
# #     except Exception as e:
# #         print(f"Prediction error: {e}")
# #         raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

# # server/main.py
# import os
# from contextlib import asynccontextmanager
# from typing import List, Dict, Union, Optional
# import random
# import string
# import hashlib
# import dotenv  # <-- Import dotenv library
# # Database & ORM
# from sqlalchemy import create_engine, Column, Integer, String, text
# from sqlalchemy.orm import sessionmaker, declarative_base
# from sqlalchemy.exc import IntegrityError, OperationalError
# # FastAPI & Pydantic
# from fastapi import FastAPI, Depends, HTTPException, status
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field, validator
# # Model/Data handling (using joblib for scikit-learn model)
# import joblib
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPClassifier

# # --- Load Environment Variables ---
# dotenv.load_dotenv()

# # --- 1. CONFIGURATION & MAPPINGS ---

# # Your Supabase Connection String loaded from .env
# DATABASE_URL = os.getenv("DATABASE_URL")

# # --- Essential check for running the app ---
# if not DATABASE_URL:
#     raise ValueError(
#         "DATABASE_URL not found. Please ensure it is correctly set in your .env file."
#     )

# MODEL_FILENAME = 'routing_model.pkl' # Using joblib extension
# ROUTE_MAP = {0: 'straight', 1: 'left', 2: 'right'}
# ROUTE_MAP_REV = {'straight': 0, 'left': 1, 'right': 2}
# PARCEL_MAP = {'normal': 0, 'fast': 1}
# PARCEL_MAP_REV = {0: 'normal', 1: 'fast'}
# FEATURE_COLS = ['source_city_id', 'destination_city_id', 'parcel_type']
# TARGET_COL = 'route_direction'


# # --- 2. DATABASE SETUP ---

# Base = declarative_base()

# # Define Database Models (Tables) - ORM definitions allow FastAPI to map data easily
# class AdminLogin(Base):
#     __tablename__ = 'admin-logins'
#     id = Column(Integer, primary_key=True, index=True)
#     username = Column(String, unique=True, index=True, nullable=False)
#     hashed_password = Column(String, nullable=False)

# class EmployeeLogin(Base):
#     __tablename__ = 'employee-logins'
#     id = Column(Integer, primary_key=True, index=True)
#     username = Column(String, unique=True, index=True, nullable=False)
#     hashed_password = Column(String, nullable=False)
#     email = Column(String, unique=True, index=True, nullable=True)

# class City(Base):
#     __tablename__ = 'cities'
#     id = Column(Integer, primary_key=True)
#     city_name = Column(String, unique=True, index=True, nullable=False)
#     unique_id = Column(Integer, unique=True, index=True, nullable=False)

# class DataPoint(Base):
#     __tablename__ = 'datapoints'
#     id = Column(Integer, primary_key=True, index=True)
#     source_city_id = Column(Integer, index=True, nullable=False)
#     destination_city_id = Column(Integer, index=True, nullable=False)
#     parcel_type = Column(Integer, nullable=False)
#     route_direction = Column(Integer, nullable=False)

# # Create the engine
# engine = create_engine(DATABASE_URL)
# # Session Local
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# # Dependency to get the DB session
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# # --- 3. HELPER FUNCTIONS ---

# def hash_password(password: str) -> str:
#     """Hashes a password using SHA256."""
#     return hashlib.sha256(password.encode()).hexdigest()

# def verify_password(plain_password: str, hashed_password: str) -> bool:
#     """Verifies a plain password against a hash."""
#     return hash_password(plain_password) == hashed_password

# def generate_credentials(email: str) -> tuple:
#     """Generates a random username and one-time password."""
#     username = email.split('@')[0] + "".join(random.choices(string.digits, k=4))
#     password = "".join(random.choices(string.ascii_letters + string.digits, k=10))
#     return username, password

# def get_next_city_id(db) -> int:
#     """Gets the next unique_id for a new city."""
#     max_id = db.query(text("MAX(unique_id)")).select_from(City).scalar()
#     return (max_id or 0) + 1
    
# def prepare_retrain_data(datapoints: List[DataPoint]) -> tuple:
#     """Prepares data from DB models for model retraining."""
#     data_list = [{
#         'source_city_id': dp.source_city_id,
#         'destination_city_id': dp.destination_city_id,
#         'parcel_type': dp.parcel_type,
#         'route_direction': dp.route_direction
#     } for dp in datapoints]
    
#     df = pd.DataFrame(data_list)
    
#     if len(df) == 0:
#         raise ValueError("No data points found for retraining.")
        
#     X = df[FEATURE_COLS].values.astype(np.float32)
#     y = df[TARGET_COL].values.astype(np.int32)
    
#     return X, y

# # --- 4. DATA MODELS (Pydantic Schemas) ---
# # (Skipped for brevity, as they remain the same)

# # --- 5. APPLICATION LIFESPAN & MODEL LOADING ---

# smart_conveyor_model = None

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global smart_conveyor_model
    
#     try:
#         print("Database structure check complete.")
#     except OperationalError as e:
#         print(f"Could not connect to database or check tables: {e}")

#     # 2. Load the Model using joblib
#     try:
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         model_path = os.path.join(current_dir, MODEL_FILENAME)
#         if os.path.exists(model_path):
#             smart_conveyor_model = joblib.load(model_path)
#             print("Model loaded successfully using joblib.")
#         else:
#             print(f"WARNING: Model file not found at {model_path}. Run train_model.py first.")
#     except Exception as e:
#         print(f"ERROR: Could not load the model: {e}")
        
#     yield
    
# # --- 6. FASTAPI APP INITIALIZATION ---

# app = FastAPI(lifespan=lifespan, 
#               title="Smart Conveyor API",
#               description="Backend for the smart conveyor belt system using FastAPI.")

# origins = [
#     "http://localhost:5173",
#     "http://127.0.0.1:5173",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --- 7. DEPENDENCIES (Authentication) ---

# async def authenticate_user(login_data: LoginRequest, db, model_cls):
#     """Authenticates a user against a given login table model."""
#     user = db.query(model_cls).filter(model_cls.username == login_data.username).first()
#     if not user or not verify_password(login_data.password, user.hashed_password):
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid credentials",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#     return {"id": user.id, "username": user.username}

# async def get_current_admin(login_data: LoginRequest = Depends(), db: SessionLocal = Depends(get_db)):
#     """Dependency for Admin authentication."""
#     return await authenticate_user(login_data, db, AdminLogin)

# async def get_current_employee(login_data: LoginRequest = Depends(), db: SessionLocal = Depends(get_db)):
#     """Dependency for Employee authentication."""
#     return await authenticate_user(login_data, db, EmployeeLogin)


# # --- 8. ENDPOINTS ---
# # (Endpoints remain the same)
# @app.get("/")
# def read_root():
#     return {"message": "Smart Conveyor API is running!"}

# # --- AUTHENTICATION ENDPOINTS ---

# @app.post("/auth/admin/login")
# async def admin_login(authenticated_user: Dict = Depends(get_current_admin)):
#     return {"message": "Admin login successful", "username": authenticated_user['username'], "user_id": authenticated_user['id']}

# @app.post("/auth/employee/login")
# async def employee_login(authenticated_user: Dict = Depends(get_current_employee)):
#     return {"message": "Employee login successful", "username": authenticated_user['username'], "user_id": authenticated_user['id']}

# # --- ADMIN ENDPOINTS ---

# @app.post("/admin/new_admin", status_code=status.HTTP_201_CREATED)
# async def add_new_admin(new_user: NewUserRequest, db: SessionLocal = Depends(get_db)):
#     username, password = generate_credentials(new_user.email)
#     hashed_password = hash_password(password)
    
#     new_admin = AdminLogin(username=username, hashed_password=hashed_password)
    
#     try:
#         db.add(new_admin)
#         db.commit()
#         return {
#             "message": "New admin added successfully. **Save these credentials immediately.**",
#             "username": username,
#             "password": password, 
#         }
#     except IntegrityError:
#         db.rollback()
#         raise HTTPException(status_code=400, detail="Username already exists (try again)")

# @app.post("/admin/new_employee", status_code=status.HTTP_201_CREATED)
# async def add_new_employee(new_user: NewUserRequest, db: SessionLocal = Depends(get_db)):
#     username, password = generate_credentials(new_user.email)
#     hashed_password = hash_password(password)
    
#     new_employee = EmployeeLogin(username=username, hashed_password=hashed_password, email=new_user.email)
    
#     try:
#         db.add(new_employee)
#         db.commit()
#         return {
#             "message": "New employee added successfully. **Save these credentials immediately.**",
#             "username": username,
#             "password": password, 
#             "email": new_user.email
#         }
#     except IntegrityError:
#         db.rollback()
#         raise HTTPException(status_code=400, detail="Username or email already exists")

# @app.post("/admin/edit_profile/{admin_id}")
# async def edit_admin_profile(admin_id: int, profile_data: ProfileUpdate, db: SessionLocal = Depends(get_db)):
#     admin = db.query(AdminLogin).filter(AdminLogin.id == admin_id).first()
    
#     if not admin:
#         raise HTTPException(status_code=404, detail="Admin not found")
        
#     if profile_data.new_username != admin.username:
#         if db.query(AdminLogin).filter(AdminLogin.username == profile_data.new_username).first():
#             raise HTTPException(status_code=400, detail="New username already taken")
#         admin.username = profile_data.new_username
        
#     admin.hashed_password = hash_password(profile_data.new_password)
    
#     db.commit()
#     return {"message": "Admin profile updated successfully", "new_username": admin.username}

# # --- EMPLOYEE ENDPOINTS ---

# @app.get("/employee/cities", response_model=List[CityRead])
# async def get_all_cities(db: SessionLocal = Depends(get_db)):
#     return db.query(City).order_by(City.unique_id).all()

# @app.post("/employee/add_city", response_model=CityRead, status_code=status.HTTP_201_CREATED)
# async def add_new_city(city_data: CityBase, db: SessionLocal = Depends(get_db)):
#     next_id = get_next_city_id(db)
#     new_city = City(city_name=city_data.city_name, unique_id=next_id)
    
#     try:
#         db.add(new_city)
#         db.commit()
#         db.refresh(new_city)
#         return new_city
#     except IntegrityError:
#         db.rollback()
#         raise HTTPException(status_code=400, detail="City name already exists")

# @app.post("/employee/add_datapoint", response_model=DataPointRead, status_code=status.HTTP_201_CREATED)
# async def add_new_datapoint(data_point: DataPointBase, db: SessionLocal = Depends(get_db)):
#     try:
#         parcel_type_int = PARCEL_MAP[data_point.parcel_type_text]
#         route_direction_int = ROUTE_MAP_REV[data_point.route_direction_text]
#     except KeyError as e:
#         raise HTTPException(status_code=400, detail=f"Invalid mapping for {e}")
        
#     new_dp = DataPoint(
#         source_city_id=data_point.source_city_id,
#         destination_city_id=data_point.destination_city_id,
#         parcel_type=parcel_type_int,
#         route_direction=route_direction_int
#     )
    
#     try:
#         db.add(new_dp)
#         db.commit()
#         db.refresh(new_dp)
        
#         cities = db.query(City).all()
#         city_map = {c.unique_id: c.city_name for c in cities}
        
#         return DataPointRead(
#             id=new_dp.id,
#             source_city_id=new_dp.source_city_id,
#             destination_city_id=new_dp.destination_city_id,
#             parcel_type_text=PARCEL_MAP_REV.get(new_dp.parcel_type, 'unknown'),
#             route_direction_text=ROUTE_MAP.get(new_dp.route_direction, 'unknown'),
#             source_city_name=city_map.get(new_dp.source_city_id),
#             destination_city_name=city_map.get(new_dp.destination_city_id)
#         )
#     except Exception as e:
#         db.rollback()
#         raise HTTPException(status_code=500, detail=f"Database error: {e}")

# @app.get("/employee/datapoints", response_model=List[DataPointRead])
# async def get_datapoints(db: SessionLocal = Depends(get_db)):
#     datapoints = db.query(DataPoint).order_by(DataPoint.id).all()
#     cities = db.query(City).all()
#     city_map = {c.unique_id: c.city_name for c in cities}
    
#     result = []
#     for dp in datapoints:
#         result.append(DataPointRead(
#             id=dp.id,
#             source_city_id=dp.source_city_id,
#             destination_city_id=dp.destination_city_id,
#             parcel_type_text=PARCEL_MAP_REV.get(dp.parcel_type, 'unknown'),
#             route_direction_text=ROUTE_MAP.get(dp.route_direction, 'unknown'),
#             source_city_name=city_map.get(dp.source_city_id),
#             destination_city_name=city_map.get(dp.destination_city_id)
#         ))
        
#     return result

# @app.put("/employee/update_datapoint/{datapoint_id}")
# async def update_datapoint(datapoint_id: int, data_point: DataPointBase, db: SessionLocal = Depends(get_db)):
#     db_dp = db.query(DataPoint).filter(DataPoint.id == datapoint_id).first()
    
#     if not db_dp:
#         raise HTTPException(status_code=404, detail="DataPoint not found")
        
#     try:
#         db_dp.route_direction = ROUTE_MAP_REV[data_point.route_direction_text]
#         db.commit()
#     except KeyError:
#         raise HTTPException(status_code=400, detail="Invalid route direction")
#     except Exception as e:
#         db.rollback()
#         raise HTTPException(status_code=500, detail=f"Database error: {e}")
        
#     return {"message": "DataPoint updated successfully"}

# @app.post("/employee/retrain_model")
# async def retrain_model(db: SessionLocal = Depends(get_db)):
#     global smart_conveyor_model
    
#     datapoints = db.query(DataPoint).all()
#     if not datapoints:
#         raise HTTPException(status_code=400, detail="No data points found in the database for retraining.")
        
#     try:
#         X, y = prepare_retrain_data(datapoints)

#         if len(X) < 10:
#             X_train, y_train = X, y
#             X_test, y_test = X, y
#         else:
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         model = MLPClassifier(hidden_layer_sizes=(50,), 
#                               max_iter=500, 
#                               activation='relu', 
#                               solver='adam', 
#                               random_state=42)

#         model.fit(X_train, y_train)
#         accuracy = model.score(X_test, y_test)

#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         model_path = os.path.join(current_dir, MODEL_FILENAME)
#         joblib.dump(model, model_path)
#         smart_conveyor_model = model

#         return {"message": "Model retrained successfully!", "test_accuracy": f"{accuracy:.4f}", "new_data_size": len(X)}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Model retraining failed: {e}")

# @app.post("/employee/edit_profile/{employee_id}")
# async def edit_employee_profile(employee_id: int, profile_data: ProfileUpdate, db: SessionLocal = Depends(get_db)):
#     employee = db.query(EmployeeLogin).filter(EmployeeLogin.id == employee_id).first()
    
#     if not employee:
#         raise HTTPException(status_code=404, detail="Employee not found")
        
#     if profile_data.new_username != employee.username:
#         if db.query(EmployeeLogin).filter(EmployeeLogin.username == profile_data.new_username).first():
#             raise HTTPException(status_code=400, detail="New username already taken")
#         employee.username = profile_data.new_username
        
#     employee.hashed_password = hash_password(profile_data.new_password)
    
#     db.commit()
#     return {"message": "Employee profile updated successfully", "new_username": employee.username}


# # --- USER ENDPOINTS (Prediction) ---

# @app.post("/user/predict_route")
# async def predict_route(request: PredictionRequest, db: SessionLocal = Depends(get_db)):
#     global smart_conveyor_model

#     if smart_conveyor_model is None:
#         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded or trained yet.")
        
#     try:
#         parcel_type_int = PARCEL_MAP[request.parcel_type_text]
#         features = np.array([[
#             request.source_city_id, 
#             request.destination_city_id, 
#             parcel_type_int
#         ]])
        
#         prediction_int = smart_conveyor_model.predict(features)[0]
#         predicted_route = ROUTE_MAP.get(prediction_int, 'unknown')
        
#         cities = db.query(City).filter(City.unique_id.in_([request.source_city_id, request.destination_city_id])).all()
#         city_map = {c.unique_id: c.city_name for c in cities}

#         return {
#             "source_id": request.source_city_id,
#             "destination_id": request.destination_city_id,
#             "source_city_name": city_map.get(request.source_city_id, 'ID Not Found'),
#             "destination_city_name": city_map.get(request.destination_city_id, 'ID Not Found'),
#             "parcel_type": request.parcel_type_text,
#             "predicted_route": predicted_route
#         }

#     except KeyError:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid parcel type in request.")
#     except Exception as e:
#         print(f"Prediction error: {e}")
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred during prediction: {e}")

# server/main.py
import os
from contextlib import asynccontextmanager
from typing import List, Dict, Union, Optional
import random
import string
import hashlib
import dotenv 
# Database & ORM
from sqlalchemy import create_engine, Column, Integer, String, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError, OperationalError
# FastAPI & Pydantic
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
# Model/Data handling (using joblib for scikit-learn model)
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# --- Load Environment Variables ---
dotenv.load_dotenv()

# --- 1. CONFIGURATION & MAPPINGS ---

# Your Supabase Connection String loaded from .env
DATABASE_URL = os.getenv("DATABASE_URL")

# --- Essential check for running the app ---
if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL not found. Please ensure it is correctly set in your .env file."
    )

MODEL_FILENAME = 'routing_model.pkl'
ROUTE_MAP = {0: 'straight', 1: 'left', 2: 'right'}
ROUTE_MAP_REV = {'straight': 0, 'left': 1, 'right': 2}
PARCEL_MAP = {'normal': 0, 'fast': 1}
PARCEL_MAP_REV = {0: 'normal', 1: 'fast'}
FEATURE_COLS = ['source_city_id', 'destination_city_id', 'parcel_type']
TARGET_COL = 'route_direction'


# --- 2. DATA MODELS (Pydantic Schemas - MOVED TO TOP) ---

class CityBase(BaseModel):
    city_name: str = Field(..., min_length=1, description="Name of the city.")

class CityRead(CityBase):
    id: int
    unique_id: int

    class Config:
        from_attributes = True

class LoginRequest(BaseModel):
    username: str
    password: str

class ProfileUpdate(BaseModel):
    new_username: str
    new_password: str

    @validator('new_username')
    def validate_username(cls, v):
        if not v or len(v) < 3:
            raise ValueError('Username must be at least 3 characters long')
        return v

class NewUserRequest(BaseModel):
    email: str

    @validator('email')
    def validate_email(cls, v):
        if "@" not in v or "." not in v:
            raise ValueError('Invalid email format')
        return v

class DataPointBase(BaseModel):
    source_city_id: int
    destination_city_id: int
    parcel_type_text: str = Field(..., description="Parcel type: 'normal' or 'fast'")
    route_direction_text: Optional[str] = Field(None, description="Route: 'straight', 'left', 'right'")
    
    @validator('parcel_type_text')
    def validate_parcel_type(cls, v):
        if v.lower() not in PARCEL_MAP:
            raise ValueError(f"Parcel type must be one of: {list(PARCEL_MAP.keys())}")
        return v.lower()
    
    @validator('route_direction_text')
    def validate_route_direction(cls, v):
        if v is not None and v.lower() not in ROUTE_MAP_REV:
            raise ValueError(f"Route direction must be one of: {list(ROUTE_MAP_REV.keys())}")
        return v.lower() if v else None

class DataPointRead(DataPointBase):
    id: int
    source_city_name: Optional[str] = None
    destination_city_name: Optional[str] = None
    
    class Config:
        from_attributes = True

class PredictionRequest(BaseModel):
    source_city_id: int
    destination_city_id: int
    parcel_type_text: str = Field(..., description="Parcel type: 'normal' or 'fast'")
    
    @validator('parcel_type_text')
    def validate_parcel_type(cls, v):
        if v.lower() not in PARCEL_MAP:
            raise ValueError(f"Parcel type must be one of: {list(PARCEL_MAP.keys())}")
        return v.lower()

# --- 3. DATABASE SETUP (ORM Definitions) ---

Base = declarative_base()

class AdminLogin(Base):
    __tablename__ = 'admin-logins'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

class EmployeeLogin(Base):
    __tablename__ = 'employee-logins'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=True)

class City(Base):
    __tablename__ = 'cities'
    id = Column(Integer, primary_key=True)
    city_name = Column(String, unique=True, index=True, nullable=False)
    unique_id = Column(Integer, unique=True, index=True, nullable=False)

class DataPoint(Base):
    __tablename__ = 'datapoints'
    id = Column(Integer, primary_key=True, index=True)
    source_city_id = Column(Integer, index=True, nullable=False)
    destination_city_id = Column(Integer, index=True, nullable=False)
    parcel_type = Column(Integer, nullable=False)
    route_direction = Column(Integer, nullable=False)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- 4. HELPER FUNCTIONS ---

def hash_password(password: str) -> str:
    """Hashes a password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a hash."""
    return hash_password(plain_password) == hashed_password

def generate_credentials(email: str) -> tuple:
    """Generates a random username and one-time password."""
    username = email.split('@')[0] + "".join(random.choices(string.digits, k=4))
    password = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    return username, password

def get_next_city_id(db) -> int:
    """Gets the next unique_id for a new city."""
    max_id = db.query(text("MAX(unique_id)")).select_from(City).scalar()
    return (max_id or 0) + 1
    
def prepare_retrain_data(datapoints: List[DataPoint]) -> tuple:
    """Prepares data from DB models for model retraining."""
    data_list = [{
        'source_city_id': dp.source_city_id,
        'destination_city_id': dp.destination_city_id,
        'parcel_type': dp.parcel_type,
        'route_direction': dp.route_direction
    } for dp in datapoints]
    
    df = pd.DataFrame(data_list)
    
    if len(df) == 0:
        raise ValueError("No data points found for retraining.")
        
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int32)
    
    return X, y

# --- 5. APPLICATION LIFESPAN & MODEL LOADING ---

smart_conveyor_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global smart_conveyor_model
    
    try:
        print("Database structure check complete.")
    except OperationalError as e:
        print(f"Could not connect to database or check tables: {e}")

    # 2. Load the Model using joblib
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, MODEL_FILENAME)
        if os.path.exists(model_path):
            smart_conveyor_model = joblib.load(model_path)
            print("Model loaded successfully using joblib.")
        else:
            print(f"WARNING: Model file not found at {model_path}. Run train_model.py first.")
    except Exception as e:
        print(f"ERROR: Could not load the model: {e}")
        
    yield
    
# --- 6. FASTAPI APP INITIALIZATION ---

app = FastAPI(lifespan=lifespan, 
              title="Smart Conveyor API",
              description="Backend for the smart conveyor belt system using FastAPI.")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 7. DEPENDENCIES (Authentication) ---

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def authenticate_user(login_data: LoginRequest, db, model_cls):
    """Authenticates a user against a given login table model."""
    user = db.query(model_cls).filter(model_cls.username == login_data.username).first()
    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"id": user.id, "username": user.username}

async def get_current_admin(login_data: LoginRequest = Depends(), db: SessionLocal = Depends(get_db)):
    """Dependency for Admin authentication."""
    return await authenticate_user(login_data, db, AdminLogin)

async def get_current_employee(login_data: LoginRequest = Depends(), db: SessionLocal = Depends(get_db)):
    """Dependency for Employee authentication."""
    return await authenticate_user(login_data, db, EmployeeLogin)


# --- 8. ENDPOINTS ---

@app.get("/")
def read_root():
    return {"message": "Smart Conveyor API is running!"}

# --- AUTHENTICATION ENDPOINTS ---

@app.post("/auth/admin/login")
async def admin_login(authenticated_user: Dict = Depends(get_current_admin)):
    return {"message": "Admin login successful", "username": authenticated_user['username'], "user_id": authenticated_user['id']}

@app.post("/auth/employee/login")
async def employee_login(authenticated_user: Dict = Depends(get_current_employee)):
    return {"message": "Employee login successful", "username": authenticated_user['username'], "user_id": authenticated_user['id']}

# --- ADMIN ENDPOINTS ---

@app.post("/admin/new_admin", status_code=status.HTTP_201_CREATED)
async def add_new_admin(new_user: NewUserRequest, db: SessionLocal = Depends(get_db)):
    username, password = generate_credentials(new_user.email)
    hashed_password = hash_password(password)
    
    new_admin = AdminLogin(username=username, hashed_password=hashed_password)
    
    try:
        db.add(new_admin)
        db.commit()
        return {
            "message": "New admin added successfully. **Save these credentials immediately.**",
            "username": username,
            "password": password, 
        }
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Username already exists (try again)")

@app.post("/admin/new_employee", status_code=status.HTTP_201_CREATED)
async def add_new_employee(new_user: NewUserRequest, db: SessionLocal = Depends(get_db)):
    username, password = generate_credentials(new_user.email)
    hashed_password = hash_password(password)
    
    new_employee = EmployeeLogin(username=username, hashed_password=hashed_password, email=new_user.email)
    
    try:
        db.add(new_employee)
        db.commit()
        return {
            "message": "New employee added successfully. **Save these credentials immediately.**",
            "username": username,
            "password": password, 
            "email": new_user.email
        }
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Username or email already exists")

@app.post("/admin/edit_profile/{admin_id}")
async def edit_admin_profile(admin_id: int, profile_data: ProfileUpdate, db: SessionLocal = Depends(get_db)):
    admin = db.query(AdminLogin).filter(AdminLogin.id == admin_id).first()
    
    if not admin:
        raise HTTPException(status_code=404, detail="Admin not found")
        
    if profile_data.new_username != admin.username:
        if db.query(AdminLogin).filter(AdminLogin.username == profile_data.new_username).first():
            raise HTTPException(status_code=400, detail="New username already taken")
        admin.username = profile_data.new_username
        
    admin.hashed_password = hash_password(profile_data.new_password)
    
    db.commit()
    return {"message": "Admin profile updated successfully", "new_username": admin.username}

# --- EMPLOYEE ENDPOINTS ---

@app.get("/employee/cities", response_model=List[CityRead])
async def get_all_cities(db: SessionLocal = Depends(get_db)):
    return db.query(City).order_by(City.unique_id).all()

@app.post("/employee/add_city", response_model=CityRead, status_code=status.HTTP_201_CREATED)
async def add_new_city(city_data: CityBase, db: SessionLocal = Depends(get_db)):
    next_id = get_next_city_id(db)
    new_city = City(city_name=city_data.city_name, unique_id=next_id)
    
    try:
        db.add(new_city)
        db.commit()
        db.refresh(new_city)
        return new_city
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="City name already exists")

@app.post("/employee/add_datapoint", response_model=DataPointRead, status_code=status.HTTP_201_CREATED)
async def add_new_datapoint(data_point: DataPointBase, db: SessionLocal = Depends(get_db)):
    try:
        parcel_type_int = PARCEL_MAP[data_point.parcel_type_text]
        route_direction_int = ROUTE_MAP_REV[data_point.route_direction_text]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Invalid mapping for {e}")
        
    new_dp = DataPoint(
        source_city_id=data_point.source_city_id,
        destination_city_id=data_point.destination_city_id,
        parcel_type=parcel_type_int,
        route_direction=route_direction_int
    )
    
    try:
        db.add(new_dp)
        db.commit()
        db.refresh(new_dp)
        
        cities = db.query(City).all()
        city_map = {c.unique_id: c.city_name for c in cities}
        
        return DataPointRead(
            id=new_dp.id,
            source_city_id=new_dp.source_city_id,
            destination_city_id=new_dp.destination_city_id,
            parcel_type_text=PARCEL_MAP_REV.get(new_dp.parcel_type, 'unknown'),
            route_direction_text=ROUTE_MAP.get(new_dp.route_direction, 'unknown'),
            source_city_name=city_map.get(new_dp.source_city_id),
            destination_city_name=city_map.get(new_dp.destination_city_id)
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@app.get("/employee/datapoints", response_model=List[DataPointRead])
async def get_datapoints(db: SessionLocal = Depends(get_db)):
    datapoints = db.query(DataPoint).order_by(DataPoint.id).all()
    cities = db.query(City).all()
    city_map = {c.unique_id: c.city_name for c in cities}
    
    result = []
    for dp in datapoints:
        result.append(DataPointRead(
            id=dp.id,
            source_city_id=dp.source_city_id,
            destination_city_id=dp.destination_city_id,
            parcel_type_text=PARCEL_MAP_REV.get(dp.parcel_type, 'unknown'),
            route_direction_text=ROUTE_MAP.get(dp.route_direction, 'unknown'),
            source_city_name=city_map.get(dp.source_city_id),
            destination_city_name=city_map.get(dp.destination_city_id)
        ))
        
    return result

@app.put("/employee/update_datapoint/{datapoint_id}")
async def update_datapoint(datapoint_id: int, data_point: DataPointBase, db: SessionLocal = Depends(get_db)):
    db_dp = db.query(DataPoint).filter(DataPoint.id == datapoint_id).first()
    
    if not db_dp:
        raise HTTPException(status_code=404, detail="DataPoint not found")
        
    try:
        db_dp.route_direction = ROUTE_MAP_REV[data_point.route_direction_text]
        db.commit()
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid route direction")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
        
    return {"message": "DataPoint updated successfully"}

@app.post("/employee/retrain_model")
async def retrain_model(db: SessionLocal = Depends(get_db)):
    global smart_conveyor_model
    
    datapoints = db.query(DataPoint).all()
    if not datapoints:
        raise HTTPException(status_code=400, detail="No data points found in the database for retraining.")
        
    try:
        X, y = prepare_retrain_data(datapoints)

        if len(X) < 10:
            X_train, y_train = X, y
            X_test, y_test = X, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = MLPClassifier(hidden_layer_sizes=(50,), 
                              max_iter=500, 
                              activation='relu', 
                              solver='adam', 
                              random_state=42)

        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, MODEL_FILENAME)
        joblib.dump(model, model_path)
        smart_conveyor_model = model

        return {"message": "Model retrained successfully!", "test_accuracy": f"{accuracy:.4f}", "new_data_size": len(X)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model retraining failed: {e}")

@app.post("/employee/edit_profile/{employee_id}")
async def edit_employee_profile(employee_id: int, profile_data: ProfileUpdate, db: SessionLocal = Depends(get_db)):
    employee = db.query(EmployeeLogin).filter(EmployeeLogin.id == employee_id).first()
    
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
        
    if profile_data.new_username != employee.username:
        if db.query(EmployeeLogin).filter(EmployeeLogin.username == profile_data.new_username).first():
            raise HTTPException(status_code=400, detail="New username already taken")
        employee.username = profile_data.new_username
        
    employee.hashed_password = hash_password(profile_data.new_password)
    
    db.commit()
    return {"message": "Employee profile updated successfully", "new_username": employee.username}


# --- USER ENDPOINTS (Prediction) ---

@app.post("/user/predict_route")
async def predict_route(request: PredictionRequest, db: SessionLocal = Depends(get_db)):
    global smart_conveyor_model

    if smart_conveyor_model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded or trained yet.")
        
    try:
        parcel_type_int = PARCEL_MAP[request.parcel_type_text]
        features = np.array([[
            request.source_city_id, 
            request.destination_city_id, 
            parcel_type_int
        ]])
        
        prediction_int = smart_conveyor_model.predict(features)[0]
        predicted_route = ROUTE_MAP.get(prediction_int, 'unknown')
        
        cities = db.query(City).filter(City.unique_id.in_([request.source_city_id, request.destination_city_id])).all()
        city_map = {c.unique_id: c.city_name for c in cities}

        return {
            "source_id": request.source_city_id,
            "destination_id": request.destination_city_id,
            "source_city_name": city_map.get(request.source_city_id, 'ID Not Found'),
            "destination_city_name": city_map.get(request.destination_city_id, 'ID Not Found'),
            "parcel_type": request.parcel_type_text,
            "predicted_route": predicted_route
        }

    except KeyError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid parcel type in request.")
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred during prediction: {e}")