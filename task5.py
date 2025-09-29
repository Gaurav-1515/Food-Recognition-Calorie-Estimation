import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

class FoodCalorieEstimator:
    def __init__(self, dataset_path, img_size=(224, 224)):
        """
        Initialize the Food Recognition and Calorie Estimation model
        
        Args:
            dataset_path (str): Path to the Food-101 dataset
            img_size (tuple): Input image size for the model
        """
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.model = None
        self.label_encoder = LabelEncoder()
        self.class_names = []
        
        # Calorie database (calories per 100g for each food class)
        self.calorie_db = self._create_calorie_database()
        
    def _create_calorie_database(self):
        """Create a comprehensive calorie database for Food-101 classes"""
        # Approximate calories per 100g for Food-101 categories
        calorie_data = {
            'apple_pie': 237, 'baby_back_ribs': 292, 'baklava': 428, 'beef_carpaccio': 135,
            'beef_tartare': 196, 'beet_salad': 88, 'beignets': 347, 'bibimbap': 121,
            'bread_pudding': 212, 'breakfast_burrito': 188, 'bruschetta': 195, 'caesar_salad': 158,
            'cannoli': 301, 'caprese_salad': 125, 'carrot_cake': 385, 'ceviche': 134,
            'cheese_plate': 368, 'cheesecake': 321, 'chicken_curry': 128, 'chicken_quesadilla': 218,
            'chicken_wings': 203, 'chocolate_cake': 371, 'chocolate_mousse': 168, 'churros': 367,
            'clam_chowder': 112, 'club_sandwich': 282, 'crab_cakes': 197, 'creme_brulee': 296,
            'croque_madame': 235, 'cup_cakes': 305, 'deviled_eggs': 155, 'donuts': 452,
            'dumplings': 241, 'edamame': 121, 'eggs_benedict': 238, 'escargots': 90,
            'falafel': 333, 'filet_mignon': 277, 'fish_and_chips': 232, 'foie_gras': 462,
            'french_fries': 365, 'french_onion_soup': 67, 'french_toast': 222, 'fried_calamari': 175,
            'fried_rice': 163, 'frozen_yogurt': 127, 'garlic_bread': 350, 'gnocchi': 131,
            'greek_salad': 150, 'grilled_cheese_sandwich': 291, 'grilled_salmon': 231, 'guacamole': 160,
            'gyoza': 201, 'hamburger': 295, 'hot_and_sour_soup': 56, 'hot_dog': 290,
            'huevos_rancheros': 149, 'hummus': 166, 'ice_cream': 207, 'lasagna': 135,
            'lobster_bisque': 110, 'lobster_roll_sandwich': 436, 'macaroni_and_cheese': 164, 'macarons': 407,
            'miso_soup': 40, 'mussels': 86, 'nachos': 346, 'omelette': 154,
            'onion_rings': 331, 'oysters': 68, 'pad_thai': 153, 'paella': 139,
            'pancakes': 227, 'panna_cotta': 133, 'peking_duck': 337, 'pho': 46,
            'pizza': 266, 'pork_chop': 231, 'poutine': 365, 'prime_rib': 291,
            'pulled_pork_sandwich': 227, 'ramen': 436, 'ravioli': 175, 'red_velvet_cake': 478,
            'risotto': 142, 'samosa': 308, 'sashimi': 127, 'scallops': 69,
            'seaweed_salad': 45, 'shrimp_and_grits': 149, 'spaghetti_bolognese': 151, 'spaghetti_carbonara': 174,
            'spring_rolls': 140, 'steak': 271, 'strawberry_shortcake': 227, 'sushi': 142,
            'tacos': 226, 'takoyaki': 112, 'tiramisu': 240, 'tuna_tartare': 144,
            'waffles': 291
        }
        return calorie_data
    
    def load_and_preprocess_data(self, sample_size=None):
        """
        Load and preprocess the Food-101 dataset
        
        Args:
            sample_size (int): Number of samples per class to use (for testing)
        """
        print("Loading Food-101 dataset...")
        
        # Check if dataset path exists
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        
        # Try to load class names from meta/classes.txt first
        classes_file = os.path.join(self.dataset_path, 'meta', 'classes.txt')
        if os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
        else:
            # If meta folder doesn't exist, get class names from directory structure
            images_path = os.path.join(self.dataset_path, 'images')
            if os.path.exists(images_path):
                self.class_names = sorted([d for d in os.listdir(images_path) 
                                         if os.path.isdir(os.path.join(images_path, d))])
            else:
                raise FileNotFoundError(f"Neither meta/classes.txt nor images directory found in {self.dataset_path}")
        
        print(f"Found {len(self.class_names)} food classes")
        
        # Create data generators with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            validation_split=0.2
        )
        
        # Training generator
        self.train_generator = train_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'images'),
            target_size=self.img_size,
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        
        # Validation generator
        self.validation_generator = train_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'images'),
            target_size=self.img_size,
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )
        
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.validation_generator.samples}")
        print(f"Number of classes: {len(self.class_names)}")
        
    def build_model(self):
        """Build the food recognition model using transfer learning"""
        print("Building model with EfficientNet backbone...")
        
        # Load pre-trained EfficientNet
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification head
        inputs = layers.Input(shape=(*self.img_size, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(len(self.class_names), activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model built successfully!")
        self.model.summary()
        
    def train_model(self, epochs=50):
        """
        Train the food recognition model
        
        Args:
            epochs (int): Number of training epochs
        """
        print("Starting training...")
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7)
        ]
        
        # Initial training with frozen base
        history = self.model.fit(
            self.train_generator,
            epochs=epochs//2,
            validation_data=self.validation_generator,
            callbacks=callbacks
        )
        
        # Fine-tuning: unfreeze top layers of base model
        print("Fine-tuning model...")
        self.model.layers[1].trainable = True
        
        # Freeze first 80% of layers
        fine_tune_at = int(len(self.model.layers[1].layers) * 0.8)
        for layer in self.model.layers[1].layers[:fine_tune_at]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001/10),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Continue training
        history_fine = self.model.fit(
            self.train_generator,
            epochs=epochs//2,
            validation_data=self.validation_generator,
            callbacks=callbacks,
            initial_epoch=len(history.history['loss'])
        )
        
        return history, history_fine
    
    def predict_food_and_calories(self, image_path, portion_size=100):
        """
        Predict food class and estimate calories
        
        Args:
            image_path (str): Path to the image
            portion_size (float): Portion size in grams
        
        Returns:
            dict: Prediction results with food class, confidence, and calories
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = self.model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get food class name
        predicted_food = list(self.train_generator.class_indices.keys())[predicted_class_idx]
        
        # Estimate calories
        base_calories = self.calorie_db.get(predicted_food, 200)  # Default 200 if not found
        estimated_calories = (base_calories * portion_size) / 100
        
        return {
            'food_class': predicted_food,
            'confidence': confidence,
            'calories_per_100g': base_calories,
            'portion_size_g': portion_size,
            'estimated_calories': estimated_calories
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def visualize_predictions(self, test_images, num_images=9):
        """Visualize predictions on test images"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.ravel()
        
        for i, img_path in enumerate(test_images[:num_images]):
            # Make prediction
            result = self.predict_food_and_calories(img_path)
            
            # Display image
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].set_title(
                f"Food: {result['food_class']}\n"
                f"Confidence: {result['confidence']:.2f}\n"
                f"Calories: {result['estimated_calories']:.0f}"
            )
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

# Example usage and training pipeline
def main():
    # IMPORTANT: UPDATE THIS PATH TO YOUR ACTUAL DATASET LOCATION
    # Current placeholder path - you MUST change this!
    dataset_path = r"D:\Internships_tasks\prodigy_internship\task5\food-101"
    
    # Check if the user forgot to update the path
    if dataset_path == r"D:\Internships_tasks\prodigy_internship\task5\food-101" and not os.path.exists(dataset_path):
        print(" DATASET PATH NOT CONFIGURED! 🚨")
        print("=" * 50)
        print("You need to:")
        print("1. Download the Food-101 dataset")
        print("2. Update the 'dataset_path' variable in main() function")
        print("")
        print("Download options:")
        print("• Kaggle: https://www.kaggle.com/dansbecker/food-101")
        print("• Direct: http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz")
        print("")
        print("Then update this line in main():")
        print(f"dataset_path = r'YOUR_ACTUAL_PATH_HERE'")
        print("")
        print("Example paths:")
        print("• Windows: r'C:\\Users\\YourName\\Downloads\\food-101'")
        print("• Linux/Mac: '/home/username/datasets/food-101'")
        print("=" * 50)
        return
    
    # Check if path exists
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path not found: {dataset_path}")
        print("")
        print("Please check:")
        print("1. Is the path correct?")
        print("2. Did you extract the dataset?")
        print("3. Does the folder contain 'images' and 'meta' subfolders?")
        return
    
    # Verify basic structure
    images_path = os.path.join(dataset_path, 'images')
    if not os.path.exists(images_path):
        print(f"ERROR: Images folder not found in {dataset_path}")
        print("Expected structure:")
        print("food-101/")
        print("  ├── images/")
        print("  │   ├── apple_pie/")
        print("  │   └── ...")
        print("  └── meta/ (optional)")
        return
    
    print(f"✅ Using dataset at: {dataset_path}")
    
    # Initialize the model
    food_estimator = FoodCalorieEstimator(
        dataset_path=dataset_path,
        img_size=(224, 224)
    )
    
    try:
        # Load and preprocess data
        food_estimator.load_and_preprocess_data()
        
        # Build model
        food_estimator.build_model()
        
        # Train model
        print("Starting training... This will take a while!")
        history = food_estimator.train_model(epochs=30)
        
        # Save model
        food_estimator.save_model('food_calorie_model.h5')
        print(" Model training completed and saved!")
        
    except Exception as e:
        print(f" Error during execution: {e}")
        print("Please check your dataset structure and try again.")

if __name__ == "__main__":
    main()

# Additional utility functions for calorie tracking

class CalorieTracker:
    """Simple calorie tracking system"""
    
    def __init__(self):
        self.daily_intake = []
        self.daily_goal = 2000  # Default daily calorie goal
    
    def add_food_item(self, food_result):
        """Add a food item to daily intake"""
        self.daily_intake.append({
            'timestamp': pd.Timestamp.now(),
            'food': food_result['food_class'],
            'calories': food_result['estimated_calories'],
            'portion_size': food_result['portion_size_g']
        })
    
    def get_daily_summary(self):
        """Get summary of daily calorie intake"""
        if not self.daily_intake:
            return {"total_calories": 0, "remaining": self.daily_goal, "items": 0}
        
        df = pd.DataFrame(self.daily_intake)
        today_intake = df[df['timestamp'].dt.date == pd.Timestamp.now().date()]
        total_calories = today_intake['calories'].sum()
        
        return {
            "total_calories": total_calories,
            "remaining": self.daily_goal - total_calories,
            "items": len(today_intake),
            "breakdown": today_intake.groupby('food')['calories'].sum().to_dict()
        }
    
    def visualize_intake(self):
        """Visualize daily calorie intake"""
        if not self.daily_intake:
            print("No food items tracked yet!")
            return
        
        df = pd.DataFrame(self.daily_intake)
        today_intake = df[df['timestamp'].dt.date == pd.Timestamp.now().date()]
        
        if today_intake.empty:
            print("No food items tracked today!")
            return
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart of food breakdown
        food_calories = today_intake.groupby('food')['calories'].sum()
        ax1.pie(food_calories.values, labels=food_calories.index, autopct='%1.1f%%')
        ax1.set_title('Today\'s Calorie Breakdown by Food Type')
        
        # Progress bar
        total_calories = today_intake['calories'].sum()
        remaining = max(0, self.daily_goal - total_calories)
        
        ax2.barh(['Consumed', 'Remaining'], [total_calories, remaining], 
                color=['#ff6b6b', '#51cf66'])
        ax2.set_xlabel('Calories')
        ax2.set_title(f'Daily Progress (Goal: {self.daily_goal} cal)')
        
        plt.tight_layout()
        plt.show()

# Performance evaluation functions
def evaluate_model_performance(model, test_generator):
    """Evaluate model performance on test data"""
    # Generate predictions
    test_generator.reset()
    predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true labels
    true_classes = test_generator.classes[:len(predicted_classes)]
    class_labels = list(test_generator.class_indices.keys())
    
    # Classification report
    print("Classification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))
    
    # Confusion matrix for top 10 classes
    top_classes = np.bincount(true_classes).argsort()[-10:][::-1]
    
    cm = confusion_matrix(true_classes, predicted_classes)
    cm_subset = cm[np.ix_(top_classes, top_classes)]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues',
                xticklabels=[class_labels[i] for i in top_classes],
                yticklabels=[class_labels[i] for i in top_classes])
    plt.title('Confusion Matrix (Top 10 Classes)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

print("Food Recognition and Calorie Estimation Model Ready!")
print("Key Features:")
print("1. Transfer learning with EfficientNet for accurate food recognition")
print("2. Comprehensive calorie database for 101 food categories")
print("3. Portion size adjustment for calorie estimation")
print("4. Built-in calorie tracking system")
print("5. Performance evaluation and visualization tools")
print("\nTo use: Update the dataset_path and run the main() function")
# Quick Dataset Path Checker
# Run this first to verify your Food-101 dataset is set up correctly

import os

def check_dataset_path():
    """Check if Food-101 dataset exists and is properly structured"""
    
    print(" Food-101 Dataset Checker")
    print("=" * 40)
    
    # Common possible locations
    possible_paths = [
        r"D:\Internships_tasks\prodigy_internship\task5\food-101",
        r"D:\Internships_tasks\prodigy_internship\task5\food-101-dataset",
        r".\food-101",
        r".\dataset\food-101",
        r"C:\Users\%USERNAME%\Downloads\food-101"
    ]
    
    print("Checking common locations:")
    found_paths = []
    
    for path in possible_paths:
        # Expand environment variables
        expanded_path = os.path.expandvars(path)
        print(f"   {expanded_path}")
        
        if os.path.exists(expanded_path):
            images_path = os.path.join(expanded_path, 'images')
            if os.path.exists(images_path):
                # Count classes
                class_dirs = [d for d in os.listdir(images_path) 
                            if os.path.isdir(os.path.join(images_path, d))]
                print(f"      FOUND! {len(class_dirs)} classes")
                found_paths.append((expanded_path, len(class_dirs)))
            else:
                print(f"      No 'images' folder")
        else:
            print(f"      Path not found")
    
    print("\n" + "=" * 40)
    
    if found_paths:
        print(" DATASET FOUND!")
        for path, num_classes in found_paths:
            print(f" {path} ({num_classes} classes)")
            
        # Use the first valid path
        best_path = found_paths[0][0]
        print(f"\n Recommended path:")
        print(f"dataset_path = r'{best_path}'")
        
        # Create a simple test file
        with open('dataset_path.txt', 'w') as f:
            f.write(best_path)
        print(f" Path saved to 'dataset_path.txt'")
        
        return best_path
        
    else:
        print(" NO DATASET FOUND!")
        print("\n You need to download the Food-101 dataset:")
        print("1. Go to: https://www.kaggle.com/dansbecker/food-101")
        print("2. Download the dataset")
        print("3. Extract it to one of these locations:")
        for path in possible_paths[:3]:
            print(f"   • {os.path.expandvars(path)}")
        
        return None

def create_test_structure():
    """Create a minimal test structure to verify code works"""
    print("\n Creating test structure...")
    
    test_path = r".\food-101-test"
    os.makedirs(os.path.join(test_path, "images", "pizza"), exist_ok=True)
    os.makedirs(os.path.join(test_path, "images", "burger"), exist_ok=True)
    os.makedirs(os.path.join(test_path, "meta"), exist_ok=True)
    
    # Create classes.txt
    with open(os.path.join(test_path, "meta", "classes.txt"), 'w') as f:
        f.write("pizza\nburger\n")
    
    print(f" Test structure created at: {os.path.abspath(test_path)}")
    print("Note: This is just for testing - you still need real images!")
    
    return test_path

if __name__ == "__main__":
    # Check for existing dataset
    found_path = check_dataset_path()
    
    if not found_path:
        # Ask user if they want a test structure
        create_test = input("\nCreate a test structure to verify code works? (y/n): ").lower()
        if create_test == 'y':
            test_path = create_test_structure()
            print(f"\nUse this for testing: dataset_path = r'{os.path.abspath(test_path)}'")
    
    print("\n" + "=" * 40)
    print("Next steps:")
    print("1. If dataset found: Update your main script with the correct path")
    print("2. If not found: Download from Kaggle and extract")
    print("3. Run your main script again")
    print("=" * 40)# Simple Food Recognition Model - Fallback Solution
# Use this if the main EfficientNet model has compatibility issues

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class SimpleFoodModel:
    """Simple CNN model for food recognition - no transfer learning"""
    
    def __init__(self, dataset_path, img_size=(224, 224)):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.model = None
        self.class_names = []
        
        # Calorie database (same as main model)
        self.calorie_db = self._create_calorie_database()
    
    def _create_calorie_database(self):
        """Create calorie database for Food-101 classes"""
        calorie_data = {
            'apple_pie': 237, 'baby_back_ribs': 292, 'baklava': 428, 'beef_carpaccio': 135,
            'beef_tartare': 196, 'beet_salad': 88, 'beignets': 347, 'bibimbap': 121,
            'bread_pudding': 212, 'breakfast_burrito': 188, 'bruschetta': 195, 'caesar_salad': 158,
            'cannoli': 301, 'caprese_salad': 125, 'carrot_cake': 385, 'ceviche': 134,
            'cheese_plate': 368, 'cheesecake': 321, 'chicken_curry': 128, 'chicken_quesadilla': 218,
            'chicken_wings': 203, 'chocolate_cake': 371, 'chocolate_mousse': 168, 'churros': 367,
            'clam_chowder': 112, 'club_sandwich': 282, 'crab_cakes': 197, 'creme_brulee': 296,
            'croque_madame': 235, 'cup_cakes': 305, 'deviled_eggs': 155, 'donuts': 452,
            'dumplings': 241, 'edamame': 121, 'eggs_benedict': 238, 'escargots': 90,
            'falafel': 333, 'filet_mignon': 277, 'fish_and_chips': 232, 'foie_gras': 462,
            'french_fries': 365, 'french_onion_soup': 67, 'french_toast': 222, 'fried_calamari': 175,
            'fried_rice': 163, 'frozen_yogurt': 127, 'garlic_bread': 350, 'gnocchi': 131,
            'greek_salad': 150, 'grilled_cheese_sandwich': 291, 'grilled_salmon': 231, 'guacamole': 160,
            'gyoza': 201, 'hamburger': 295, 'hot_and_sour_soup': 56, 'hot_dog': 290,
            'huevos_rancheros': 149, 'hummus': 166, 'ice_cream': 207, 'lasagna': 135,
            'lobster_bisque': 110, 'lobster_roll_sandwich': 436, 'macaroni_and_cheese': 164, 'macarons': 407,
            'miso_soup': 40, 'mussels': 86, 'nachos': 346, 'omelette': 154,
            'onion_rings': 331, 'oysters': 68, 'pad_thai': 153, 'paella': 139,
            'pancakes': 227, 'panna_cotta': 133, 'peking_duck': 337, 'pho': 46,
            'pizza': 266, 'pork_chop': 231, 'poutine': 365, 'prime_rib': 291,
            'pulled_pork_sandwich': 227, 'ramen': 436, 'ravioli': 175, 'red_velvet_cake': 478,
            'risotto': 142, 'samosa': 308, 'sashimi': 127, 'scallops': 69,
            'seaweed_salad': 45, 'shrimp_and_grits': 149, 'spaghetti_bolognese': 151, 'spaghetti_carbonara': 174,
            'spring_rolls': 140, 'steak': 271, 'strawberry_shortcake': 227, 'sushi': 142,
            'tacos': 226, 'takoyaki': 112, 'tiramisu': 240, 'tuna_tartare': 144,
            'waffles': 291
        }
        return calorie_data
    
    def load_data(self):
        """Load dataset with simple approach"""
        print("Loading Food-101 dataset...")
        
        # Get class names from directory structure
        images_path = os.path.join(self.dataset_path, 'images')
        self.class_names = sorted([d for d in os.listdir(images_path) 
                                 if os.path.isdir(os.path.join(images_path, d))])
        
        print(f"Found {len(self.class_names)} food classes")
        
        # Create data generators
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            validation_split=0.2
        )
        
        self.train_generator = datagen.flow_from_directory(
            images_path,
            target_size=self.img_size,
            batch_size=16,
            class_mode='categorical',
            subset='training'
        )
        
        self.validation_generator = datagen.flow_from_directory(
            images_path,
            target_size=self.img_size,
            batch_size=16,
            class_mode='categorical',
            subset='validation'
        )
        
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.validation_generator.samples}")
    
    def build_simple_model(self):
        """Build a simple CNN model from scratch"""
        print("Building simple CNN model...")
        
        model = tf.keras.Sequential([
            # Input layer
            layers.Input(shape=(*self.img_size, 3)),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Global average pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print(" Simple model built successfully!")
        print(f"Total parameters: {model.count_params():,}")
        model.summary()
    
    def train(self, epochs=20):
        """Train the model"""
        print("Starting training...")
        
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-7)
        ]
        
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.validation_generator,
            callbacks=callbacks
        )
        
        return history
    
    def predict_food_and_calories(self, image_path, portion_size=100):
        """Predict food and calories"""
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get food class name
        predicted_food = list(self.train_generator.class_indices.keys())[predicted_class_idx]
        
        # Estimate calories
        base_calories = self.calorie_db.get(predicted_food, 200)
        estimated_calories = (base_calories * portion_size) / 100
        
        return {
            'food_class': predicted_food,
            'confidence': confidence,
            'calories_per_100g': base_calories,
            'portion_size_g': portion_size,
            'estimated_calories': estimated_calories
        }
    
    def save_model(self, filepath):
        """Save the model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

# Usage function
def run_simple_model():
    """Run the simple model as fallback"""
    dataset_path = r"D:\Internships_tasks\prodigy_internship\task5\food-101"
    
    # Initialize simple model
    simple_model = SimpleFoodModel(dataset_path)
    
    # Load data
    simple_model.load_data()
    
    # Build model
    simple_model.build_simple_model()
    
    # Train model
    history = simple_model.train(epochs=15)
    
    # Save model
    simple_model.save_model('simple_food_model.h5')
    
    print(" Simple model training completed!")
    return simple_model

if __name__ == "__main__":
    print(" Running Simple Food Recognition Model")
    print("This is a fallback solution without transfer learning")
    print("=" * 50)
    
    model = run_simple_model()
    print("Model ready for predictions!")