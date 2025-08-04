"""Tests for the food database."""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from src.database import FoodDatabase
from src.config import NutritionConfig

class TestFoodDatabase:
    """Test cases for FoodDatabase class."""
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing."""
        return """name,category,calories,protein,carbs,fat,serving_unit
Lentils,PRIMARY PROTEIN SOURCES,230,18,40,0.8,1 cup
Brown Rice,PRIMARY CARBOHYDRATE SOURCES,216,5,45,1.8,1 cup
Almonds,PRIMARY FAT SOURCES,164,6,6,14,1 oz
Spinach,PRIMARY MICRONUTRIENT SOURCES,7,0.9,1.1,0.1,1 cup"""
    
    @pytest.fixture
    def temp_csv_file(self, sample_csv_data):
        """Create temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(sample_csv_data)
            temp_path = Path(f.name)
        
        yield temp_path
        
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()
    
    def test_database_initialization(self, temp_csv_file):
        """Test database initialization with valid CSV."""
        db = FoodDatabase(temp_csv_file)
        
        assert len(db.foods) > 0
        assert "PRIMARY PROTEIN SOURCES" in db.foods
        assert "PRIMARY CARBOHYDRATE SOURCES" in db.foods
        assert "PRIMARY FAT SOURCES" in db.foods
        assert "PRIMARY MICRONUTRIENT SOURCES" in db.foods
    
    def test_food_item_creation(self, temp_csv_file):
        """Test that food items are created correctly."""
        db = FoodDatabase(temp_csv_file)
        
        protein_foods = db.foods["PRIMARY PROTEIN SOURCES"]
        assert len(protein_foods) == 1
        
        lentils = protein_foods[0]
        assert "Lentils" in lentils.name
        assert "(1 cup)" in lentils.name
        assert lentils.calories == 230
        assert lentils.protein == 18
        assert lentils.carbs == 40
        assert lentils.fat == 0.8
    
    def test_emoji_assignment(self, temp_csv_file):
        """Test that emojis are assigned to foods."""
        db = FoodDatabase(temp_csv_file)
        
        # Check that at least some foods have emojis
        all_foods = []
        for category_foods in db.foods.values():
            all_foods.extend(category_foods)
        
        foods_with_emojis = [f for f in all_foods if f.emoji]
        assert len(foods_with_emojis) > 0
    
    def test_get_available_categories(self, temp_csv_file):
        """Test getting available categories."""
        db = FoodDatabase(temp_csv_file)
        categories = db.get_available_categories()
        
        expected_categories = [
            "PRIMARY CARBOHYDRATE SOURCES",
            "PRIMARY FAT SOURCES", 
            "PRIMARY MICRONUTRIENT SOURCES",
            "PRIMARY PROTEIN SOURCES"
        ]
        
        assert sorted(categories) == sorted(expected_categories)
    
    def test_get_sorted_foods_by_category(self, temp_csv_file):
        """Test getting sorted foods by category."""
        db = FoodDatabase(temp_csv_file)
        protein_foods = db.get_sorted_foods_by_category("PRIMARY PROTEIN SOURCES")
        
        assert len(protein_foods) == 1
        assert "Lentils" in protein_foods[0].name
    
    def test_search_foods(self, temp_csv_file):
        """Test food search functionality."""
        db = FoodDatabase(temp_csv_file)
        
        # Search for lentils
        results = db.search_foods("lentils")
        assert len(results) == 1
        assert "Lentils" in results[0].name
        
        # Search for non-existent food
        results = db.search_foods("pizza")
        assert len(results) == 0
    
    def test_get_food_by_name(self, temp_csv_file):
        """Test getting specific food by name."""
        db = FoodDatabase(temp_csv_file)
        
        # Find the exact name format
        lentils_name = None
        for foods in db.foods.values():
            for food in foods:
                if "Lentils" in food.name:
                    lentils_name = food.name
                    break
        
        assert lentils_name is not None
        
        # Test successful retrieval
        food = db.get_food_by_name(lentils_name)
        assert "Lentils" in food.name
        
        # Test food not found
        with pytest.raises(ValueError):
            db.get_food_by_name("Non-existent Food")
    
    def test_invalid_csv_file(self):
        """Test handling of invalid CSV file."""
        with pytest.raises(FileNotFoundError):
            FoodDatabase(Path("non_existent_file.csv"))
    
    def test_missing_columns(self):
        """Test handling of CSV with missing columns."""
        invalid_csv = """name,category,calories
Lentils,PRIMARY PROTEIN SOURCES,230"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(invalid_csv)
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Missing required columns"):
                FoodDatabase(temp_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
