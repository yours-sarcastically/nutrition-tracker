"""Tests for the nutrition calculator."""

import pytest
from src.models import UserProfile
from src.calculator import NutritionCalculator
from src.config import NutritionConfig

class TestNutritionCalculator:
    """Test cases for NutritionCalculator class."""
    
    @pytest.fixture
    def calculator(self):
        """Create calculator instance for testing."""
        return NutritionCalculator()
    
    @pytest.fixture
    def sample_profile(self):
        """Create sample user profile for testing."""
        return UserProfile(
            age=26,
            height_cm=180,
            weight_kg=70,
            gender="Male",
            activity_level="moderately_active"
        )
    
    def test_bmr_calculation_male(self, calculator, sample_profile):
        """Test BMR calculation for male user."""
        bmr = calculator.calculate_bmr(sample_profile)
        
        # Expected BMR for male: (10 * 70) + (6.25 * 180) - (5 * 26) + 5 = 1700
        expected_bmr = (10 * 70) + (6.25 * 180) - (5 * 26) + 5
        assert bmr == expected_bmr
        assert bmr > 0
    
    def test_bmr_calculation_female(self, calculator):
        """Test BMR calculation for female user."""
        female_profile = UserProfile(
            age=26,
            height_cm=165,
            weight_kg=60,
            gender="Female",
            activity_level="moderately_active"
        )
        
        bmr = calculator.calculate_bmr(female_profile)
        
        # Expected BMR for female: (10 * 60) + (6.25 * 165) - (5 * 26) - 161 = 1370.25
        expected_bmr = (10 * 60) + (6.25 * 165) - (5 * 26) - 161
        assert bmr == expected_bmr
        assert bmr > 0
    
    def test_tdee_calculation(self, calculator, sample_profile):
        """Test TDEE calculation."""
        bmr = calculator.calculate_bmr(sample_profile)
        tdee = calculator.calculate_tdee(bmr, sample_profile.activity_level)
        
        # For moderately active, multiplier is 1.55
        expected_tdee = bmr * 1.55
        assert tdee == expected_tdee
        assert tdee > bmr
    
    def test_calculate_targets(self, calculator, sample_profile):
        """Test complete target calculation."""
        targets = calculator.calculate_targets(sample_profile)
        
        # Basic validation
        assert targets.bmr > 0
        assert targets.tdee > targets.bmr
        assert targets.total_calories > targets.tdee
        assert targets.protein_g > 0
        assert targets.fat_g > 0
        assert targets.carb_g > 0
        
        # Check that calories add up
        total_macro_calories = (
            targets.protein_calories + 
            targets.fat_calories + 
            targets.carb_calories
        )
        assert abs(total_macro_calories - targets.total_calories) <= 1  # Allow for rounding
    
    def test_validate_targets(self, calculator, sample_profile):
        """Test target validation."""
        targets = calculator.calculate_targets(sample_profile)
        assert calculator.validate_targets(targets) is True

    def test_invalid_activity_level(self, calculator):
        """Test handling of invalid activity level."""
        invalid_profile = UserProfile(
            age=26,
            height_cm=180,
            weight_kg=70,
            gender="Male",
            activity_level="invalid_level"
        )
        
        # Should use default multiplier (1.55 for moderately active)
        bmr = calculator.calculate_bmr(invalid_profile)
        tdee = calculator.calculate_tdee(bmr, invalid_profile.activity_level)
        
        expected_tdee = bmr * 1.55  # Default multiplier
        assert tdee == expected_tdee
    
    def test_custom_parameters(self, calculator, sample_profile):
        """Test calculation with custom parameters."""
        targets = calculator.calculate_targets(
            profile=sample_profile,
            caloric_surplus=500,
            protein_per_kg=2.5,
            fat_percentage=0.30
        )
        
        # Check custom parameters are applied
        expected_protein = 2.5 * sample_profile.weight_kg
        assert targets.protein_g == round(expected_protein)
        
        expected_fat_calories = targets.total_calories * 0.30
        assert abs(targets.fat_calories - expected_fat_calories) <= 1
