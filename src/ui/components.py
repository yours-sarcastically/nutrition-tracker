"""UI components for the nutrition tracker."""

import streamlit as st
import logging
from typing import Dict, List, Any, Optional, Tuple
from ..models import NutritionTargets, FoodItem, DailyIntake, NutritionProgress, SelectedFood
from ..config import NutritionConfig
from ..session_manager import SessionManager

logger = logging.getLogger(__name__)

class UIComponents:
    """Reusable UI components for the nutrition tracker."""
    
    def __init__(self, config: NutritionConfig = None, session_manager: SessionManager = None):
        """Initialize UI components."""
        self.config = config or NutritionConfig()
        self.session_manager = session_manager or SessionManager(config)
    
    def setup_page_config(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Personalized Nutrition Tracker",
            page_icon="🍽️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Hide input instructions
        st.markdown(
            """<style>[data-testid="InputInstructions"] {display: none;}</style>""",
            unsafe_allow_html=True
        )
    
    def render_header(self) -> None:
        """Render the main application header."""
        st.title("Personalized Evidence-Based Nutrition Tracker 🍽️")
        st.markdown(
            "Ready to turbocharge your health game? This awesome tool dishes out daily "
            "nutrition goals made just for you and makes tracking meals as easy as pie. "
            "Let's get those macros on your team! 🚀"
        )
    
    def setup_sidebar(self) -> Dict[str, Any]:
        """
        Handle all user input widgets in the sidebar.
        
        Returns:
            Dictionary containing all user inputs and validation status
        """
        st.sidebar.header("Personal Parameters for Daily Target Calculation 📊")
        
        # Get current defaults
        defaults = self.session_manager.get_user_input_defaults()
        
        # Personal Information Section
        age = st.sidebar.number_input(
            "Age (Years)",
            *self.config.UI_CONFIG['age_range'],
            value=defaults['age'],
            placeholder="Enter your age"
        )
        
        height_cm = st.sidebar.number_input(
            "Height (Centimeters)",
            *self.config.UI_CONFIG['height_range'],
            value=defaults['height'],
            placeholder="Enter your height"
        )
        
        weight_kg = st.sidebar.number_input(
            "Weight (kg)",
            *self.config.UI_CONFIG['weight_range'],
            value=defaults['weight'],
            step=0.5,
            placeholder="Enter your weight"
        )
        
        # Gender selection
        sex_options = ["Select Sex"] + self.config.UI_CONFIG['gender_options']
        sex_index = 0
        if defaults['sex'] in sex_options:
            sex_index = sex_options.index(defaults['sex'])
        
        sex = st.sidebar.selectbox("Sex", sex_options, index=sex_index)
        
        # Activity level selection
        activity_options = self.config.UI_CONFIG['activity_options']
        activity_labels = list(activity_options.keys())
        activity_values = list(activity_options.values())
        
        activity_index = 0
        if defaults['activity'] in activity_values:
            activity_index = activity_values.index(defaults['activity'])
        
        activity_selection = st.sidebar.selectbox("Activity Level", activity_labels, index=activity_index)
        activity_level = activity_options[activity_selection]
        
        # Update session state
        self.session_manager.update_user_data(
            age=age, height=height_cm, weight=weight_kg, sex=sex, activity=activity_level
        )
        
        # Advanced Settings
        advanced_settings = self._render_advanced_settings()
        
        # Validation
        user_has_entered_info = all([
            age, height_cm, weight_kg,
            sex != "Select Sex",
            activity_level is not None
        ])
        
        # Render sidebar footer
        self._render_sidebar_footer()
        
        return {
            "age": age or self.config.DEFAULT_USER['age'],
            "height_cm": height_cm or self.config.DEFAULT_USER['height_cm'],
            "weight_kg": weight_kg or self.config.DEFAULT_USER['weight_kg'],
            "sex": sex if sex != "Select Sex" else self.config.DEFAULT_USER['gender'],
            "activity_level": activity_level or self.config.DEFAULT_USER['activity_level'],
            "user_has_entered_info": user_has_entered_info,
            **advanced_settings
        }
    
    def _render_advanced_settings(self) -> Dict[str, Any]:
        """Render advanced settings in sidebar expander."""
        with st.sidebar.expander("Advanced Settings ⚙️"):
            caloric_surplus = st.number_input(
                "Caloric Surplus (kcal)",
                *self.config.UI_CONFIG['caloric_surplus_range'],
                value=None,
                step=50,
                placeholder=f"Default: {self.config.NUTRITION_CONSTANTS['caloric_surplus']}",
                help="Additional calories above maintenance for weight gain."
            )
            
            protein_per_kg = st.number_input(
                "Protein (g/kg)",
                *self.config.UI_CONFIG['protein_range'],
                value=None,
                step=0.1,
                placeholder=f"Default: {self.config.NUTRITION_CONSTANTS['protein_per_kg']}",
                help="Protein intake per kilogram of body weight."
            )
            
            fat_percentage_input = st.number_input(
                "Fat (% of Calories)",
                *self.config.UI_CONFIG['fat_percentage_range'],
                value=None,
                step=1,
                placeholder=f"Default: {int(self.config.NUTRITION_CONSTANTS['fat_percentage'] * 100)}",
                help="Percentage of total calories from fat."
            )
        
        return {
            "caloric_surplus": caloric_surplus or self.config.NUTRITION_CONSTANTS['caloric_surplus'],
            "protein_per_kg": protein_per_kg or self.config.NUTRITION_CONSTANTS['protein_per_kg'],
            "fat_percentage": (fat_percentage_input / 100) if fat_percentage_input else self.config.NUTRITION_CONSTANTS['fat_percentage']
        }
    
    def _render_sidebar_footer(self) -> None:
        """Render informational content in sidebar footer."""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Activity Level Guide for Accurate TDEE 🏃‍♂️")
        st.sidebar.markdown(
            "- **Sedentary**: Little to no exercise.\n"
            "- **Lightly Active**: Light exercise 1-3 days/week.\n"
            "- **Moderately Active**: Moderate exercise 3-5 days/week.\n"
            "- **Very Active**: Hard exercise 6-7 days/week.\n"
            "- **Extremely Active**: Very hard exercise/physical job."
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Emoji Guide for Food Ranking 💡")
        st.sidebar.markdown(
            "- 🥇 **Superfood**: High in multiple nutrients.\n"
            "- 💥 **Nutrient & Calorie Dense**: High in both.\n"
            "- 🔥 **High-Calorie**: Energy-dense.\n"
            "- 💪 **Top Protein**\n"
            "- 🍚 **Top Carb**\n"
            "- 🥑 **Top Fat**\n"
            "- 🥦 **Top Micronutrient**"
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### About This Nutrition Calculator 📖")
        st.sidebar.markdown(f"""
        - **BMR**: Mifflin-St Jeor equation.
        - **Protein**: {self.config.NUTRITION_CONSTANTS['protein_per_kg']} g/kg of body weight.
        - **Fat**: {int(self.config.NUTRITION_CONSTANTS['fat_percentage'] * 100)}% of total calories.
        - **Weight Gain Target**: {self.config.NUTRITION_CONSTANTS['target_weekly_gain_rate'] * 100}% of body weight/week.
        """)
    
    def display_dashboard(self, targets: NutritionTargets, user_has_entered_info: bool) -> None:
        """
        Display the personalized targets and metabolic information.
        
        Args:
            targets: Calculated nutrition targets
            user_has_entered_info: Whether user has entered complete information
        """
        if not user_has_entered_info:
            st.info("👈 Please enter your personal information in the sidebar to view your daily nutritional targets.")
            st.header("Sample Daily Targets for Reference 🎯")
            st.caption("These are example targets. Enter your information in the sidebar for personalized calculations.")
        else:
            st.header("Your Personalized Daily Nutritional Targets for Healthy Weight Gain 🎯")
        
        # Metabolic and Weight Gain Information
        col1, col2, col3, _ = st.columns(4)
        with col1:
            st.metric("Basal Metabolic Rate (BMR)", f"{targets.bmr} kcal/day")
        with col2:
            st.metric("Total Daily Energy Expenditure (TDEE)", f"{targets.tdee} kcal/day")
        with col3:
            st.metric("Est. Weekly Weight Gain", f"{targets.target_weight_gain_per_week} kg/week")
        
        # Nutritional Targets
        st.subheader("Daily Nutritional Target Breakdown")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Daily Calorie Target", f"{targets.total_calories} kcal")
        with col2:
            st.metric("Protein Target", f"{targets.protein_g} g")
        with col3:
            st.metric("Carbohydrate Target", f"{targets.carb_g} g")
        with col4:
            st.metric("Fat Target", f"{targets.fat_g} g")
        
        # Macronutrient Distribution
        st.subheader("Macronutrient Distribution as a Percent of Daily Calories")
        percentages = targets.get_macronutrient_percentages()
        
        col1, col2, col3, _ = st.columns(4)
        with col1:
            st.metric("Protein", f"{percentages['protein']:.1f}%", f"{targets.protein_calories} kcal")
        with col2:
            st.metric("Carbohydrates", f"{percentages['carbs']:.1f}%", f"{targets.carb_calories} kcal")
        with col3:
            st.metric("Fat", f"{percentages['fat']:.1f}%", f"{targets.fat_calories} kcal")
        
        st.markdown("---")
    
    def display_food_item(self, food: FoodItem, category: str, col) -> None:
        """
        Create the UI for a single food item in a given column.
        
        Args:
            food: Food item to display
            category: Category the food belongs to
            col: Streamlit column to render in
        """
        with col:
            st.subheader(f"{food.emoji} {food.name}")
            key_prefix = f"{category}_{food.name}"
            current_serving = self.session_manager.get_food_selections().get(food.name, 0.0)
            
            # Quick selection buttons
            button_cols = st.columns(5)
            for k in range(1, 6):
                with button_cols[k-1]:
                    button_type = "primary" if current_serving == float(k) else "secondary"
                    if st.button(f"{k}", key=f"{key_prefix}_{k}", type=button_type, use_container_width=True):
                        self.session_manager.update_food_selection(food.name, float(k))
                        st.rerun()
            
            # Custom serving input
            custom_serving = st.number_input(
                "Custom Servings:",
                0.0, 10.0,
                value=float(current_serving),
                step=0.1,
                key=f"{key_prefix}_custom"
            )
            
            if custom_serving != current_serving:
                self.session_manager.update_food_selection(food.name, custom_serving)
                st.rerun()
            
            # Nutritional information
            st.caption(
                f"Per Serving: {food.calories} kcal | {food.protein}g protein | "
                f"{food.carbs}g carbs | {food.fat}g fat"
            )
    
    def create_food_log_ui(self, foods_by_category: Dict[str, List[FoodItem]]) -> None:
        """
        Create the interactive tabs for food selection.
        
        Args:
            foods_by_category: Dictionary of foods organized by category
        """
        st.header("Select Foods and Log Servings for Today 📝")
        st.markdown("Choose foods using the buttons for preset servings or enter a custom serving amount for each item.")
        
        available_categories = sorted([cat for cat, items in foods_by_category.items() if items])
        
        if not available_categories:
            st.error("No food categories available. Please check your data file.")
            return
        
        tabs = st.tabs(available_categories)
        
        for i, category in enumerate(available_categories):
            with tabs[i]:
                # Sort foods by emoji priority and calories
                sorted_items = sorted(
                    foods_by_category[category],
                    key=lambda x: (self.config.EMOJI_ORDER.get(x.emoji, 4), -x.calories)
                )
                
                # Display foods in two columns
                for j in range(0, len(sorted_items), 2):
                    col1, col2 = st.columns(2)
                    
                    if j < len(sorted_items):
                        self.display_food_item(sorted_items[j], category, col1)
                    
                    if j + 1 < len(sorted_items):
                        self.display_food_item(sorted_items[j + 1], category, col2)
        
        st.markdown("---")
    
    def display_calculation_results(self, daily_intake: DailyIntake, targets: NutritionTargets) -> None:
        """
        Display calculation results and progress tracking.
        
        Args:
            daily_intake: User's daily food intake
            targets: Nutrition targets to compare against
        """
        st.header("Summary of Your Daily Nutritional Intake 📊")
        
        if not daily_intake.selected_foods:
            st.info("No foods have been selected for today. 🍽️")
            return
        
        # Display selected foods
        st.subheader("Foods Logged for Today 🥣")
        cols = st.columns(3)
        for i, selected_food in enumerate(daily_intake.selected_foods):
            with cols[i % 3]:
                st.write(f"• {selected_food.food.emoji} {selected_food.food.name} × {selected_food.servings:.1f}")
        
        # Total nutritional intake
        total_nutrition = daily_intake.total_nutrition
        st.subheader("Total Nutritional Intake for the Day 📈")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Calories", f"{total_nutrition.calories:.0f} kcal")
        with col2:
            st.metric("Total Protein", f"{total_nutrition.protein:.1f} g")
        with col3:
            st.metric("Total Carbohydrates", f"{total_nutrition.carbs:.1f} g")
        with col4:
            st.metric("Total Fat", f"{total_nutrition.fat:.1f} g")
        
        # Progress tracking
        progress = NutritionProgress(targets, total_nutrition)
        self._display_progress_bars(progress)
        
        # Recommendations
        self._display_recommendations(progress)
        
        # Detailed breakdown
        self._display_detailed_breakdown(daily_intake)
    
    def _display_progress_bars(self, progress: NutritionProgress) -> None:
        """Display progress bars for each nutrient."""
        st.subheader("Progress Toward Daily Targets 🎯")
        
        progress_data = progress.get_all_progress()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Calories**")
            st.progress(min(progress_data['calories'] / 100, 1.0))
            st.write(f"{progress_data['calories']:.1f}% of target")
            
            st.write("**Carbohydrates**")
            st.progress(min(progress_data['carbs'] / 100, 1.0))
            st.write(f"{progress_data['carbs']:.1f}% of target")
        
        with col2:
            st.write("**Protein**")
            st.progress(min(progress_data['protein'] / 100, 1.0))
            st.write(f"{progress_data['protein']:.1f}% of target")
            
            st.write("**Fat**")
            st.progress(min(progress_data['fat'] / 100, 1.0))
            st.write(f"{progress_data['fat']:.1f}% of target")
    
    def _display_recommendations(self, progress: NutritionProgress) -> None:
        """Display personalized recommendations based on progress."""
        st.subheader("Personalized Recommendations 💡")
        
        progress_data = progress.get_all_progress()
        recommendations = []
        
        if progress_data['calories'] < 80:
            recommendations.append("🔥 You need more calories! Consider adding calorie-dense foods like nuts, oils, or dried fruits.")
        
        if progress_data['protein'] < 80:
            recommendations.append("💪 Increase protein intake with legumes, dairy, or protein-rich grains.")
        
        if progress_data['carbs'] < 80:
            recommendations.append("🍚 Add more carbohydrates with whole grains, fruits, or starchy vegetables.")
        
        if progress_data['fat'] < 80:
            recommendations.append("🥑 Include healthy fats from avocados, nuts, seeds, or olive oil.")
        
        if progress_data['calories'] > 120:
            recommendations.append("⚠️ You're exceeding your calorie target. Consider smaller portions or lower-calorie alternatives.")
        
        if not recommendations:
            recommendations.append("🎉 Great job! You're on track with your nutritional targets.")
        
        for rec in recommendations:
            st.write(f"- {rec}")
    
    def _display_detailed_breakdown(self, daily_intake: DailyIntake) -> None:
        """Display detailed nutritional breakdown by food item."""
        st.subheader("Detailed Nutritional Breakdown by Food Item 📋")
        
        breakdown_data = []
        for selected_food in daily_intake.selected_foods:
            nutrition = selected_food.nutrition
            breakdown_data.append({
                "Food": selected_food.food.name,
                "Servings": f"{selected_food.servings:.1f}",
                "Calories": f"{nutrition.calories:.0f}",
                "Protein (g)": f"{nutrition.protein:.1f}",
                "Carbs (g)": f"{nutrition.carbs:.1f}",
                "Fat (g)": f"{nutrition.fat:.1f}"
            })
        
        if breakdown_data:
            import pandas as pd
            df = pd.DataFrame(breakdown_data)
            st.dataframe(df, use_container_width=True)
    
    def render_action_buttons(self) -> Tuple[bool, bool]:
        """
        Render action buttons and return their states.
        
        Returns:
            Tuple of (calculate_clicked, clear_clicked)
        """
        col1, col2 = st.columns(2)
        
        with col1:
            calculate_clicked = st.button(
                "Calculate Daily Intake",
                type="primary",
                use_container_width=True
            )
        
        with col2:
            clear_clicked = st.button(
                "Clear All Selections",
                use_container_width=True
            )
        
        return calculate_clicked, clear_clicked
