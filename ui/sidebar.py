# ui/sidebar.py
# Description: Functions to build and manage the Streamlit sidebar.

import streamlit as st
from typing import Tuple
from config import CONFIG, DEFAULTS
from core.models import UserProfile, AdvancedSettings

def _create_unified_input(field_name, field_config, container=st.sidebar):
    """Helper to create input widgets using unified configuration."""
    session_key = f'user_{field_name}'
    
    if field_config['type'] == 'number':
        if field_config.get('advanced'):
            default_val = DEFAULTS.get(field_name, 0)
            display_val = int(default_val * 100) if field_name == 'fat_percentage' else default_val
            placeholder = f"Default: {display_val}"
        else:
            placeholder = field_config.get('placeholder')

        value = container.number_input(
            field_config['label'], min_value=field_config['min'], max_value=field_config['max'],
            value=st.session_state[session_key], step=field_config['step'],
            placeholder=placeholder, help=field_config.get('help')
        )
    elif field_config['type'] == 'selectbox':
        current_value = st.session_state[session_key]
        if field_name == 'activity_level':
            options = field_config['options']
            index = next((i for i, (_, val) in enumerate(options) if val == current_value), 0)
            selection = container.selectbox(field_config['label'], options, index=index, format_func=lambda x: x[0])
            value = selection[1]
        else:
            options = field_config['options']
            index = options.index(current_value) if current_value in options else 0
            value = container.selectbox(field_config['label'], options, index=index)
    
    st.session_state[session_key] = value
    return value

def render_sidebar() -> Tuple[UserProfile, AdvancedSettings, bool]:
    """Renders the entire sidebar and returns user input models."""
    st.sidebar.header("Personal Parameters 📊")
    
    all_inputs = {}
    containers = {'standard': st.sidebar, 'advanced': st.sidebar.expander("Advanced Settings ⚙️")}

    for field_name, field_config in CONFIG['form_fields'].items():
        container = containers['advanced'] if field_config.get('advanced', False) else containers['standard']
        value = _create_unified_input(field_name, field_config, container=container)
        if 'convert' in field_config:
            value = field_config['convert'](value)
        all_inputs[field_name] = value

    # Create UserProfile from inputs, using defaults for missing values
    user_profile = UserProfile(
        age=all_inputs['age'] or DEFAULTS['age'],
        height_cm=all_inputs['height_cm'] or DEFAULTS['height_cm'],
        weight_kg=all_inputs['weight_kg'] or DEFAULTS['weight_kg'],
        sex=all_inputs['sex'] if all_inputs['sex'] != "Select Sex" else DEFAULTS['sex'],
        activity_level=all_inputs['activity_level'] or DEFAULTS['activity_level']
    )

    # Create AdvancedSettings from inputs, using defaults for missing values
    advanced_settings = AdvancedSettings(
        caloric_surplus=all_inputs['caloric_surplus'] or DEFAULTS['caloric_surplus'],
        protein_per_kg=all_inputs['protein_per_kg'] or DEFAULTS['protein_per_kg'],
        fat_percentage=all_inputs['fat_percentage'] or DEFAULTS['fat_percentage']
    )

    # Check if all required fields have been filled by the user
    required_fields = [f for f, c in CONFIG['form_fields'].items() if c.get('required')]
    user_has_entered_info = all(
        (all_inputs[field] is not None and all_inputs[field] != CONFIG['form_fields'][field].get('placeholder'))
        for field in required_fields
    )

    # Add informational sections
    info_sections = [
        {'title': "Activity Level Guide 🏃‍♂️", 'content': "- **Sedentary**: Little to no exercise.\n- **Lightly Active**: Light exercise 1-3 days/week.\n- **Moderately Active**: Moderate exercise 3-5 days/week.\n- **Very Active**: Hard exercise 6-7 days/week.\n- **Extremely Active**: Very hard exercise, physical job."},
        {'title': "Emoji Guide 💡", 'content': "- 🥇 **Superfood**: Top-tier in multiple categories.\n- 💥 **Nutrient & Calorie Dense**: High in both.\n- 🔥 **High-Calorie**: Energy-dense.\n- 💪/🍚/🥑/🥦 **Top Source**: Protein/Carbs/Fat/Micros."},
        {'title': "About This Calculator 📖", 'content': "Uses Mifflin-St Jeor for BMR, adds a caloric surplus for weight gain (target 0.25% body weight/week), and follows evidence-based macronutrient targets."}
    ]
    for section in info_sections:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### {section['title']}")
        st.sidebar.markdown(section['content'])

    return user_profile, advanced_settings, user_has_entered_info
