# Refactored code for the "Your Food Choices Today" expander
st.subheader("What You've Logged")

# Reuse the existing helper function to get calculated food data
prepared_data = prepare_summary_data(totals, targets, selected_foods)
consumed_foods_list = prepared_data['consumed_foods']

# Reformat the data for display in the DataFrame
if consumed_foods_list:
    display_data = [
        {
            'Food': item['name'],
            'Servings': f"{item['servings']:.1f}",
            'Calories (kcal)': f"{item['calories']:.0f}",
            'Protein (g)': f"{item['protein']:.1f}",
            'Carbs (g)': f"{item['carbs']:.1f}",
            'Fat (g)': f"{item['fat']:.1f}"
        } for item in consumed_foods_list
    ]
    df_summary = pd.DataFrame(display_data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
else:
    st.caption("No foods logged yet.")
