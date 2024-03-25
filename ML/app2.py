# import streamlit as st
# import pickle
# import pandas as pd

# # Load the trained model from the pickle file
# with open('trained_model.pkl', 'rb') as file:
#     model = pickle.load(file)

# def predict_expenses(json_input):
#     # Convert JSON input to DataFrame
#     query_df = pd.DataFrame([json_input])

#     # Use the loaded model to make predictions
#     prediction = model.predict(query_df)

#     return prediction.tolist()

# def main():
#     st.title('Person financial status Prediction')

#     # JSON input field
#     # json_input = st.text_area("Enter JSON input")
    
    
#         # Get inputs from the user
#     age = st.text_input("Age", key="age", type="default")
#     sex = st.text_input("Sex {0-male 1-female}", key="sex", type="default")
#     bmi = st.text_input("BMI", key="bmi", type="default")
#     children = st.text_input("Children", key="children", type="default")
#     smoker = st.text_input("Smoker {0-NO 1-YES}", key="smoker", type="default")
#     expenses = st.text_input("Expenses", key="expenses", type="default")
#     salary = st.text_input("Salary", key="salary", type="default")

#     # Convert inputs to floats
#     try:
#         age = float(age) if age else None
#         sex = float(sex) if sex else None
#         bmi = float(bmi) if bmi else None
#         children = float(children) if children else None
#         smoker = float(smoker) if smoker else None
#         expenses = float(expenses) if expenses else None
#         salary = float(salary) if salary else None

#         # Calculate difference if both salary and expenses are provided
#         difference = salary - expenses if expenses is not None and salary is not None else None

#         # Create JSON object
#         json_input = {
#             "age": age,
#             "sex": sex,
#             "bmi": bmi,
#             "children": children,
#             "smoker": smoker,
#             "region": 0,  # Include the appropriate region value based on your dataset
#             "expenses": expenses,
#             "Salary": salary,
#             "Difference": difference
#         }
#     except ValueError:
#         st.error("Please enter valid numeric values.")
       
#     st.title('0-POOR')
#     st.title('1-Middle class')
#     st.title('2-Rich')


#     if st.button('Predict'):
#         try:
#             # Parse JSON input
#             json_data = eval(json_input)
#             prediction = predict_expenses(json_data)
#             st.write("Prediction:", prediction)
#         except Exception as e:
#             st.error(f"An error occurred: {e}. Please ensure your input is in valid JSON format.")

# if __name__ == "__main__":
#     main()

import streamlit as st
import pickle
import pandas as pd

# Load the trained model from the pickle file
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

def predict_expenses(json_input):
    # Convert JSON input to DataFrame
    query_df = pd.DataFrame([json_input])

    # Use the loaded model to make predictions
    prediction = model.predict(query_df)

    return prediction.tolist()

def main():
    st.title('Person financial status Prediction')

    # Get inputs from the user
    age = st.text_input("Age", key="age", type="default")
    sex = st.text_input("Sex {0-male 1-female}", key="sex", type="default")
    # sex = 0
    bmi = st.text_input("BMI", key="bmi", type="default")
    children = st.text_input("Children", key="children", type="default")
    smoker = st.text_input("Smoker {0-NO     1-YES}", key="smoker", type="default")
    expenses = st.text_input("Expenses", key="expenses", type="default")
    salary = st.text_input("Salary", key="salary", type="default")

    # Convert inputs to floats
    try:
        age = float(age) if age else None
        sex = float(sex) if sex else None
        bmi = float(bmi) if bmi else None
        children = float(children) if children else None
        smoker = float(smoker) if smoker else None
        expenses = float(expenses) if expenses else None
        salary = float(salary) if salary else None

        # Calculate difference if both salary and expenses are provided
        difference = salary - expenses if expenses is not None and salary is not None else None

        # Create JSON object
        json_input = {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": 0,  # Include the appropriate region value based on your dataset
            "expenses": expenses,
            "Salary": salary,
            "Difference": difference
        }
    except ValueError:
        st.error("Please enter valid numeric values.")
        return

    # st.title('0 - Financial planning is poor')
    # st.title('1 - Need more work on Finance')
    # st.title('2 - Great! You have good knowlege about finance, keep it up')
    
    
    css_styles = """
    <style>
        .normal-text {
            font-size: 16px;  /* Adjust font size as needed */
            font-family: 'Arial', sans-serif;  /* Change font family as needed */
        }
    </style>
    """

    # Render CSS styles
    st.markdown(css_styles, unsafe_allow_html=True)

    # Display titles with normal text style
    st.title('0 - Financial planning is poor')
    st.title('1 - Need more work on Finance')
    st.title('2 - Great! You have good knowledge about finance, keep it up')
    

    if st.button('Predict'):
        try:
            prediction = predict_expenses(json_input)
            st.write("Prediction:", prediction)
        except Exception as e:
            st.error(f"An error occurred: {e}. Please ensure your input is in valid JSON format.")

if __name__ == "__main__":
    main()
