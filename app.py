from flask import Flask, request, render_template
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import os
import re
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

app = Flask(__name__)

llm_resto = OpenAI(temperature=0.7, max_tokens=1000)

prompt_template_resto = PromptTemplate(
    input_variables=['age', 'gender', 'weight', 'height', 'veg_or_nonveg', 'disease', 'region', 'allergies',
                     'foodtype'],
    template="""As a diet and wellness expert, provide recommendations based on:

Age: {age}
Gender: {gender}
Weight: {weight} kg
Height: {height} m
Diet: {veg_or_nonveg}
Health: {disease}
Region: {region}
Allergies: {allergies}
Food Type: {foodtype}

Provide exactly:
1. Six restaurant recommendations
2. Six breakfast ideas
3. Five dinner suggestions
4. Six workout recommendations

Use this exact format:

RESTAURANTS
1. [Name] - [Brief description]
2. [Name] - [Brief description]
3. [Name] - [Brief description]
4. [Name] - [Brief description]
5. [Name] - [Brief description]
6. [Name] - [Brief description]

BREAKFAST IDEAS
1. [Meal] - [Brief description]
2. [Meal] - [Brief description]
3. [Meal] - [Brief description]
4. [Meal] - [Brief description]
5. [Meal] - [Brief description]
6. [Meal] - [Brief description]

DINNER SUGGESTIONS
1. [Meal] - [Brief description]
2. [Meal] - [Brief description]
3. [Meal] - [Brief description]
4. [Meal] - [Brief description]
5. [Meal] - [Brief description]

WORKOUT RECOMMENDATIONS
1. [Exercise] - [Brief description]
2. [Exercise] - [Brief description]
3. [Exercise] - [Brief description]
4. [Exercise] - [Brief description]
5. [Exercise] - [Brief description]
6. [Exercise] - [Brief description]

Keep descriptions concise and relevant to the person's details."""
)


def parse_recommendations(text):
    logger.debug(f"Raw LLM response:\n{text}")

    categories = {
        'RESTAURANTS': [],
        'BREAKFAST IDEAS': [],
        'DINNER SUGGESTIONS': [],
        'WORKOUT RECOMMENDATIONS': []
    }

    current_category = None

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Check if this line is a category header
        upper_line = line.upper()
        if any(category.upper() in upper_line for category in categories.keys()):
            current_category = next(cat for cat in categories.keys() if cat.upper() in upper_line)
            continue

        # Check if line starts with a number and contains a description
        if current_category and re.match(r'^\d+\.', line):
            # Remove the number and any leading/trailing whitespace
            item = re.sub(r'^\d+\.\s*', '', line).strip()
            if item:
                categories[current_category].append(item)

    return categories


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        input_data = {
            'age': request.form['age'],
            'gender': request.form['gender'],
            'weight': request.form['weight'],
            'height': request.form['height'],
            'veg_or_nonveg': request.form['veg_or_nonveg'],
            'disease': request.form['disease'],
            'region': request.form['region'],
            'allergies': request.form['allergies'],
            'foodtype': request.form['foodtype']
        }

        logger.debug(f"Input data: {input_data}")

        try:
            chain_resto = LLMChain(llm=llm_resto, prompt=prompt_template_resto)
            results = chain_resto.run(input_data)

            parsed_results = parse_recommendations(results)

            # Map the category names to the template variables
            category_mapping = {
                'RESTAURANTS': 'restaurant_names',
                'BREAKFAST IDEAS': 'breakfast_names',
                'DINNER SUGGESTIONS': 'dinner_names',
                'WORKOUT RECOMMENDATIONS': 'workout_names'
            }

            template_data = {
                template_var: parsed_results[category]
                for category, template_var in category_mapping.items()
            }

            return render_template('result.html', **template_data)
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            return render_template('error.html', error=str(e))

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)