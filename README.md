# COVID-19 Risk Factor Analysis 

## Summary 
Using the [CORD-19](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks)
dataset, we plan to use NLP/NLU methods to identify risk factors for COVID-19. Time permitting, 
we will compare risk factors discussed in research with those mentioned 
in the main-stream media. 

We will make 3 passes through the corpus: 

1. Train a language model on the CORD-19 research papers. 
2. Store the `h_t` (hidden state/prediction pt of last layer) for each manually defined risk factor (below) 
like so: `{'Diabetes': [.342, ..., -.123], 'Smoking': [0.323, ..., -0.432]}`
3. Compare all `h_t`'s in the data with our stored set of `h_t`'s
using some similarity metric in order to try to identify new risk factors. 

Between steps 2-3, we might do clustering analysis on the stored 
`h_t`'s to see how well they group. Can also PCA down to 2 dims 
and visualize. 

## Manually Identified Risk Factors 
[CDC Guidance on Risk Factors](https://www.cdc.gov/coronavirus/2019-ncov/need-extra-precautions/people-at-higher-risk.html)
* 65 years and older
* Living in a nursing home or long-term care facility
* Chronic lung disease or moderate to severe asthma 
* Immunocomprimised:
    - Cancer treatment
    - Smoking 
    - Bone marrow transplantation 
    - Immune deficiencies 
    - Pooly controlled HIV or AIDS
    - Prolonged use of corticosteroids 
    - "Other immune weaking medication"
* Severe obesity (BMI >= 40)
* Diabetes 
* Chronic kidney disease undergoing dialysis 
* Liver disease 
* Hemoglobin disorders 
    - Sickle cell disease 
    - Thalassemia 
* Serious heart conditions 
    - Heart failure 
    - Coronary artery disease 
    - Congenital heart disease 
    - Cardiomyopathies 
    - Pulmonary hypertension 